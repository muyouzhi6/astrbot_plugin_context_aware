"""Bounded GIF timeline sampling and reusable numbered contact sheets.

Call the synchronous functions in a worker thread. They return bytes, never
create files, and check a monotonic deadline between bounded decode operations.
"""

from __future__ import annotations

import bisect
import io
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image, ImageChops, ImageDraw, ImageStat

MIB = 1024 * 1024


def number(raw, key, default, low, high):
    try:
        value = float(raw.get(key, default))
        if not math.isfinite(value):
            return default
        return min(high, max(low, value))
    except (ValueError, TypeError, OverflowError):
        return default


@dataclass(frozen=True)
class GifOptions:
    mode: str = "grid"
    max_frames: int = 6
    frame_max_edge: int = 512
    max_total_pixels: int = 2_000_000
    max_output_bytes: int = 2 * MIB
    max_source_bytes: int = 20 * MIB
    max_source_pixels: int = 25_000_000
    max_decode_frames: int = 300
    timeout_sec: float = 5
    dedup_threshold: float = 2.0

    @classmethod
    def from_mapping(cls, raw: Any):
        raw = raw if isinstance(raw, Mapping) else {}
        mode = raw.get("gif_mode", "grid")
        return cls(
            mode=mode if mode in ("grid", "frames", "first_frame") else "grid",
            max_frames=int(number(raw, "gif_max_frames", 6, 1, 16)),
            frame_max_edge=int(number(raw, "gif_frame_max_edge", 512, 64, 2048)),
            max_total_pixels=int(
                number(raw, "gif_max_total_pixels", 2_000_000, 4096, 16_000_000)
            ),
            max_output_bytes=int(
                number(raw, "gif_max_output_bytes", 2 * MIB, 1024, 20 * MIB)
            ),
            max_source_bytes=int(
                number(raw, "gif_max_source_bytes", 20 * MIB, 1024, 100 * MIB)
            ),
            max_source_pixels=int(
                number(raw, "gif_max_source_pixels", 25_000_000, 4096, 25_000_000)
            ),
            max_decode_frames=int(number(raw, "gif_max_decode_frames", 300, 2, 1000)),
            timeout_sec=number(raw, "gif_timeout_sec", 5, 0.1, 30),
            dedup_threshold=number(raw, "gif_dedup_threshold", 2, 0, 30),
        )


@dataclass(frozen=True)
class FrameResult:
    images: tuple[tuple[bytes, str], ...] = ()
    indices: tuple[int, ...] = ()
    times: tuple[float, ...] = ()
    reason: str = "unchanged"
    animated: bool = False

    @property
    def note(self):
        if not self.images or not self.animated:
            return ""
        if self.reason in ("gif_first_frame", "gif_first_frame_fallback"):
            return "[动图：已提取首帧]"
        layout = (
            "拼成网格（从左到右、从上到下）" if len(self.images) == 1 else "，分别附图"
        )
        return f"[动图：已按时间顺序抽取{len(self.indices)}帧{layout}]"


def _check(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("frame processing deadline")


def _rgb(image, edge):
    # Pillow applies GIF disposal/compositing during seek/load.
    with image.convert("RGBA") as rgba:
        rgba.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background


def _different(left, right, threshold):
    with left.resize((32, 32)) as a, right.resize((32, 32)) as b:
        with ImageChops.difference(a, b) as diff:
            return sum(ImageStat.Stat(diff).mean) / 3 > threshold


def render_frames(frames, times, options: GifOptions, *, deadline=None):
    """Encode a grid or independent images, enforcing aggregate pixel/byte caps."""
    deadline = (
        deadline if deadline is not None else time.monotonic() + options.timeout_sec
    )
    if not frames:
        return ()
    frames = frames[: options.max_frames]
    edge = options.frame_max_edge
    while edge >= 8:
        _check(deadline)
        tiles = []
        canvases = []
        try:
            for i, frame in enumerate(frames):
                tile = _rgb(frame, max(8, edge - 20))
                # Labels consume pixels too; reserve them inside each tile.
                canvas = Image.new("RGB", (tile.width, tile.height + 20), "white")
                canvas.paste(tile, (0, 20))
                tile.close()
                ImageDraw.Draw(canvas).text(
                    (3, 3), f"{i + 1}  {times[i]:.2f}s", fill="black"
                )
                tiles.append(canvas)
            if options.mode == "frames":
                canvases = tiles
            else:
                cols = math.ceil(math.sqrt(len(tiles)))
                rows = math.ceil(len(tiles) / cols)
                w, h = max(t.width for t in tiles), max(t.height for t in tiles)
                grid = Image.new("RGB", (cols * w, rows * h), "white")
                for i, tile in enumerate(tiles):
                    grid.paste(tile, ((i % cols) * w, (i // cols) * h))
                canvases = [grid]
            pixels = sum(c.width * c.height for c in canvases)
            if pixels <= options.max_total_pixels:
                for fmt, quality in (
                    ("PNG", 0),
                    ("JPEG", 85),
                    ("JPEG", 65),
                    ("JPEG", 45),
                ):
                    encoded = []
                    for canvas in canvases:
                        _check(deadline)
                        out = io.BytesIO()
                        canvas.save(
                            out,
                            fmt,
                            **(
                                {"compress_level": 6}
                                if fmt == "PNG"
                                else {"quality": quality}
                            ),
                        )
                        encoded.append(
                            (
                                out.getvalue(),
                                "image/png" if fmt == "PNG" else "image/jpeg",
                            )
                        )
                    if sum(len(b) for b, _ in encoded) <= options.max_output_bytes:
                        return tuple(encoded)
            edge = min(edge - 1, int(edge * 0.75))
        finally:
            for img in {id(c): c for c in tiles + canvases}.values():
                img.close()
    raise ValueError("frame output exceeds limits")


class _SourcePixelsError(ValueError):
    pass


def _check_gif_extents(data, options, frame_limit):
    """Inspect descriptors before Pillow seek can allocate disposal canvases."""
    if len(data) < 13:
        raise ValueError("truncated GIF header")
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")

    def check_size():
        if width * height > options.max_source_pixels:
            raise _SourcePixelsError("GIF canvas pixels")

    check_size()
    offset = 13 + (3 * (2 ** ((data[10] & 7) + 1)) if data[10] & 128 else 0)
    frames = 0
    while offset < len(data):
        marker = data[offset]
        offset += 1
        if marker == 0x3B:
            return
        if marker == 0x21:
            offset += 1  # Extension label, followed by data sub-blocks.
        elif marker == 0x2C:
            if offset + 9 > len(data):
                raise ValueError("truncated GIF descriptor")
            left, top, w, h = (
                int.from_bytes(data[offset + i:offset + i + 2], "little")
                for i in (0, 2, 4, 6)
            )
            width, height = max(width, left + w), max(height, top + h)
            check_size()
            frames += 1
            if frames >= frame_limit:
                return
            flags = data[offset + 8]
            offset += 9 + (3 * (2 ** ((flags & 7) + 1)) if flags & 128 else 0)
            offset += 1  # LZW minimum code size.
        else:
            raise ValueError("invalid GIF block")
        while True:
            if offset >= len(data):
                raise ValueError("truncated GIF data")
            size = data[offset]
            offset += 1 + size
            if not size:
                break


def _first_frame(data, options, animated, reason):
    # A separate short deadline allows first-frame recovery after sampling timeout.
    _check_gif_extents(data, options, 1)
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > options.max_source_pixels:
            return FrameResult(reason="source_pixels")
        image.seek(0)
        with image.convert("RGBA") as frame:
            # Preserve the old plain first-frame PNG, without a numbered header.
            edge = options.frame_max_edge
            deadline = time.monotonic() + options.timeout_sec
            while edge >= 8:
                _check(deadline)
                frame.thumbnail((edge, edge))
                out = io.BytesIO()
                frame.save(out, "PNG", compress_level=6)
                if (
                    len(out.getvalue()) <= options.max_output_bytes
                    and frame.width * frame.height <= options.max_total_pixels
                ):
                    return FrameResult(
                        ((out.getvalue(), "image/png"),), (0,), (0.0,), reason, animated
                    )
                edge = int(edge * 0.75)
    return FrameResult(reason="output_limit")


def prepare_gif(data: bytes, options: GifOptions | None = None) -> FrameResult:
    options = options or GifOptions()
    if not data.startswith((b"GIF87a", b"GIF89a")):
        return FrameResult(reason="not_gif")
    # Oversized input/dimensions are never decoded, including the recovery path.
    if len(data) > options.max_source_bytes:
        return FrameResult(reason="source_bytes")
    frames = []
    try:
        _check_gif_extents(data, options, options.max_decode_frames + 1)
        deadline = time.monotonic() + options.timeout_sec
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > options.max_source_pixels:
                return FrameResult(reason="source_pixels")
            try:
                image.seek(1)
            except EOFError:
                return _first_frame(data, options, False, "gif_static")
            if options.mode == "first_frame":
                return _first_frame(data, options, True, "gif_first_frame")
            image.seek(0)
            ends = []
            total = 0.0
            # Never use n_frames: Pillow scans the entire file to compute it.
            for index in range(options.max_decode_frames + 1):
                _check(deadline)
                try:
                    image.seek(index)
                except EOFError:
                    break
                if image.width * image.height > options.max_source_pixels:
                    raise _SourcePixelsError("GIF canvas pixels")
                if index == options.max_decode_frames:
                    raise ValueError("decode frame limit")
                duration = image.info.get("duration", 100)
                total += max(10, min(float(duration or 100), 60000)) / 1000
                ends.append(total)
            count = min(options.max_frames, len(ends))
            # Midpoints of equal time bins: long held frames get proportional weight.
            targets = sorted(
                {
                    bisect.bisect_right(ends, total * (i + 0.5) / count)
                    for i in range(count)
                }
            )
            indices, times = [], []
            image.seek(0)
            for index in targets:
                _check(deadline)
                image.seek(index)
                frame = _rgb(image, options.frame_max_edge)
                if frames and not _different(
                    frames[-1], frame, options.dedup_threshold
                ):
                    frame.close()
                    continue
                frames.append(frame)
                indices.append(index)
                times.append(0.0 if index == 0 else ends[index - 1])
            encoded = render_frames(frames, times, options, deadline=deadline)
            return FrameResult(
                encoded, tuple(indices), tuple(times), "gif_" + options.mode, True
            )
    except _SourcePixelsError:
        return FrameResult(reason="source_pixels")
    except Exception:
        try:
            return _first_frame(data, options, True, "gif_first_frame_fallback")
        except Exception:
            return FrameResult(reason="decode_failed")
    finally:
        for frame in frames:
            frame.close()
