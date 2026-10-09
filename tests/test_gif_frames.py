from __future__ import annotations

import io
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from gif_frames import GifOptions, prepare_gif, render_frames


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "file"])
@pytest.mark.parametrize("no_vision", [False, True])
async def test_nested_quoted_gif_enters_request_without_core_image_urls(tmp_path, kind, no_vision):
    from test_llm_image_compression import FakeCompressionEvent, FakeContext, load_plugin_module

    module = load_plugin_module()
    path = tmp_path / "quoted.gif"
    path.write_bytes(gif_bytes())
    plugin = module.Main(FakeContext(), {
        "enable": False, "image_cache_dir": str(tmp_path / "cache"),
        "gif_mode": "frames", "video_capability_override": "none" if no_vision else "frames",
    })
    plugin._image_compress_output_dir = str(tmp_path / "output")
    component = module.File("quoted.gif", file=str(path)) if kind == "file" else module.Image.fromFileSystem(str(path))
    inner, outer = module.Reply(), module.Reply()
    inner.chain = [component]
    outer.chain = [inner]
    event = FakeCompressionEvent()
    event.get_messages = lambda: [outer]
    req = SimpleNamespace(image_urls=[], extra_user_content_parts=[])
    try:
        await plugin.on_llm_request(event, req)
        if no_vision:
            assert not req.image_urls
            assert any(p.text == "[动图]" for p in req.extra_user_content_parts)
        else:
            assert len(req.image_urls) == 3
            assert all(ref.endswith(".png") for ref in req.image_urls)
            assert any("3帧" in p.text for p in req.extra_user_content_parts)
            first = list(req.image_urls)
            await plugin.on_llm_request(event, req)
            assert req.image_urls == first
        assert path.read_bytes().startswith(b"GIF")
        if kind == "file":
            assert component.get_file_calls == 0
    finally:
        await plugin.terminate()


def gif_bytes(
    colors=("red", "green", "blue"), durations=None, size=(64, 48), transparent=False
):
    frames = []
    try:
        for i, color in enumerate(colors):
            image = Image.new(
                "RGBA" if transparent else "RGB",
                size,
                (0, 0, 0, 0) if transparent else color,
            )
            if transparent:
                ImageDraw.Draw(image).rectangle((i * 5, 5, i * 5 + 15, 25), fill=color)
            frames.append(image)
        out = io.BytesIO()
        frames[0].save(
            out,
            "GIF",
            save_all=True,
            append_images=frames[1:],
            duration=durations or [100] * len(frames),
            loop=0,
            disposal=2,
            optimize=False,
        )
        return out.getvalue()
    finally:
        for image in frames:
            image.close()


def test_grid_timeline_and_labels():
    result = prepare_gif(gif_bytes())
    assert result.indices == (0, 1, 2)
    assert result.times == pytest.approx((0, 0.1, 0.2))
    assert len(result.images) == 1 and "3帧" in result.note
    with Image.open(io.BytesIO(result.images[0][0])) as image:
        assert image.getpixel((10, 30)) == (255, 0, 0)
        assert image.getpixel((74, 30)) == (0, 128, 0)
        assert image.getpixel((10, 98)) == (0, 0, 255)
        assert image.crop((0, 0, 64, 20)).getextrema()[0][0] < 255


def test_duration_weighted_midpoints_and_order():
    result = prepare_gif(gif_bytes(durations=[100, 800, 100]), GifOptions(max_frames=4))
    assert result.indices == (1,)  # All four time-bin midpoints lie in the held frame.
    result = prepare_gif(
        gif_bytes(
            colors=("red", "green", "blue", "yellow"), durations=[100, 200, 300, 400]
        ),
        GifOptions(max_frames=3),
    )
    assert result.indices == (1, 2, 3)
    assert result.times == pytest.approx((0.1, 0.3, 0.6))


def test_near_duplicates_keep_order_and_repeated_later_scene():
    raw = gif_bytes(colors=((100, 0, 0), (101, 0, 0), (0, 255, 0), (100, 0, 0)))
    result = prepare_gif(raw, GifOptions(mode="frames"))
    assert result.indices == (0, 2, 3)
    assert len(result.images) == 3
    assert prepare_gif(raw, GifOptions(dedup_threshold=0)).indices == (0, 1, 2, 3)


def test_transparency_and_disposal_are_composited():
    result = prepare_gif(gif_bytes(transparent=True))
    assert len(result.indices) == 3
    with Image.open(io.BytesIO(result.images[0][0])) as image:
        assert image.getpixel((50, 50)) == (255, 255, 255)
        assert image.getpixel((64 + 7, 30)) == (0, 128, 0)
        assert image.getpixel((64 + 1, 30)) == (255, 255, 255)


def test_frame_decode_limit_falls_back_to_first():
    result = prepare_gif(gif_bytes(), GifOptions(max_decode_frames=2))
    assert result.reason == "gif_first_frame_fallback"
    assert result.indices == (0,)
    assert "首帧" in result.note
    with Image.open(io.BytesIO(result.images[0][0])) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_timeout_and_encoder_error_recover():
    with patch("gif_frames.render_frames", side_effect=TimeoutError):
        assert prepare_gif(gif_bytes()).reason == "gif_first_frame_fallback"
    with patch("gif_frames.render_frames", side_effect=OSError):
        assert prepare_gif(gif_bytes()).reason == "gif_first_frame_fallback"
    # Deadline expires before frame scan, while the separate recovery deadline remains valid.
    with patch("gif_frames.time.monotonic", side_effect=[0, 10, 10, 10]):
        assert prepare_gif(gif_bytes()).reason == "gif_first_frame_fallback"


def test_source_caps_reject_before_decode():
    raw = gif_bytes(size=(2048, 1024))
    assert not prepare_gif(raw, GifOptions(max_source_pixels=1000)).images
    assert not prepare_gif(raw, GifOptions(max_source_bytes=len(raw) - 1)).images
    # A forged huge logical screen is small on disk.
    huge = bytearray(gif_bytes())
    huge[6:10] = (65535).to_bytes(2, "little") * 2
    assert not prepare_gif(bytes(huge)).images


def test_pixels_bytes_frame_edge_caps_for_grid_and_frames():
    images = [Image.effect_noise((400, 300), 100).convert("RGB") for _ in range(6)]
    try:
        for mode in ("grid", "frames"):
            options = GifOptions(
                mode=mode,
                frame_max_edge=160,
                max_total_pixels=30000,
                max_output_bytes=12000,
            )
            result = render_frames(images, list(range(6)), options)
            assert sum(len(raw) for raw, _ in result) <= 12000
            pixels = 0
            for raw, _ in result:
                with Image.open(io.BytesIO(raw)) as image:
                    pixels += image.width * image.height
                    if mode == "frames":
                        assert image.width <= 160 and image.height <= 160
            assert pixels <= 30000
    finally:
        for image in images:
            image.close()


def test_static_gif_has_no_motion_note_and_non_gif_bytes_untouched():
    result = prepare_gif(gif_bytes(colors=("red",)))
    assert not result.animated and not result.note
    assert result.reason == "gif_static"
    for fmt in ("PNG", "JPEG", "WEBP"):
        with Image.new("RGB", (30, 20), "red") as image:
            out = io.BytesIO()
            image.save(out, fmt)
        assert prepare_gif(out.getvalue()).reason == "not_gif"
    assert not prepare_gif(b"GIF89a broken").images
    assert prepare_gif(b"GIF89a").reason == "decode_failed"


def test_explicit_first_frame_and_invalid_config():
    result = prepare_gif(gif_bytes(), GifOptions(mode="first_frame"))
    assert result.reason == "gif_first_frame" and result.indices == (0,)
    options = GifOptions.from_mapping(
        {
            "gif_mode": "oops",
            "gif_max_frames": float("nan"),
            "gif_frame_max_edge": -1,
            "gif_timeout_sec": None,
        }
    )
    assert options.mode == "grid" and options.max_frames == 6
    assert options.frame_max_edge == 64 and options.timeout_sec == 5
    assert prepare_gif(gif_bytes(), replace(options, max_output_bytes=1)).images == ()


def test_animated_webp_and_apng_keep_existing_compression_behavior(tmp_path):
    from image_compression import ImageCompressionOptions, compress_local_image

    for fmt, suffix in (("WEBP", ".webp"), ("PNG", ".png")):
        images = [Image.new("RGB", (32, 24), color) for color in ("red", "blue")]
        source = tmp_path / ("animated" + suffix)
        try:
            images[0].save(
                source,
                fmt,
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )
        finally:
            for image in images:
                image.close()
        original = source.read_bytes()
        outcome = compress_local_image(
            str(source), str(tmp_path), ImageCompressionOptions(enabled=True)
        )
        assert not outcome.changed and outcome.reason == "animated_image"
        assert source.read_bytes() == original


def test_single_frame_transparency_is_preserved():
    result = prepare_gif(gif_bytes(colors=("red",), transparent=True))
    with Image.open(io.BytesIO(result.images[0][0])) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((50, 40))[3] == 0


@pytest.mark.parametrize("frame,left,top,width,height", [
    (1, 100, 100, 32, 32),
    (1, 0, 0, 132, 132),
    (1, 65535, 0, 32, 32),
    (0, 100, 100, 32, 32),
])
def test_offset_or_descriptor_growth_rejected_before_pillow(frame, left, top, width, height):
    raw = bytearray(gif_bytes(size=(32, 32)))
    descriptor = b"\x2c\0\0\0\0\x20\0\x20\0"
    offsets = []
    start = 0
    for _ in range(3):
        start = raw.index(descriptor, start)
        offsets.append(start)
        start += len(descriptor)
    offset = offsets[frame] + 1
    raw[offset:offset + 8] = b"".join(v.to_bytes(2, "little") for v in (left, top, width, height))
    with patch("gif_frames.Image.open", side_effect=AssertionError("must reject before allocation")) as opened:
        result = prepare_gif(bytes(raw), GifOptions(max_source_pixels=4096))
    opened.assert_not_called()
    assert result.reason == "source_pixels" and not result.images


def test_canvas_growth_accumulates_across_later_frames():
    raw = bytearray(gif_bytes(size=(32, 32)))
    descriptor = b"\x2c\0\0\0\0\x20\0\x20\0"
    second = raw.index(descriptor, raw.index(descriptor) + len(descriptor))
    third = raw.index(descriptor, second + len(descriptor))
    raw[second + 1:second + 3] = (64).to_bytes(2, "little")
    raw[third + 3:third + 5] = (64).to_bytes(2, "little")
    assert prepare_gif(bytes(raw), GifOptions(max_source_pixels=4096)).reason == "source_pixels"
