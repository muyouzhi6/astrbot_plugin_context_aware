from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image as PILImage
from PIL import ImageOps


MIB = 1024 * 1024
LANCZOS = getattr(PILImage, "Resampling", PILImage).LANCZOS


def _clamp(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


@dataclass(frozen=True, slots=True)
class ImageCompressionOptions:
    enabled: bool = False
    min_size_bytes: int = 4 * MIB
    max_edge: int = 2048
    quality: int = 90
    min_quality: int = 75
    max_output_bytes: int = 2 * MIB
    max_input_bytes: int = 50 * MIB
    download_retries: int = 3
    download_timeout: int = 15

    @classmethod
    def from_mapping(cls, raw: Any) -> "ImageCompressionOptions":
        config: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        quality = int(_clamp(config.get("quality"), 90, 50, 100))
        min_quality = int(
            _clamp(config.get("min_quality"), 75, 40, quality)
        )
        max_input_mb = _clamp(config.get("max_input_size_mb"), 50, 1, 200)
        max_output_mb = _clamp(
            config.get("max_output_size_mb"),
            2,
            0.1,
            max_input_mb,
        )
        return cls(
            enabled=bool(config.get("enable", False)),
            min_size_bytes=int(
                _clamp(config.get("min_size_mb"), 4, 0.1, 100) * MIB
            ),
            max_edge=int(_clamp(config.get("max_edge"), 2048, 512, 8192)),
            quality=quality,
            min_quality=min_quality,
            max_output_bytes=int(max_output_mb * MIB),
            max_input_bytes=int(max_input_mb * MIB),
            download_retries=int(
                _clamp(config.get("download_retries"), 3, 1, 5)
            ),
            download_timeout=int(
                _clamp(config.get("download_timeout"), 15, 5, 120)
            ),
        )


@dataclass(frozen=True, slots=True)
class CompressionOutcome:
    source_path: str
    output_path: str
    source_bytes: int
    output_bytes: int
    source_size: tuple[int, int] | None
    output_size: tuple[int, int] | None
    changed: bool
    reason: str


def _unchanged(
    source_path: str,
    *,
    source_bytes: int = 0,
    source_size: tuple[int, int] | None = None,
    reason: str,
) -> CompressionOutcome:
    return CompressionOutcome(
        source_path=source_path,
        output_path=source_path,
        source_bytes=source_bytes,
        output_bytes=source_bytes,
        source_size=source_size,
        output_size=source_size,
        changed=False,
        reason=reason,
    )


def _has_alpha(image: PILImage.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )


def _quality_candidates(quality: int, minimum: int) -> list[int]:
    candidates = list(range(quality, minimum - 1, -5))
    if not candidates or candidates[-1] != minimum:
        candidates.append(minimum)
    return list(dict.fromkeys(candidates))


def compress_local_image(
    source_path: str,
    output_dir: str,
    options: ImageCompressionOptions,
) -> CompressionOutcome:
    """Create a provider-facing image copy without modifying the source file."""
    path = Path(source_path)
    if not path.is_file():
        return _unchanged(source_path, reason="source_missing")

    try:
        source_bytes = path.stat().st_size
    except OSError:
        return _unchanged(source_path, reason="source_stat_failed")

    if source_bytes > options.max_input_bytes:
        return _unchanged(
            source_path,
            source_bytes=source_bytes,
            reason="source_too_large",
        )

    candidate_path: Path | None = None
    working_image: PILImage.Image | None = None
    try:
        with PILImage.open(path) as opened_image:
            source_size = opened_image.size
            if getattr(opened_image, "is_animated", False) or getattr(
                opened_image,
                "n_frames",
                1,
            ) > 1:
                return _unchanged(
                    source_path,
                    source_bytes=source_bytes,
                    source_size=source_size,
                    reason="animated_image",
                )

            if (
                source_bytes < options.min_size_bytes
                and max(source_size) <= options.max_edge
            ):
                return _unchanged(
                    source_path,
                    source_bytes=source_bytes,
                    source_size=source_size,
                    reason="below_threshold",
                )

            working_image = ImageOps.exif_transpose(opened_image)
            working_image.load()
            alpha = _has_alpha(working_image)
            suffix = ".png" if alpha else ".jpg"

            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            candidate_path = destination / (
                f"context-aware-compressed-{uuid.uuid4().hex}{suffix}"
            )

            current_max_edge = max(working_image.size)
            target_edge = min(current_max_edge, options.max_edge)
            minimum_edge = min(512, target_edge)
            output_size: tuple[int, int] | None = None
            output_bytes = source_bytes

            while True:
                resized = working_image.copy()
                try:
                    if max(resized.size) > target_edge:
                        resized.thumbnail(
                            (target_edge, target_edge),
                            LANCZOS,
                        )

                    if alpha:
                        if resized.mode != "RGBA":
                            converted = resized.convert("RGBA")
                            resized.close()
                            resized = converted
                        resized.save(
                            candidate_path,
                            "PNG",
                            optimize=True,
                            compress_level=6,
                        )
                        output_bytes = candidate_path.stat().st_size
                    else:
                        if resized.mode != "RGB":
                            converted = resized.convert("RGB")
                            resized.close()
                            resized = converted
                        for quality in _quality_candidates(
                            options.quality,
                            options.min_quality,
                        ):
                            resized.save(
                                candidate_path,
                                "JPEG",
                                quality=quality,
                                optimize=True,
                                progressive=True,
                            )
                            output_bytes = candidate_path.stat().st_size
                            if output_bytes <= options.max_output_bytes:
                                break

                    output_size = resized.size
                finally:
                    resized.close()

                if output_bytes <= options.max_output_bytes:
                    break
                if target_edge <= minimum_edge:
                    break
                next_edge = max(minimum_edge, int(target_edge * 0.85))
                if next_edge == target_edge:
                    break
                target_edge = next_edge

            if not candidate_path.exists():
                return _unchanged(
                    source_path,
                    source_bytes=source_bytes,
                    source_size=source_size,
                    reason="output_missing",
                )

            output_bytes = candidate_path.stat().st_size
            if output_bytes >= source_bytes:
                candidate_path.unlink(missing_ok=True)
                return _unchanged(
                    source_path,
                    source_bytes=source_bytes,
                    source_size=source_size,
                    reason="not_smaller",
                )

            return CompressionOutcome(
                source_path=source_path,
                output_path=str(candidate_path),
                source_bytes=source_bytes,
                output_bytes=output_bytes,
                source_size=source_size,
                output_size=output_size,
                changed=True,
                reason=(
                    "compressed"
                    if output_bytes <= options.max_output_bytes
                    else "compressed_above_target"
                ),
            )
    except Exception as exc:  # noqa: BLE001
        if candidate_path is not None:
            try:
                candidate_path.unlink(missing_ok=True)
            except OSError:
                pass
        return _unchanged(
            source_path,
            source_bytes=source_bytes,
            reason=f"error:{type(exc).__name__}",
        )
    finally:
        if working_image is not None:
            try:
                working_image.close()
            except Exception:
                pass
