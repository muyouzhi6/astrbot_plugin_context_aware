"""Pure capability decisions for GIF previews and opt-in video input."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from typing import Mapping


def video_family(model="", provider_id=""):
    model = str(model or "").casefold()
    # A routing prefix/Provider ID must not change a recognized actual model.
    for name in (model.rsplit("/", 1)[-1], model, str(provider_id or "").casefold()):
        if "gemini" in name:
            return "gemini"
        if re.search(r"qwen(?:[\d.]+)?[-_](?:vl|omni)", name):
            return "qwen"
        if re.search(r"glm[-_]4(?:\.\d+)?v", name):
            return "glm"
        if re.search(r"doubao[-_][^ /]*(?:vision|seed)", name):
            return "doubao"
    return ""


def _matches(patterns, names):
    if not isinstance(patterns, (list, tuple)):
        return False
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        pattern = pattern.strip().casefold()
        if any(
            fnmatchcase(name, pattern) if any(c in pattern for c in "*?[")
            else pattern in name
            for name in names if name
        ):
            return True
    return False


def detect_video_capability(model="", provider_id="", modalities=None, config=None):
    """Return none/frames/native; unknown provider information defaults to frames.

    Explicit global override wins, then lists (none > frames > native on overlap).
    Empty modalities is an explicit absence of image support, unlike missing data.
    """
    config = config if isinstance(config, Mapping) else {}
    override = config.get("video_capability_override", "auto")
    if override in ("native", "frames", "none"):
        return override
    names = (str(model or "").casefold(), str(provider_id or "").casefold())
    for capability, key in (
        ("none", "video_no_vision_models"),
        ("frames", "video_frames_only_models"),
        ("native", "video_native_models"),
    ):
        if _matches(config.get(key), names):
            return capability
    # Schema default 'auto' must not conceal persisted pre-3.8 video_mode.
    legacy = config.get("video_mode", "auto")
    if legacy in ("native", "frames", "none"):
        return legacy
    if isinstance(modalities, (list, tuple, set)):
        modes = {str(value).casefold() for value in modalities}
        if "image" not in modes:
            return "none"
        if "video" in modes:
            return "native"
    return "native" if video_family(model, provider_id) else "frames"
