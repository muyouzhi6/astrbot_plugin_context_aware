from model_capability import detect_video_capability, video_family

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("name,family", [
    ("newapigemini/gemini-3.8-flash", "gemini"),
    ("vendor/QWEN-VL-PLUS", "qwen"),
    ("Qwen/Qwen2.5-VL-72B-Instruct", "qwen"),
    ("qwen2-vl", "qwen"), ("qwen3-vl-flash", "qwen"),
    ("qwen-omni", "qwen"), ("Qwen3-Omni-30B", "qwen"),
    ("glm-4v-plus-0111", "glm"), ("vendor/GLM-4.5V", "glm"),
    ("glm-4.6v-flash", "glm"),
    ("Doubao-1.5-vision-pro", "doubao"),
    ("doubao-seed-1-6-250615", "doubao"),
])
def test_native_names_and_provider_ids(name, family):
    assert video_family(name) == family
    assert detect_video_capability(name, modalities=["text", "image", "tool_use"]) == "native"
    assert detect_video_capability("ep-123", name, ["image"]) == "native"
    assert detect_video_capability(name, modalities=["text", "tool_use"]) == "none"


@pytest.mark.parametrize("model,modalities,expected", [
    ("gpt-4o", ["image"], "frames"),
    ("gpt-4o", ["text"], "none"),
    ("gemini-3-flash", [], "none"),
    ("gemini-3-flash", ["video"], "native"),
    ("unknown", ["text", "image", "video"], "native"),
    ("unknown", None, "frames"), ("", None, "frames"),
    ("grok-vision", ["image"], "frames"),
    ("qwen3-32b", ["image"], "frames"),
    ("glm-4-plus", ["image"], "frames"),
    ("doubao-pro", ["image"], "frames"),
    ("unknown", "invalid", "frames"),
])
def test_modalities_and_safe_defaults(model, modalities, expected):
    assert detect_video_capability(model, modalities=modalities) == expected


def test_lists_substrings_wildcards_overlap_and_invalid_values():
    assert video_family("qwen3-vl", "newapigemini") == "qwen"
    assert video_family("newapigemini/qwen3-vl") == "qwen"
    assert video_family("glm-4.5v", "qwen-vl-provider") == "glm"
    assert detect_video_capability("ep-123", "Ark/Prod", ["text"], {"video_native_models": ["ark/*"]}) == "native"
    assert detect_video_capability("vendor/GEMINI-3", config={"video_frames_only_models": ["gemini"]}) == "frames"
    assert detect_video_capability("qwen3-vl-flash", config={"video_no_vision_models": ["*vl-f?ash"]}) == "none"
    config = {"video_native_models": ["model"], "video_frames_only_models": ["model"], "video_no_vision_models": ["model"]}
    assert detect_video_capability("model", config=config) == "none"
    del config["video_no_vision_models"]
    assert detect_video_capability("model", config=config) == "frames"
    assert detect_video_capability("unknown", config={"video_native_models": [None, "", " "]}) == "frames"
    assert detect_video_capability("unknown", config={"video_native_models": "unknown"}) == "frames"


@pytest.mark.parametrize("override", ["native", "frames", "none"])
def test_global_override_and_legacy_migration(override):
    config = {"video_capability_override": override, "video_no_vision_models": ["gemini"]}
    assert detect_video_capability("gemini", modalities=["text"], config=config) == override
    assert detect_video_capability("gemini", config={"video_mode": override, "video_capability_override": "auto"}) == override
    assert detect_video_capability("model", config={"video_mode": "native", "video_frames_only_models": ["model"]}) == "frames"
    assert detect_video_capability("unknown", config={"video_capability_override": "typo"}) == "frames"


def test_schema_retains_hidden_legacy_key_during_default_merge():
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    # AstrBot drops undeclared keys before constructing the plugin.
    assert schema["video_mode"]["invisible"] is True
    config = {key: value.get("default") for key, value in schema.items()}
    config["video_mode"] = "frames"
    assert detect_video_capability("gemini-3-flash", modalities=["image"], config=config) == "frames"


def test_video_panel_exposes_three_simple_defaults_and_keeps_legacy_keys():
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    visible = {key for key, item in schema.items() if key.startswith("video_") and not item.get("invisible")}
    assert visible == {"video_understanding_enabled", "video_max_duration_sec", "video_audio_transcribe"}
    assert schema["video_understanding_enabled"]["default"] is True
    assert schema["video_max_duration_sec"]["default"] == 120
    assert schema["video_audio_transcribe"]["default"] == "auto"
    assert schema["video_audio_transcribe"]["options"] == ["auto", "off"]
    config = {key: value.get("default") for key, value in schema.items()}
    assert detect_video_capability("gemini-3-flash", modalities=["image"], config=config) == "native"
    assert detect_video_capability("gpt-4o", modalities=["image"], config=config) == "frames"
    assert detect_video_capability("deepseek-chat", modalities=["text"], config=config) == "none"


@pytest.mark.parametrize("model", ["gpt-4o", "claude-sonnet-4", "deepseek-chat", "qwen3-32b", "glm-4-plus"])
def test_switching_away_from_video_model_does_not_inherit_provider_name(model):
    assert video_family(model, "newapigemini") == ""
    assert video_family("newapigemini/" + model, "newapigemini") == ""
    assert detect_video_capability(model, "newapigemini", ["image"]) == "frames"


@pytest.mark.parametrize("model", ["deepseek-chat", "qwen3-32b", "glm-4-plus", "llama-3.3-70b", "mistral-large"])
def test_known_text_models_without_modalities_are_textual(model):
    assert detect_video_capability(model, "newapigemini") == "none"
