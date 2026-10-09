from __future__ import annotations

import asyncio
import base64
import io
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from gif_frames import GifOptions
from video_input import (
    NO_VISION,
    UNPARSED,
    MainVideoAdapter,
    VideoOptions,
    describe_native,
    is_video_component,
    understand_video,
)


@pytest.fixture
def video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe 未安装，跳过真实 MP4 验证")
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=6:duration=2",
            "-an",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            "-y",
            str(path),
        ],
        check=True,
        timeout=15,
    )
    return path


def native_provider():
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=AsyncMock(
                return_value=SimpleNamespace(text="物体从左向右移动")
            )
        ),
        files=SimpleNamespace(
            upload=AsyncMock(
                return_value=SimpleNamespace(
                    name="files/test",
                    uri="https://generativelanguage.googleapis.com/v1beta/files/test",
                    state="ACTIVE",
                )
            ),
            get=AsyncMock(),
            delete=AsyncMock(),
        ),
    )
    return SimpleNamespace(client=client, get_model=lambda: "gemini-2.5-flash")


def compat_provider(model="gemini-2.5-flash"):
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="一辆汽车驶过"))]
        )
    )
    return SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        get_model=lambda: model,
    )


@pytest.mark.asyncio
async def test_disabled_is_noop_without_io():
    with patch("video_input._materialize", side_effect=AssertionError):
        result = await understand_video("not-readable", None)
    assert result.reason == "disabled" and not result.text and not result.images


@pytest.mark.asyncio
async def test_native_inline(video):
    provider = native_provider()
    result = await understand_video(str(video), provider, VideoOptions(enabled=True))
    assert result.reason == "native" and not result.images
    part = provider.client.models.generate_content.call_args.kwargs["contents"][0][
        "parts"
    ][0]
    assert part["inline_data"]["data"] == video.read_bytes()
    assert part["inline_data"]["mime_type"] == "video/mp4"
    provider.client.files.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_upload_poll_and_delete(video):
    provider = native_provider()
    active = provider.client.files.upload.return_value
    provider.client.files.upload.return_value = SimpleNamespace(
        name=active.name, state="PROCESSING"
    )
    provider.client.files.get.return_value = active
    with patch("video_input.asyncio.sleep", new=AsyncMock()):
        result = await understand_video(
            str(video), provider, VideoOptions(enabled=True, inline_max_bytes=1)
        )
    assert result.reason == "native"
    provider.client.files.delete.assert_awaited_once_with(name=active.name)
    part = provider.client.models.generate_content.call_args.kwargs["contents"][0][
        "parts"
    ][0]
    assert part["file_data"]["file_uri"] == active.uri


@pytest.mark.asyncio
async def test_upload_failure_and_inference_rejection_fall_back(video):
    provider = native_provider()
    provider.client.models.generate_content.side_effect = ValueError(
        "unsupported media"
    )
    options = VideoOptions(enabled=True, inline_max_bytes=1, fallback_frames=3)
    result = await understand_video(str(video), provider, options)
    assert result.reason == "frames" and result.images
    provider.client.files.delete.assert_awaited_once()
    provider.client.files.upload.side_effect = OSError("Files API unavailable")
    result = await understand_video(str(video), provider, options)
    assert result.reason == "frames"


@pytest.mark.asyncio
async def test_native_timeout_cleans_uploaded_file(video):
    provider = native_provider()
    provider.client.files.upload.return_value.state = "PROCESSING"
    # Exercise the model deadline without coupling it to local ffprobe startup time.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            describe_native(
                provider, str(video), VideoOptions(inline_max_bytes=1), "describe"
            ),
            0.05,
        )
    provider.client.files.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_compat_video_mime_and_fallback(video):
    provider = compat_provider()
    result = await understand_video(str(video), provider, VideoOptions(enabled=True))
    assert result.reason == "native"
    create = provider.client.chat.completions.create
    url = create.call_args.kwargs["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == video.read_bytes()
    create.side_effect = ValueError("400 unknown input type")
    result = await understand_video(str(video), provider, VideoOptions(enabled=True))
    assert result.reason == "frames"


@pytest.mark.asyncio
async def test_large_compat_and_non_gemini_skip_native(video):
    for provider, options in [
        (compat_provider(), VideoOptions(enabled=True, inline_max_bytes=1)),
        (compat_provider("grok-vision"), VideoOptions(enabled=True)),
        (compat_provider(), VideoOptions(enabled=True, mode="frames")),
    ]:
        result = await understand_video(str(video), provider, options)
        assert result.reason == "frames"
        provider.client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_frame_grid_caps_and_order(video):
    options = VideoOptions(enabled=True, mode="frames", fallback_frames=4)
    result = await understand_video(
        str(video),
        None,
        options,
        GifOptions(frame_max_edge=80, max_total_pixels=22000, max_output_bytes=16000),
    )
    assert result.reason == "frames" and "4帧" in result.text
    assert sum(len(data) for data, _ in result.images) <= 16000
    with Image.open(io.BytesIO(result.images[0][0])) as image:
        assert image.width * image.height <= 22000
        # The test source moves; first/last timeline cells differ.
        w, h = image.width // 2, image.height // 2
        assert (
            image.crop((0, 20, w, h)).tobytes()
            != image.crop((w, h + 20, 2 * w, 2 * h)).tobytes()
        )


@pytest.mark.asyncio
async def test_limits_corruption_and_missing_binaries(video, tmp_path):
    for options in (
        VideoOptions(enabled=True, max_bytes=1),
        VideoOptions(enabled=True, max_duration_sec=0.1),
    ):
        result = await understand_video(str(video), native_provider(), options)
        assert result.text == UNPARSED
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"broken video")
    assert (
        await understand_video(str(bad), None, VideoOptions(enabled=True))
    ).text == UNPARSED
    actual = shutil.which
    with patch(
        "video_input.shutil.which",
        side_effect=lambda tool: None if tool == "ffmpeg" else actual(tool),
    ):
        assert (
            await understand_video(str(video), None, VideoOptions(enabled=True))
        ).text == UNPARSED
    with patch("video_input.shutil.which", return_value=None):
        assert (
            await understand_video(str(video), None, VideoOptions(enabled=True))
        ).text == UNPARSED


@pytest.mark.asyncio
async def test_optional_transcode_and_temp_cleanup(video):
    captured = []

    async def describe(provider, path, options, prompt, model):
        captured.append(path)
        assert Path(path).name == "normalized.mp4"
        assert Path(path).stat().st_size > 0
        return "转码视频描述"

    with patch("video_input.describe_native", side_effect=describe):
        result = await understand_video(
            str(video), None, VideoOptions(enabled=True, transcode=True), capability="native"
        )
    assert result.reason == "native"
    assert not Path(captured[0]).exists()
    assert video.exists()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_remote_file_deleted(video):
    provider = native_provider()
    started = asyncio.Event()

    async def pending(**kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.client.models.generate_content.side_effect = pending
    task = asyncio.create_task(
        understand_video(
            str(video), provider, VideoOptions(enabled=True, inline_max_bytes=1)
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    provider.client.files.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_inline_budget_reserves_base64_and_envelope(tmp_path):
    path = tmp_path / "size.mp4"
    # Sparse local input is enough for request-sizing verification, without ffmpeg.
    with path.open("wb") as handle:
        handle.truncate(14 * 1024 * 1024)
    provider = compat_provider()
    options = replace(VideoOptions(), inline_max_bytes=16 * 1024 * 1024)
    with pytest.raises(ValueError, match="Files API"):
        await describe_native(provider, str(path), options, "x" * 500000)
    provider.client.chat.completions.create.assert_not_awaited()


def test_video_component_detection_and_defaults():
    Video = type("Video", (), {})
    File = type("File", (), {})
    assert is_video_component(Video())
    file = File()
    file.name = "CLIP.MP4"
    assert is_video_component(file)
    file.name = "document.pdf"
    assert not is_video_component(file)
    assert not VideoOptions.from_mapping(
        {"video_understanding_enabled": "false"}
    ).enabled
    assert (
        VideoOptions.from_mapping(
            {"video_inline_max_bytes": 999999999}
        ).inline_max_bytes
        == 12 * 1024 * 1024
    )


@pytest.mark.asyncio
async def test_plugin_disabled_and_hook_injection(video, tmp_path):
    from test_llm_image_compression import (
        FakeCompressionEvent,
        FakeContext,
        load_plugin_module,
    )

    module = load_plugin_module()
    plugin = module.Main(
        FakeContext(), {"enable": False, "image_cache_dir": str(tmp_path / "cache")}
    )
    event = FakeCompressionEvent(private=True)
    Video = type("Video", (), {})
    component = Video()
    component.file = str(video)
    event.get_messages = lambda: [component]
    req = SimpleNamespace(
        image_urls=[], extra_user_content_parts=[], prompt="发生了什么？"
    )
    try:
        with patch.object(module, "understand_video", side_effect=AssertionError):
            await plugin._prepare_request_videos(event, req)
        assert req.image_urls == [] and req.extra_user_content_parts == []
        plugin._video_options = VideoOptions(enabled=True, mode="frames")
        provider = MockChatProvider("gpt-4o")
        plugin._context.get_using_provider = lambda **kw: provider
        plugin._recall_supports_vision = lambda ev: True
        await plugin.on_llm_request(event, req)
        assert not req.image_urls
        await run_chat(provider, req)
        assert any(p["type"] == "image_url" for p in provider.client.chat.completions.create.call_args.kwargs["messages"][0]["content"])
        count = len(req.image_urls)
        with patch.object(module, "understand_video", side_effect=AssertionError):
            await plugin._prepare_request_videos(event, req)
        assert len(req.image_urls) == count
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_no_video_message_does_not_resolve_provider(tmp_path):
    from test_llm_image_compression import (
        FakeCompressionEvent,
        FakeContext,
        load_plugin_module,
    )

    module = load_plugin_module()
    plugin = module.Main(
        FakeContext(),
        {"video_understanding_enabled": True, "image_cache_dir": str(tmp_path)},
    )
    event = FakeCompressionEvent(private=True)
    event.get_messages = lambda: []
    req = SimpleNamespace(image_urls=[], extra_user_content_parts=[])
    try:
        with patch.object(
            plugin._context,
            "get_using_provider",
            create=True,
            side_effect=AssertionError,
        ):
            await plugin._prepare_request_videos(event, req)
        assert not req.image_urls and not req.extra_user_content_parts
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_remote_private_address_is_rejected():
    # The download path uses the same public-host validation as image recall.
    result = await understand_video(
        "http://127.0.0.1/private.mp4", None, VideoOptions(enabled=True)
    )
    assert result.text == UNPARSED and not result.images


@pytest.mark.asyncio
async def test_cancelled_worker_keeps_cancellation_when_subprocess_fails():
    import threading
    from video_input import _worker

    started = threading.Event()
    finish = threading.Event()

    def fail_after_cancel():
        started.set()
        finish.wait(timeout=2)
        raise subprocess.TimeoutExpired("ffmpeg", 1)

    task = asyncio.create_task(_worker(fail_after_cancel))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_each_video_component_is_processed_and_cached(tmp_path):
    from test_llm_image_compression import (
        FakeCompressionEvent,
        FakeContext,
        load_plugin_module,
    )
    from video_input import VideoResult

    module = load_plugin_module()
    plugin = module.Main(
        FakeContext(),
        {"video_understanding_enabled": True, "image_cache_dir": str(tmp_path)},
    )
    event = FakeCompressionEvent(private=True)
    Video = type("Video", (), {})
    components = [Video() for _ in range(3)]
    for i, component in enumerate(components):
        component.file = f"video-{i}.mp4"
    event.get_messages = lambda: components
    req = SimpleNamespace(image_urls=[], extra_user_content_parts=[])
    try:
        plugin._context.get_using_provider = lambda **kwargs: None
        mock = AsyncMock(
            side_effect=[
                VideoResult(text=f"观察 {i}", reason="native") for i in range(3)
            ]
        )
        with patch.object(module, "understand_video", mock):
            await plugin._prepare_request_videos(event, req)
            await plugin._prepare_request_videos(event, req)
        assert mock.await_count == 3
        assert [part.text for part in req.extra_user_content_parts] == [
            f"观察 {i}" for i in range(3)
        ]
    finally:
        await plugin.terminate()


class MockChatProvider:
    """The 4.28.2 assemble_context -> _query contract, backed by an SDK mock."""

    def __init__(self, model, modalities=None):
        self.model = model
        self.provider_config = {"id": "test-provider", "modalities": modalities or ["text", "image"]}
        self.client = compat_provider(model).client

    def get_model(self):
        return self.model

    async def _query(self, payloads, tools=None, **kwargs):
        return await self.client.chat.completions.create(**payloads, stream=False)

    async def _query_stream(self, payloads, tools=None, **kwargs):
        response = await self.client.chat.completions.create(**payloads, stream=True)
        yield response

    async def text_chat(self, prompt="", image_urls=None, extra_user_content_parts=None, model=None, contexts=None):
        content = [{"type": "text", "text": prompt}]
        for part in extra_user_content_parts or []:
            # Real 4.28.2 accepts TextPart/ImageURLPart/AudioURLPart only.
            if hasattr(part, "text"):
                content.append({"type": "text", "text": part.text})
            else:
                raise ValueError("unsupported extra part")
        content += [{"type": "image_url", "image_url": {"url": url}} for url in image_urls or []]
        return await self._query({"model": model or self.model, "messages": list(contexts or []) + [{"role": "user", "content": content}]})


class MockGeminiProvider(MockChatProvider):
    def __init__(self):
        super().__init__("gemini-3-flash")
        self.client = native_provider().client

    async def _prepare_conversation(self, payloads):
        conversation = []
        for message in payloads["messages"]:
            parts = []
            for part in message["content"]:
                if part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    parts.append(SimpleNamespace(inline_data={"mime_type": url[5:].split(";", 1)[0], "data": base64.b64decode(url.split(",", 1)[1])}))
                    continue
                if part["type"] != "text":
                    raise ValueError("Core interprets unknown parts as audio")
                parts.append(SimpleNamespace(text=part["text"]))
            conversation.append(SimpleNamespace(role=message["role"], parts=parts))
        return conversation

    async def _query(self, payloads, tools=None, **kwargs):
        contents = await self._prepare_conversation(payloads)
        return await self.client.models.generate_content(model=payloads["model"], contents=contents)


def plugin_request(tmp_path, provider, config=None, model=None):
    from test_llm_image_compression import FakeCompressionEvent, FakeContext, load_plugin_module
    module = load_plugin_module()
    context = FakeContext()
    context.get_using_provider = lambda **kwargs: provider
    context.get_provider_by_id = lambda provider_id: provider
    plugin = module.Main(context, {
        "enable": False, "image_cache_dir": str(tmp_path / "cache"),
        "video_understanding_enabled": True, **(config or {}),
    })
    event = FakeCompressionEvent(private=True)
    req = SimpleNamespace(prompt="请解释视频", model=model, image_urls=[], extra_user_content_parts=[])
    return module, plugin, event, req


def attach_video(event, path):
    component = type("Video", (), {})()
    component.file = str(path)
    component.get_file = AsyncMock(side_effect=AssertionError("none must not read video"))
    event.get_messages = lambda: [component]
    return component


async def run_chat(provider, req):
    return await provider.text_chat(
        prompt=req.prompt, model=getattr(req, "model", None), image_urls=req.image_urls,
        extra_user_content_parts=req.extra_user_content_parts,
    )


@pytest.mark.asyncio
async def test_no_vision_skips_all_video_work_and_source_access(tmp_path):
    provider = MockChatProvider("newapigemini/gemini-3.8-flash", ["text", "tool_use"])
    module, plugin, event, req = plugin_request(tmp_path, provider)
    component = attach_video(event, tmp_path / "unreadable.mp4")
    try:
        with patch.object(module, "understand_video", side_effect=AssertionError), patch("video_input.extract_video_frames", side_effect=AssertionError):
            await plugin.on_llm_request(event, req)
        component.get_file.assert_not_awaited()
        assert not req.image_urls
        assert [part.text for part in req.extra_user_content_parts] == [NO_VISION]
        provider.client.chat.completions.create.assert_not_awaited()
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_image_only_frames_reach_main_request(video, tmp_path):
    provider = MockChatProvider("gpt-4o")
    _, plugin, event, req = plugin_request(tmp_path, provider, {"gif_mode": "frames", "video_fallback_frames": 3})
    attach_video(event, video)
    try:
        with patch("video_input.describe_native", side_effect=AssertionError):
            await plugin.on_llm_request(event, req)
        assert not req.image_urls
        provider.client.chat.completions.create.assert_not_awaited()
        await run_chat(provider, req)
        content = provider.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert len([p for p in content if p["type"] == "image_url"]) == 3
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("model,family", [
    ("newapigemini/gemini-3.8-flash", "gemini"),
    ("Qwen/Qwen2.5-VL-72B-Instruct", "qwen"),
    ("qwen3-vl-plus", "qwen"), ("qwen-omni", "qwen"),
    ("GLM-4.5V", "glm"), ("glm-4v-plus-0111", "glm"),
    ("doubao-1.5-vision-pro", "doubao"), ("doubao-seed-1-6", "doubao"),
])
async def test_real_native_video_in_main_compat_request(video, tmp_path, model, family):
    provider = MockChatProvider(model)
    _, plugin, event, req = plugin_request(tmp_path, provider)
    attach_video(event, video)
    original = provider._query
    try:
        with patch("video_input.describe_native", side_effect=AssertionError), patch("video_input.extract_video_frames", side_effect=AssertionError):
            await plugin.on_llm_request(event, req)
            assert not req.image_urls
            provider.client.chat.completions.create.assert_not_awaited()
            await run_chat(provider, req)
        content = provider.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        kind = "image_url" if family == "gemini" else "video_url"
        part = next(part for part in content if part["type"] == kind)
        url = part[kind]["url"]
        if family == "glm":
            assert content[0] is part
            assert base64.b64decode(url) == video.read_bytes()
        else:
            assert url.startswith("data:video/mp4;base64,")
            assert base64.b64decode(url.split(",", 1)[1]) == video.read_bytes()
        if family == "doubao":
            assert part["video_url"]["fps"] == 1
        assert not any("context-aware-video:" in part.get("text", "") for part in content)
    finally:
        await plugin.terminate()
    assert provider._query == original


@pytest.mark.asyncio
@pytest.mark.parametrize("upload", [False, True])
async def test_gemini_main_inline_and_files_cleanup(video, tmp_path, upload):
    provider = MockGeminiProvider()
    _, plugin, event, req = plugin_request(tmp_path, provider)
    if upload:
        plugin._video_options = replace(plugin._video_options, inline_max_bytes=1)
    attach_video(event, video)
    try:
        await plugin.on_llm_request(event, req)
        provider.client.models.generate_content.assert_not_awaited()
        await run_chat(provider, req)
        contents = provider.client.models.generate_content.call_args.kwargs["contents"]
        parts = contents[-1].parts
        if upload:
            native = next(p for p in parts if hasattr(p, "file_data"))
            assert native.file_data["file_uri"].startswith("https://generativelanguage.")
            provider.client.files.delete.assert_awaited_once()
            assert not Path(provider.client.files.upload.call_args.kwargs["file"]).exists()
        else:
            native = next(p for p in parts if hasattr(p, "inline_data"))
            assert native.inline_data["data"] == video.read_bytes()
            assert native.inline_data["mime_type"] == "video/mp4"
            provider.client.files.upload.assert_not_awaited()
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_frames", [False, True])
async def test_main_native_rejection_retries_frames_then_text(video, tmp_path, reject_frames):
    provider = MockChatProvider("qwen3-vl-plus")
    _, plugin, event, req = plugin_request(tmp_path, provider)
    attach_video(event, video)
    create = provider.client.chat.completions.create
    success = create.return_value
    create.side_effect = [ValueError("reject video"), ValueError("reject frames"), success] if reject_frames else [ValueError("reject video"), success]
    try:
        await plugin.on_llm_request(event, req)
        assert await run_chat(provider, req) is success
        calls = create.call_args_list
        assert calls[0].kwargs["messages"][0]["content"][-1]["type"] == "video_url"
        assert any(p["type"] == "image_url" for p in calls[1].kwargs["messages"][0]["content"])
        if reject_frames:
            assert len(calls) == 3
            assert all(p["type"] == "text" for p in calls[2].kwargs["messages"][0]["content"])
            assert calls[2].kwargs["messages"][0]["content"][-1]["text"] == UNPARSED
        else:
            assert len(calls) == 2
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_actual_selected_provider_and_request_model_override(video, tmp_path):
    provider = MockChatProvider("gpt-4o")
    _, plugin, event, req = plugin_request(tmp_path, provider, model="qwen3-vl-flash")
    attach_video(event, video)
    event.set_extra("selected_provider", "actual-provider")
    plugin._context.get_using_provider = lambda **kw: MockChatProvider("text", ["text"])
    try:
        await plugin.on_llm_request(event, req)
        assert not req.image_urls
        await run_chat(provider, req)
        call = provider.client.chat.completions.create.call_args.kwargs
        assert call["model"] == "qwen3-vl-flash"
        assert any(p["type"] == "video_url" for p in call["messages"][0]["content"])
        req.model = "gpt-4o"
        await plugin.on_llm_request(event, req)
        assert not req.image_urls
        await run_chat(provider, req)
        assert any(p["type"] == "image_url" for p in provider.client.chat.completions.create.call_args.kwargs["messages"][0]["content"])
        plugin._config["video_capability_override"] = "none"
        with patch("video_input.extract_video_frames", side_effect=AssertionError):
            await plugin.on_llm_request(event, req)
        assert req.image_urls == []
        assert [p.text for p in req.extra_user_content_parts] == [NO_VISION]
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_unknown_main_adapter_uses_isolated_native_description(video, tmp_path):
    provider = compat_provider("glm-4.5v")  # No inspected _query entry point.
    _, plugin, event, req = plugin_request(tmp_path, provider)
    attach_video(event, video)
    try:
        await plugin.on_llm_request(event, req)
        provider.client.chat.completions.create.assert_awaited_once()
        assert req.image_urls == []
        assert req.extra_user_content_parts[0].text.startswith("[视频观察]")
        part = provider.client.chat.completions.create.call_args.kwargs["messages"][0]["content"][0]
        assert base64.b64decode(part["video_url"]["url"]) == video.read_bytes()
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_stream_late_failure_has_no_partial_output_and_no_shared_mutation(video):
    from video_input import VideoResult
    provider = MockChatProvider("qwen3-vl")
    seen = []

    async def stream(payloads, tools=None, **kwargs):
        seen.append(payloads)
        if any(p.get("type") == "video_url" for p in payloads["messages"][0]["content"]):
            yield "discard partial video output"
            raise ValueError("late native error")
        yield "frames answer"

    provider._query_stream = stream
    adapter = MainVideoAdapter(provider, VideoOptions(enabled=True), GifOptions())
    assert adapter.install()
    result = VideoResult(video=video.read_bytes(), family="qwen")
    token = adapter.register(result)
    payload = {"model": "qwen3-vl", "messages": [{"role": "user", "content": [{"type": "text", "text": token}]}]}
    try:
        assert [item async for item in provider._query_stream(payload)] == ["frames answer"]
        assert len(seen) == 2
        assert payload["messages"][0]["content"][0]["text"] == token
        unrelated = {"model": "qwen3-vl", "messages": [{"role": "user", "content": "普通文本"}]}
        await provider._query(unrelated)
        assert provider.client.chat.completions.create.call_args.kwargs["messages"] is unrelated["messages"]
    finally:
        adapter.close()


@pytest.mark.asyncio
async def test_stream_copies_mutable_core_chunks(video):
    from video_input import VideoResult
    provider = MockChatProvider("qwen3-vl")

    async def stream(payloads, *args, **kwargs):
        chunk = SimpleNamespace(text="第一段", is_chunk=True)
        yield chunk
        chunk.text = "第二段"
        yield chunk
        yield SimpleNamespace(text="第一段第二段", is_chunk=False)

    provider._query_stream = stream
    adapter = MainVideoAdapter(provider, VideoOptions(enabled=True), GifOptions())
    assert adapter.install()
    result = VideoResult(video=video.read_bytes(), family="qwen")
    token = adapter.register(result)
    payload = {"model": "qwen3-vl", "messages": [{"role": "user", "content": [{"type": "text", "text": token}]}]}
    try:
        chunks = [chunk async for chunk in provider._query_stream(payload)]
        assert [chunk.text for chunk in chunks] == ["第一段", "第二段", "第一段第二段"]
    finally:
        adapter.close()


@pytest.mark.asyncio
async def test_gemini_main_upload_failure_uses_frame_images(video, tmp_path):
    provider = MockGeminiProvider()
    _, plugin, event, req = plugin_request(tmp_path, provider)
    plugin._video_options = replace(plugin._video_options, inline_max_bytes=1)
    provider.client.files.upload.side_effect = ValueError("upload rejected")
    attach_video(event, video)
    try:
        await plugin.on_llm_request(event, req)
        await run_chat(provider, req)
        contents = provider.client.models.generate_content.call_args.kwargs["contents"]
        assert any(getattr(part, "inline_data", {}).get("mime_type", "").startswith("image/") for part in contents[-1].parts)
        provider.client.files.upload.assert_awaited_once()
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_main_video_cancel_deletes_uploaded_file_and_cleans_temp(video, tmp_path):
    provider = MockGeminiProvider()
    _, plugin, event, req = plugin_request(tmp_path, provider)
    plugin._video_options = replace(plugin._video_options, inline_max_bytes=1)
    started = asyncio.Event()

    async def pending(**kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.client.models.generate_content.side_effect = pending
    attach_video(event, video)
    try:
        await plugin.on_llm_request(event, req)
        task = asyncio.create_task(run_chat(provider, req))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        provider.client.files.delete.assert_awaited_once()
        assert not Path(provider.client.files.upload.call_args.kwargs["file"]).exists()
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_unknown_provider_defaults_to_frames_even_with_request_model(video, tmp_path):
    _, plugin, event, req = plugin_request(tmp_path, None, model="gemini-3-flash")
    attach_video(event, video)
    try:
        with patch("video_input.describe_native", side_effect=AssertionError):
            await plugin.on_llm_request(event, req)
        assert not req.image_urls
        assert req.extra_user_content_parts[0].text == UNPARSED
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_main_format_uses_actual_query_model(video):
    from video_input import VideoResult
    provider = MockChatProvider("gemini-3-flash")
    provider.provider_config["id"] = "newapigemini"
    adapter = MainVideoAdapter(provider, VideoOptions(enabled=True), GifOptions())
    assert adapter.install()
    result = VideoResult(video=video.read_bytes(), family="gemini")
    token = adapter.register(result)
    payload = {"model": "qwen3-vl", "messages": [{"role": "user", "content": [{"type": "text", "text": token}]}]}
    try:
        await provider._query(payload)
        part = provider.client.chat.completions.create.call_args.kwargs["messages"][0]["content"][0]
        assert part["type"] == "video_url"
        assert base64.b64decode(part["video_url"]["url"].split(",", 1)[1]) == video.read_bytes()
    finally:
        adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_frames_main_rejection_retries_text_without_partial_output(video, tmp_path, stream):
    provider = MockChatProvider("gpt-4o")
    calls = []

    async def streaming(payload, *args, **kwargs):
        calls.append(payload)
        if any(p.get("type") == "image_url" for p in payload["messages"][0]["content"]):
            yield "discard partial frames answer"
            raise ValueError("frames rejected")
        yield "text answer"

    if stream:
        provider._query_stream = streaming
    else:
        provider.client.chat.completions.create.side_effect = [ValueError("frames rejected"), "text answer"]
    _, plugin, event, req = plugin_request(tmp_path, provider)
    attach_video(event, video)
    try:
        await plugin.on_llm_request(event, req)
        if stream:
            token = req.extra_user_content_parts[0].text
            payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": token}]}]}
            assert [response async for response in provider._query_stream(payload)] == ["text answer"]
        else:
            assert await run_chat(provider, req) == "text answer"
            calls = [call.kwargs for call in provider.client.chat.completions.create.call_args_list]
        assert len(calls) == 2
        assert any(p["type"] == "image_url" for p in calls[0]["messages"][0]["content"])
        assert all(p["type"] == "text" for p in calls[1]["messages"][0]["content"])
        assert calls[1]["messages"][0]["content"][-1]["text"] == UNPARSED
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["data:video/mp4;base64,", "base64://"])
async def test_inline_video_decode_allows_timeout_and_loop_progress(tmp_path, prefix):
    import threading
    from video_input import _materialize

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    thread_ids = []
    loop_thread = threading.get_ident()

    def slow_decode(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        entered.set()
        try:
            release.wait(5)
            return b"video bytes"
        finally:
            finished.set()

    try:
        with patch("image_context.base64.b64decode", slow_decode):
            task = asyncio.create_task(_materialize(prefix + "YQ==", tmp_path, VideoOptions(enabled=True)))
            assert await asyncio.to_thread(entered.wait, 2)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(task, .03)
            assert thread_ids == [thread_ids[0]] and thread_ids[0] != loop_thread
            assert not finished.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/v.mp4", "http://10.1.2.3/v.mp4", "http://192.168.1.2/v.mp4",
    "http://169.254.169.254/v.mp4", "https://[::1]/v.mp4", "http://[fe80::1]/v.mp4",
    "http://[::ffff:127.0.0.1]/v.mp4", "ftp://example.com/v.mp4", "gopher://example.com/v.mp4",
])
async def test_video_private_and_unsupported_urls_never_connect(tmp_path, url):
    import aiohttp
    from video_input import _materialize

    with patch.object(aiohttp.ClientSession, "get", side_effect=AssertionError("network must not start")) as get:
        with pytest.raises(ValueError):
            await _materialize(url, tmp_path, VideoOptions(enabled=True))
        get.assert_not_called()
    assert not list(tmp_path.iterdir())


class DownloadResponse:
    def __init__(self, *, location=None, chunks=(), content_length=None):
        self.status = 302 if location else 200
        self.headers = {"Location": location} if location else {}
        self.content_length = content_length
        self.chunks = chunks
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["http://127.0.0.1/v", "https://10.0.0.1/v", "http://169.254.169.254/v", "http://[::1]/v", "file:///etc/passwd", "ftp://example.com/v"])
async def test_video_redirect_revalidates_target(tmp_path, location):
    import aiohttp
    from video_input import _materialize

    with patch.object(aiohttp.ClientSession, "get", return_value=DownloadResponse(location=location)) as get:
        with pytest.raises(ValueError):
            await _materialize("https://example.com/v.mp4", tmp_path, VideoOptions(enabled=True))
    get.assert_called_once_with("https://example.com/v.mp4", allow_redirects=False)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["10.1.2.3", "127.0.0.1", "169.254.169.254", "::1"])
async def test_video_dns_private_result_never_connects(tmp_path, host):
    from aiohttp.resolver import DefaultResolver
    from video_input import _materialize

    records = [{"hostname": "media.example", "host": host, "port": 80, "family": 2, "proto": 0, "flags": 0}]
    with patch.object(DefaultResolver, "resolve", AsyncMock(return_value=records)):
        with pytest.raises(ValueError, match="non-public"):
            await _materialize("http://media.example/v.mp4", tmp_path, VideoOptions(enabled=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [None, 4096])
async def test_video_download_caps_headers_and_stream(tmp_path, declared):
    import aiohttp
    from video_input import _materialize

    with patch.object(aiohttp.ClientSession, "get", return_value=DownloadResponse(chunks=[b"x" * 1024, b"x"], content_length=declared)):
        with pytest.raises(ValueError, match="download limit"):
            await _materialize("https://example.com/v.mp4", tmp_path, VideoOptions(enabled=True, max_bytes=1024))
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_video_public_redirect_and_bounded_inline_succeed(tmp_path):
    import aiohttp
    from video_input import _materialize

    with patch.object(aiohttp.ClientSession, "get", side_effect=[DownloadResponse(location="/final.mp4"), DownloadResponse(chunks=[b"video"], content_length=5)]) as get:
        path = await _materialize("https://example.com/v.mp4", tmp_path, VideoOptions(enabled=True, max_bytes=5))
    assert Path(path).read_bytes() == b"video"
    assert get.call_args.args[0] == "https://example.com/final.mp4"
    for prefix in ("data:video/mp4;base64,", "base64://"):
        with pytest.raises(ValueError, match="download limit"):
            await _materialize(prefix + base64.b64encode(b"123456").decode(), tmp_path, VideoOptions(enabled=True, max_bytes=5))


async def _verify_real_astrbot(video, root):
    """Executed in a fresh process: real Core, providers and SDK data models."""
    from importlib.metadata import version
    from openai.types.chat import ChatCompletion
    from google.genai import types as genai_types
    from astrbot.core.agent.hooks import BaseAgentRunHooks
    from astrbot.core.agent.message import TextPart, dump_messages_with_checkpoints
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
    from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.api.message_components import Video
    import main

    assert version("astrbot") == "4.28.2"
    cases = [
        ("gpt-4o", "frames", False, False),
        ("gpt-4o", "frames", True, False),
        ("gemini-3-flash", "none", False, False),
        ("gemini-3-flash", "native", False, False),
        ("qwen3-vl", "native", False, False),
        ("glm-4.5v", "native", False, False),
        ("doubao-seed-1-6", "native", False, False),
        ("qwen3-vl", "native", True, False),
        ("gemini-3-flash", "native", False, True),
        ("gemini-3-flash", "native", True, True),
        ("gemini-3-flash", "frames", True, True),
    ]
    for index, (model, capability, reject, gemini) in enumerate(cases):
        modalities = ["text"] if capability == "none" else ["text", "image", "tool_use"]
        config = {"id": "integration", "type": "gemini" if gemini else "openai_chat_completion", "key": ["offline-test-placeholder"], "api_base": "https://example.invalid", "model": model, "modalities": modalities}
        provider = (ProviderGoogleGenAI if gemini else ProviderOpenAIOfficial)(config, {})
        seen = []

        async def create(**kwargs):
            import copy
            if gemini:
                parts = kwargs["contents"][-1].parts
                seen.append(copy.deepcopy(parts))
                if reject and any(part.inline_data or part.file_data for part in parts):
                    raise ValueError("media rejected")
                return genai_types.GenerateContentResponse(candidates=[genai_types.Candidate(content=genai_types.Content(role="model", parts=[genai_types.Part(text="verified")]), finish_reason="STOP")])
            parts = kwargs["messages"][-1]["content"]
            seen.append(copy.deepcopy(parts))
            if reject and any(part.get("type") in ("image_url", "video_url") for part in parts):
                raise ValueError("media rejected")
            return ChatCompletion(id="offline", object="chat.completion", created=0, model=model, choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "verified"}}])

        context = SimpleNamespace(get_config=lambda **kw: {"wake_prefix": [], "provider_ltm_settings": {}}, get_using_provider=lambda **kw: provider)
        with patch.object(main, "get_astrbot_plugin_data_path", return_value=str(root)):
            plugin = main.Main(context, {"enable": False, "image_cache_dir": str(root / f"cache-{index}"), "video_understanding_enabled": True, "video_capability_override": capability})
        raw = AstrBotMessage()
        raw.type = MessageType.FRIEND_MESSAGE
        raw.self_id = "bot"
        raw.sender = MessageMember("sender", "Sender")
        raw.message_id = str(index)
        raw.message = [Video(file=video.as_uri())]
        raw.message_str = "解释视频"
        event = AstrMessageEvent(raw.message_str, raw, PlatformMetadata("test", "test", "test"), "test")
        req = ProviderRequest(prompt=raw.message_str, model=model)
        original_query = provider._query
        sdk = provider.client.models if gemini else provider.client.chat.completions
        method = "generate_content" if gemini else "create"
        try:
            with patch.object(sdk, method, AsyncMock(side_effect=create)):
                await plugin.on_llm_request(event, req)
                assert all(isinstance(p, TextPart) for p in req.extra_user_content_parts)
                # Core assembles ContentParts, validates Message and sanitizes
                # modalities, then calls the real provider.text_chat -> _query.
                runner = ToolLoopAgentRunner()
                await runner.reset(provider=provider, request=req, run_context=ContextWrapper(context=None), tool_executor=None, agent_hooks=BaseAgentRunHooks(), request_max_retries=1)
                replies = [reply async for reply in runner._iter_llm_responses()]
                assert replies[-1].completion_text == "verified"
                saved = dump_messages_with_checkpoints(runner.run_context.messages)
                assert "context-aware-video:" not in str(saved)
                # Real sanitizer must remove ordinary images for text models,
                # while preserving our temporary text token through the runner.
                filtered = runner._sanitize_contexts_for_provider([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}]}])
                assert (filtered[0]["content"][0]["type"] == "text") == (capability == "none")
            expected_calls = (3 if capability == "native" else 2) if reject else 1
            assert len(seen) == expected_calls, (model, capability, len(seen))
            if gemini:
                if capability == "native":
                    assert any(p.inline_data and p.inline_data.mime_type == "video/mp4" and p.inline_data.data == video.read_bytes() for p in seen[0])
                if reject:
                    assert all(p.text is not None for p in seen[-1])
                    assert any(p.inline_data and p.inline_data.mime_type in ("image/png", "image/jpeg") for p in seen[-2])
            else:
                if capability == "none":
                    assert seen[0][-1]["text"] == NO_VISION
                elif capability == "frames":
                    assert any(p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/") for p in seen[0])
                else:
                    kind = "image_url" if model.startswith("gemini") else "video_url"
                    part = next(p for p in seen[0] if p["type"] == kind)
                    encoded = part[kind]["url"]
                    assert base64.b64decode(encoded.split(",", 1)[-1]) == video.read_bytes()
                if reject:
                    assert all(p["type"] == "text" for p in seen[-1])
        finally:
            await plugin.terminate()
            assert provider._query == original_query
            event.cleanup_temporary_local_files()
            await provider.terminate()
    print("PASS real AstrBot 4.28.2: 11 provider/Core scenarios")


def test_real_astrbot_4282_core_and_providers(video, tmp_path):
    import os
    import sys

    interpreter = os.environ.get("ASTRBOT_TEST_PYTHON", sys.executable)
    check = subprocess.run([interpreter, "-c", "from importlib.metadata import version; assert version('astrbot') == '4.28.2'"], capture_output=True, timeout=10)
    if check.returncode:
        pytest.skip("使用装有 AstrBot 4.28.2 的 ASTRBOT_TEST_PYTHON 运行真实框架集成测试")
    repo = Path(__file__).resolve().parents[1]
    script = "import asyncio, runpy, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); module = runpy.run_path(sys.argv[2]); asyncio.run(module['_verify_real_astrbot'](Path(sys.argv[3]), Path.cwd()))"
    result = subprocess.run([interpreter, "-c", script, str(repo), str(Path(__file__).resolve()), str(video)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS real AstrBot 4.28.2: 11" in result.stdout
