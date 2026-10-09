from __future__ import annotations

import asyncio
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from audio_transcribe import AudioTranscriber, select_stt_provider
from video_input import NO_VISION, UNPARSED, VideoOptions, understand_video
from test_video_input import (
    MockChatProvider,
    attach_video,
    plugin_request,
    run_chat,
    video as video,
)


@pytest.fixture
def audio_video(video, tmp_path):
    target = tmp_path / "with-audio.mp4"
    subprocess.run([
        shutil.which("ffmpeg"), "-nostdin", "-v", "error",
        "-i", str(video), "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-threads", "1", "-shortest", "-y", str(target),
    ], check=True, timeout=15)
    return target


def stt_context(stt=None):
    return SimpleNamespace(
        get_using_stt_provider=lambda **kw: stt,
        get_all_stt_providers=lambda: [stt] if stt else [],
        get_provider_by_id=lambda name: stt,
    )


def test_select_stt_global_first_then_first_enabled_and_old_provider_id():
    global_stt = SimpleNamespace(get_text=AsyncMock(), provider_config={"enable": True})
    disabled = SimpleNamespace(get_text=AsyncMock(), provider_config={"enable": False})
    fallback = SimpleNamespace(get_text=AsyncMock(), provider_config={"enable": True})
    context = stt_context(global_stt)
    context.get_all_stt_providers = lambda: [disabled, fallback]
    assert select_stt_provider(context, "room") is global_stt
    context.get_using_stt_provider = Mock(return_value=disabled)
    assert select_stt_provider(context, "room") is fallback
    context.get_using_stt_provider.assert_called_with(umo="room")
    context.get_provider_by_id = lambda name: global_stt if name == "legacy" else None
    assert select_stt_provider(context, "room", "legacy") is global_stt
    assert select_stt_provider(context, "room", "missing") is fallback
    context.get_all_stt_providers = lambda: [disabled, SimpleNamespace()]
    assert select_stt_provider(context, "room") is None


@pytest.mark.asyncio
async def test_transcribe_real_audio_format_duration_truncation_cache_and_cleanup(audio_video, tmp_path):
    paths = []

    async def recognize(path):
        paths.append(path)
        with wave.open(path, "rb") as audio:
            assert audio.getframerate() == 16000
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getnframes() <= 16000
        return "语" * 3000

    stt = SimpleNamespace(get_text=AsyncMock(side_effect=recognize))
    service = AudioTranscriber(stt_context(stt))
    options = VideoOptions(max_duration_sec=1)
    result = await service.transcribe(str(audio_video), 2, options, "room")
    assert result == "[视频语音转写，约 1 秒] " + "语" * 2000
    assert all(not Path(path).exists() for path in paths)
    duplicate = tmp_path / "downloaded-again.mp4"
    duplicate.write_bytes(audio_video.read_bytes())
    assert await service.transcribe(str(duplicate), 2, options, "room") == result
    assert stt.get_text.await_count == 1
    assert await service.transcribe(str(duplicate), 2, options, "other") == result
    assert stt.get_text.await_count == 2
    assert all(not Path(path).parent.exists() for path in paths)


@pytest.mark.asyncio
async def test_no_audio_track_skips_extraction_and_stt(video):
    stt = SimpleNamespace(get_text=AsyncMock(side_effect=AssertionError))
    service = AudioTranscriber(stt_context(stt))
    with patch("audio_transcribe.extract_audio", side_effect=AssertionError):
        assert await service.transcribe(str(video), 2, VideoOptions()) == ""
        assert await service.transcribe(str(video), 2, VideoOptions()) == ""
    stt.get_text.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "off"])
async def test_no_stt_or_off_does_not_probe_or_read(mode):
    stt = SimpleNamespace(get_text=AsyncMock()) if mode == "off" else None
    service = AudioTranscriber(stt_context(stt))
    with patch("audio_transcribe.has_audio", side_effect=AssertionError), patch("audio_transcribe._digest", side_effect=AssertionError):
        assert await service.transcribe("not-readable", 2, VideoOptions(audio_transcribe=mode)) == ""


@pytest.mark.asyncio
async def test_stt_timeout_is_silent_cached_and_cleans_temp(audio_video):
    paths = []

    async def pending(path):
        paths.append(path)
        await asyncio.Event().wait()

    stt = SimpleNamespace(get_text=AsyncMock(side_effect=pending))
    service = AudioTranscriber(stt_context(stt), timeout_sec=0.1)
    with patch("audio_transcribe.has_audio", return_value=True), patch("audio_transcribe.extract_audio"):
        assert await service.transcribe(str(audio_video), 2, VideoOptions()) == ""
        assert await service.transcribe(str(audio_video), 2, VideoOptions()) == ""
    stt.get_text.assert_awaited_once()
    assert paths and all(not Path(path).parent.exists() for path in paths)


@pytest.mark.asyncio
async def test_stt_cancellation_propagates_and_cleans_temp(audio_video):
    started = asyncio.Event()
    paths = []

    async def pending(path):
        paths.append(path)
        started.set()
        await asyncio.Event().wait()

    service = AudioTranscriber(stt_context(SimpleNamespace(get_text=AsyncMock(side_effect=pending))))
    with patch("audio_transcribe.has_audio", return_value=True), patch("audio_transcribe.extract_audio"):
        task = asyncio.create_task(service.transcribe(str(audio_video), 2, VideoOptions()))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert all(not Path(path).parent.exists() for path in paths)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["probe", "extract", "stt", "empty"])
async def test_audio_failures_are_silent(audio_video, failure):
    stt = SimpleNamespace(get_text=AsyncMock(return_value="" if failure == "empty" else "词"))
    if failure == "stt":
        stt.get_text.side_effect = ValueError("STT error")
    service = AudioTranscriber(stt_context(stt))
    probe = Mock(side_effect=ValueError("probe error")) if failure == "probe" else Mock(return_value=True)
    extract = Mock(side_effect=ValueError("extract error")) if failure == "extract" else Mock()
    with patch("audio_transcribe.has_audio", probe), patch("audio_transcribe.extract_audio", extract):
        assert await service.transcribe(str(audio_video), 2, VideoOptions()) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["frames", "none"])
async def test_frames_and_text_models_receive_audio(audio_video, capability):
    stt = SimpleNamespace(get_text=AsyncMock(return_value="下一站到了"))
    service = AudioTranscriber(stt_context(stt))
    result = await understand_video(
        str(audio_video), None, capability=capability, transcriber=service, origin="room",
    )
    assert "[视频语音转写，约 2 秒] 下一站到了" in result.text
    assert result.transcript in result.text
    if capability == "frames":
        assert result.images
    else:
        assert result.text.startswith(NO_VISION) and not result.images
    stt.get_text.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_rejection_adds_audio_and_final_text_keeps_it(audio_video, tmp_path, stream):
    provider = MockChatProvider("gemini-3-flash")
    seen = []
    stt = SimpleNamespace(get_text=AsyncMock(return_value="记得关灯"))

    async def response(payload, *args, **kwargs):
        seen.append(payload)
        if any(p.get("type") == "image_url" for p in payload["messages"][0]["content"]):
            raise ValueError("media rejected")
        return "文字回答"

    async def streaming(payload, *args, **kwargs):
        yield await response(payload, *args, **kwargs)

    provider._query = response
    provider._query_stream = streaming
    module, plugin, event, req = plugin_request(tmp_path, provider)
    plugin._context.get_using_stt_provider = lambda **kw: stt
    attach_video(event, audio_video)
    try:
        with patch.object(module.logger, "info") as info, patch.object(module.logger, "warning") as warning:
            await plugin.on_llm_request(event, req)
            stt.get_text.assert_not_awaited()
            if stream:
                payload = {"model": provider.model, "messages": [{"role": "user", "content": [{"type": "text", "text": req.extra_user_content_parts[0].text}]}]}
                assert [item async for item in provider._query_stream(payload)] == ["文字回答"]
            else:
                assert await run_chat(provider, req) == "文字回答"
            assert "记得关灯" not in str(info.call_args_list) + str(warning.call_args_list)
        assert len(seen) == 3
        assert seen[0]["messages"][0]["content"][-1]["image_url"]["url"].startswith("data:video/")
        assert any(p.get("type") == "image_url" for p in seen[1]["messages"][0]["content"])
        final = seen[2]["messages"][0]["content"][-1]["text"]
        assert final.startswith(UNPARSED) and "视频语音转写" in final and "记得关灯" in final
        stt.get_text.assert_awaited_once()
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_native_success_never_probes_or_transcribes_audio(audio_video, tmp_path):
    provider = MockChatProvider("gemini-3-flash")
    _, plugin, event, req = plugin_request(tmp_path, provider)
    stt = SimpleNamespace(get_text=AsyncMock(side_effect=AssertionError))
    plugin._context.get_using_stt_provider = lambda **kw: stt
    attach_video(event, audio_video)
    try:
        with patch("audio_transcribe.has_audio", side_effect=AssertionError):
            await plugin.on_llm_request(event, req)
            await run_chat(provider, req)
        stt.get_text.assert_not_awaited()
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_text_model_main_request_contains_transcript_only(audio_video, tmp_path):
    provider = MockChatProvider("deepseek-chat", ["text"])
    _, plugin, event, req = plugin_request(tmp_path, provider)
    stt = SimpleNamespace(get_text=AsyncMock(return_value="请带雨伞"))
    plugin._context.get_using_stt_provider = lambda **kw: stt
    attach_video(event, audio_video)
    try:
        with patch("video_input.extract_video_frames", side_effect=AssertionError):
            await plugin.on_llm_request(event, req)
            await run_chat(provider, req)
        content = provider.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert all(p["type"] == "text" for p in content)
        assert "请带雨伞" in content[-1]["text"]
        assert not plugin._video_adapters
    finally:
        await plugin.terminate()
