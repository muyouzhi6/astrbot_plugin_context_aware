"""Video audio transcription through configured AstrBot STT providers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

try:
    from .video_input import _run, _worker
except ImportError:
    from video_input import _run, _worker


def _enabled(provider):
    config = getattr(provider, "provider_config", {}) or {}
    return (provider is not None and callable(getattr(provider, "get_text", None))
            and config.get("enable", True) is not False
            and config.get("enabled", True) is not False)


def select_stt_provider(context, origin, provider_id=""):
    if provider_id:
        try:
            provider = context.get_provider_by_id(provider_id)
            if _enabled(provider):
                return provider
        except Exception:
            pass
    try:
        provider = context.get_using_stt_provider(umo=origin)
        if _enabled(provider):
            return provider
    except Exception:
        pass
    try:
        for provider in context.get_all_stt_providers() or []:
            if _enabled(provider):
                return provider
    except Exception:
        pass
    return None


def has_audio(path, deadline):
    probe = shutil.which("ffprobe")
    if not probe:
        return False
    data = json.loads(_run([
        probe, "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "a:0", "-show_entries", "stream=codec_type",
        "-of", "json", str(path),
    ], deadline))
    return any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))


def extract_audio(path, target, seconds, deadline):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg missing")
    _run([
        ffmpeg, "-nostdin", "-v", "error", "-threads", "1",
        "-protocol_whitelist", "file,pipe", "-i", str(path),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", str(seconds),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-threads", "1", "-y", str(target),
    ], deadline)


def _digest(path, max_bytes):
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("audio source bytes")
            digest.update(chunk)
    return digest.hexdigest()


class AudioTranscriber:
    def __init__(self, context, config=None, *, timeout_sec=25):
        self.context = context
        self.provider_id = str((config or {}).get("video_stt_provider_id", "") or "")
        self.timeout_sec = timeout_sec
        self.cache = OrderedDict()
        self.semaphore = asyncio.Semaphore(1)

    def available(self, origin):
        return select_stt_provider(self.context, origin, self.provider_id) is not None

    async def transcribe(self, path, duration, options, origin=""):
        if options.audio_transcribe == "off":
            return ""
        provider = select_stt_provider(self.context, origin, self.provider_id)
        if provider is None:
            return ""
        seconds = min(duration, options.max_duration_sec)
        if seconds <= 0:
            return ""
        key = None

        async def work():
            nonlocal key
            deadline = time.monotonic() + self.timeout_sec
            async with self.semaphore:
                key = (origin, id(provider), await _worker(_digest, path, options.max_bytes), seconds)
                now = time.monotonic()
                for stale, (_, created) in list(self.cache.items()):
                    if now - created >= 600:
                        del self.cache[stale]
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key][0]
                text = ""
                if await _worker(has_audio, path, deadline):
                    with tempfile.TemporaryDirectory(prefix="context-aware-audio-") as directory:
                        target = Path(directory) / "audio.wav"
                        await _worker(extract_audio, path, target, seconds, deadline)
                        transcript = await provider.get_text(str(target))
                        if isinstance(transcript, str) and transcript.strip():
                            text = f"[视频语音转写，约 {seconds:.0f} 秒] " + transcript.strip()[:2000]
                self._remember(key, text)
                return text

        try:
            return await asyncio.wait_for(work(), self.timeout_sec)
        except Exception:
            # Provider errors and transcripts never enter logs.
            if key is not None:
                self._remember(key, "")
            return ""

    def _remember(self, key, text):
        self.cache[key] = (text, time.monotonic())
        self.cache.move_to_end(key)
        while len(self.cache) > 64:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()
