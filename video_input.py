"""Bounded video input, provider-specific formats and guarded main requests."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import math
import mimetypes
import shutil
import subprocess
import tempfile
import time
import uuid
import weakref
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlsplit

from PIL import Image

try:
    from .gif_frames import GifOptions, number, render_frames
    from .model_capability import detect_video_capability, video_family
except ImportError:
    from gif_frames import GifOptions, number, render_frames
    from model_capability import detect_video_capability, video_family

MIB = 1024 * 1024
UNPARSED = "[视频：未能解析]"
NO_VISION = "[视频：当前模型不支持查看]"
VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".flv",
    ".wmv",
    ".m4v",
}


@dataclass(frozen=True)
class VideoOptions:
    enabled: bool = False
    max_bytes: int = 50 * MIB
    max_duration_sec: float = 120
    mode: str = "auto"
    fallback_frames: int = 6
    inline_max_bytes: int = 12 * MIB
    timeout_sec: float = 60
    transcode: bool = False

    @classmethod
    def from_mapping(cls, raw):
        raw = raw if isinstance(raw, Mapping) else {}
        mode = raw.get("video_capability_override", "auto")
        if mode == "auto" or mode not in ("auto", "native", "frames", "none"):
            mode = raw.get("video_mode", "auto")
        return cls(
            enabled=raw.get("video_understanding_enabled", False) is True,
            max_bytes=int(number(raw, "video_max_bytes", 50 * MIB, 1024, 200 * MIB)),
            max_duration_sec=number(raw, "video_max_duration_sec", 120, 1, 600),
            mode=mode if mode in ("auto", "native", "frames", "none") else "auto",
            fallback_frames=int(number(raw, "video_fallback_frames", 6, 1, 16)),
            inline_max_bytes=int(
                number(raw, "video_inline_max_bytes", 12 * MIB, 1024, 12 * MIB)
            ),
            timeout_sec=number(raw, "video_timeout_sec", 60, 1, 180),
            transcode=raw.get("video_transcode", False) is True,
        )


@dataclass(frozen=True)
class VideoResult:
    text: str = UNPARSED
    images: tuple[tuple[bytes, str], ...] = ()
    reason: str = "unparsed"
    video: bytes = field(default=b"", repr=False)
    mime: str = "video/mp4"
    family: str = ""
    token: str = field(default_factory=lambda: "[context-aware-video:" + uuid.uuid4().hex + "]")


def is_video_component(component):
    if type(component).__name__ == "Video":
        return True
    if type(component).__name__ != "File":
        return False
    if str(getattr(component, "mime_type", "")).startswith("video/"):
        return True
    return any(
        Path(urlsplit(str(getattr(component, attr, "") or "")).path).suffix.lower()
        in VIDEO_SUFFIXES
        for attr in ("name", "file", "url", "path")
    )


async def _worker(function, *args):
    # Do not delete a temp directory while a cancelled thread is still using it.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise


def _run(command, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("video processing deadline")
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=remaining,
    ).stdout


def probe_video(path, options, deadline):
    if Path(path).stat().st_size > options.max_bytes:
        raise ValueError("video source bytes")
    probe = shutil.which("ffprobe")
    if not probe:
        raise ValueError("ffprobe missing")
    data = json.loads(
        _run(
            [
                probe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration:stream=duration,width,height,codec_type",
                "-of",
                "json",
                str(path),
            ],
            min(deadline, time.monotonic() + 10),
        )
    )
    streams = data.get("streams", [])
    if not streams or streams[0].get("codec_type") != "video":
        raise ValueError("no video stream")
    stream = streams[0]
    duration = float(
        data.get("format", {}).get("duration") or stream.get("duration", 0)
    )
    if not math.isfinite(duration) or not 0 < duration <= options.max_duration_sec:
        raise ValueError("video duration")
    if not 0 < int(stream.get("width", 0)) * int(stream.get("height", 0)) <= 25_000_000:
        raise ValueError("video source pixels")
    return duration


def extract_video_frames(path, duration, options, gif_options, directory, deadline):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return ()
    images, times = [], []
    limits = replace(
        gif_options,
        mode="frames" if gif_options.mode == "frames" else "grid",
        max_frames=options.fallback_frames,
    )
    edge = limits.frame_max_edge
    try:
        for index in range(options.fallback_frames):
            timestamp = duration * (index + 0.5) / options.fallback_frames
            target = Path(directory) / f"frame-{index}.png"
            _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    "1",
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    f"scale={edge}:{edge}:force_original_aspect_ratio=decrease",
                    "-threads",
                    "1",
                    "-y",
                    str(target),
                ],
                deadline,
            )
            if target.is_file():
                with Image.open(target) as image:
                    images.append(image.convert("RGB"))
                times.append(timestamp)
        return render_frames(images, times, limits, deadline=deadline)
    finally:
        for image in images:
            image.close()


def transcode_video(path, directory, options, deadline):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return path
    target = Path(directory) / "normalized.mp4"
    _run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-sn",
            "-dn",
            "-vf",
            "scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-maxrate",
            "1500k",
            "-bufsize",
            "3000k",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-t",
            str(options.max_duration_sec),
            "-fs",
            str(options.max_bytes),
            "-movflags",
            "+faststart",
            "-y",
            str(target),
        ],
        deadline,
    )
    probe_video(target, options, deadline)
    return str(target)


async def _materialize(source, directory, options):
    if source.startswith(("http://", "https://", "data:", "base64://")):
        # Reuse the recall downloader's redirect/DNS checks and streaming byte cap.
        try:
            from .image_context import ImageIndex
        except ImportError:
            from image_context import ImageIndex
        reader = ImageIndex(
            max_download_bytes=options.max_bytes, budget_bytes=options.max_bytes
        )
        data = await asyncio.wait_for(
            reader._fetch(source), min(15, options.timeout_sec)
        )
        if source.startswith("data:"):
            mime = source[5:source.index(";")]
            suffix = mimetypes.guess_extension(mime) or ".mp4"
        elif source.startswith("base64://"):
            suffix = ".mp4"
        else:
            suffix = Path(urlsplit(source).path).suffix.lower()
        target = Path(directory) / (
            "input" + (suffix if suffix in VIDEO_SUFFIXES else ".mp4")
        )
        await _worker(target.write_bytes, data)
        return str(target)
    path = unquote(urlsplit(source).path) if source.startswith("file://") else source
    if urlsplit(source).scheme not in ("", "file"):
        raise ValueError("unsupported video URL")
    if not Path(path).is_file() or Path(path).stat().st_size > options.max_bytes:
        raise ValueError("video source unavailable or oversized")
    return path


def _model(provider, model):
    if model:
        return str(model)
    getter = getattr(provider, "get_model", None)
    return str(
        getter()
        if callable(getter)
        else getattr(provider, "provider_config", {}).get("model", "")
    )


def _read_bounded(path, limit):
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("video bytes changed")
    return data


def openai_video_part(data, mime, family):
    mime = {"video/quicktime": "video/mov", "video/x-msvideo": "video/avi"}.get(mime, mime)
    encoded = base64.b64encode(data).decode("ascii")
    url = f"data:{mime};base64,{encoded}"
    if family == "gemini":
        # New API preserves video MIME in image_url and maps it to inline_data.
        return {"type": "image_url", "image_url": {"url": url}}
    if family == "doubao":
        return {"type": "video_url", "video_url": {"url": url, "fps": 1}}
    if family == "glm":
        # GLM's documented local-video example uses raw Base64, without data:.
        return {"type": "video_url", "video_url": {"url": encoded}}
    return {"type": "video_url", "video_url": {"url": url}}


def _inline(size, options, envelope_size=0):
    return (
        size <= options.inline_max_bytes
        and 4 * ((size + 2) // 3) + envelope_size + MIB < 20_000_000
    )


@asynccontextmanager
async def gemini_video_part(provider, path, options, envelope_size=0):
    client = getattr(provider, "client", None)
    mime = mimetypes.guess_type(path)[0] or "video/mp4"
    mime = {"video/quicktime": "video/mov", "video/x-msvideo": "video/avi"}.get(
        mime, mime
    )
    uploaded = None
    try:
        if _inline(Path(path).stat().st_size, options, envelope_size):
            data = await _worker(_read_bounded, path, options.inline_max_bytes)
            yield {"inline_data": {"mime_type": mime, "data": data}}
        else:
            files = getattr(client, "files", None)
            if not callable(getattr(files, "upload", None)):
                raise ValueError("Files API unavailable")
            uploaded = await files.upload(file=path, config={"mime_type": mime})
            while _file_state(uploaded) == "PROCESSING":
                await asyncio.sleep(1)
                uploaded = await files.get(name=uploaded.name)
            if _file_state(uploaded) != "ACTIVE" or not getattr(uploaded, "uri", None):
                raise ValueError("uploaded video not active")
            yield {"file_data": {"mime_type": mime, "file_uri": uploaded.uri}}
    finally:
        if uploaded is not None and getattr(uploaded, "name", None):
            try:
                await asyncio.wait_for(client.files.delete(name=uploaded.name), 5)
            except Exception:
                pass


def _file_state(uploaded):
    state = getattr(uploaded, "state", "")
    return str(getattr(state, "name", state)).upper()


async def describe_native(provider, path, options, prompt, model=""):
    model = _model(provider, model)
    client = getattr(provider, "client", None)
    models = getattr(client, "models", None)
    if callable(getattr(models, "generate_content", None)):
        async with gemini_video_part(provider, path, options, len(prompt.encode())) as part:
            config = {"max_output_tokens": 2048}
            safety = getattr(provider, "safety_settings", None)
            if safety is not None:
                config["safety_settings"] = safety
            response = await models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [part, {"text": prompt}]}],
                config=config,
            )
            text = getattr(response, "text", "")
    else:
        if not _inline(Path(path).stat().st_size, options, len(prompt.encode())):
            raise ValueError("compat provider has no Gemini Files API")
        create = getattr(
            getattr(getattr(client, "chat", None), "completions", None), "create", None
        )
        if not callable(create):
            raise ValueError("compat client unavailable")
        data = await _worker(_read_bounded, path, options.inline_max_bytes)
        mime = mimetypes.guess_type(path)[0] or "video/mp4"
        config = getattr(provider, "provider_config", {}) or {}
        part = await _worker(openai_video_part, data, mime, video_family(model, config.get("id", "")))
        response = await create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        part,
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            stream=False,
            max_tokens=2048,
        )
        text = response.choices[0].message.content if response.choices else ""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty video response")
    return text.strip()[:6000]


async def understand_video(
    source, provider, options=None, gif_options=None, *, question="", model="",
    capability=None, for_main=False,
):
    options = options or VideoOptions()
    gif_options = gif_options or GifOptions()
    if not options.enabled:
        return VideoResult(text="", reason="disabled")
    config = getattr(provider, "provider_config", {}) or {}
    capability = capability or detect_video_capability(
        _model(provider, model) if provider is not None else "",
        config.get("id", ""), config.get("modalities"),
        {"video_mode": options.mode},
    )
    if capability == "none":
        return VideoResult(text=NO_VISION, reason="none")
    try:
        with tempfile.TemporaryDirectory(prefix="context-aware-video-") as directory:
            path = await _materialize(source, directory, options)
            deadline = time.monotonic() + min(30, options.timeout_sec)
            duration = await _worker(probe_video, path, options, deadline)
            native_path = path
            if options.transcode and capability == "native":
                try:
                    native_path = await _worker(
                        transcode_video, path, directory, options, deadline
                    )
                except Exception:
                    native_path = path
            if capability == "native":
                try:
                    if for_main:
                        data = await _worker(_read_bounded, native_path, options.max_bytes)
                        return VideoResult(
                            text="", reason="native_main", video=data,
                            mime=mimetypes.guess_type(native_path)[0] or "video/mp4",
                            family=video_family(_model(provider, model), config.get("id", "")),
                        )
                    prompt = (
                        "请描述视频的动作、事件顺序、画面文字和声音；关键事件标注时间。结合用户问题提供可核对的观察。\n用户问题："
                        + str(question)[:4000]
                    )
                    text = await asyncio.wait_for(
                        describe_native(provider, native_path, options, prompt, model),
                        options.timeout_sec,
                    )
                    return VideoResult(text="[视频观察]\n" + text, reason="native")
                except Exception:
                    pass  # Do not log provider errors containing request media/auth.
            images = await _worker(
                extract_video_frames,
                path,
                duration,
                options,
                gif_options,
                directory,
                time.monotonic() + min(30, options.timeout_sec),
            )
            if images:
                layout = "，分别附图" if len(images) > 1 else "拼成网格"
                return VideoResult(
                    text=f"[视频：已按时间顺序抽取{options.fallback_frames}帧{layout}；标有序号和时间]",
                    images=images,
                    reason="frames",
                )
    except Exception:
        pass
    return VideoResult()


class MainVideoAdapter:
    """Adapt AstrBot 4.28 query payloads without changing shared SDK clients.

    Temporary TextParts survive Core's image resolver and modality sanitizer.
    Only registered tokens are converted, on a copied payload for this request.
    Provider methods are restored on plugin termination; plans follow event GC.
    """

    def __init__(self, provider, options, gif_options, config=None):
        self.provider = provider
        self.options = options
        self.gif_options = gif_options
        self.config = config or {}
        self.plans = weakref.WeakValueDictionary()
        self.originals = {}
        self.wrappers = {}
        self.gemini = callable(getattr(provider, "_prepare_conversation", None)) and callable(
            getattr(getattr(getattr(provider, "client", None), "models", None), "generate_content", None)
        )

    def install(self):
        if not callable(getattr(self.provider, "_query", None)):
            return False
        # Restrict the legacy adapter to the two inspected AstrBot payload shapes.
        client = getattr(self.provider, "client", None)
        create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
        if not self.gemini and not callable(create):
            return False
        methods = {"_query": self.query}
        if callable(getattr(self.provider, "_query_stream", None)):
            methods["_query_stream"] = self.query_stream
        if self.gemini:
            methods["_prepare_conversation"] = self.conversation
        try:
            for name, wrapper in methods.items():
                self.originals[name] = getattr(self.provider, name)
                setattr(self.provider, name, wrapper)
                self.wrappers[name] = wrapper
        except Exception:
            self.close()
            return False
        return True

    def close(self):
        for name, original in self.originals.items():
            if getattr(self.provider, name, None) is self.wrappers.get(name):
                setattr(self.provider, name, original)
        self.plans.clear()

    def register(self, result):
        self.plans[result.token] = result
        return result.token

    def _find(self, payload):
        found = {}
        for message in payload.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    token = part.get("text") if isinstance(part, dict) else None
                    if isinstance(token, str) and token in self.plans:
                        found[token] = self.plans[token]
        return found

    @staticmethod
    def _replace(payload, replacements):
        if not any(
            isinstance(message.get("content"), list) and any(
                isinstance(part, dict) and isinstance(part.get("text"), str)
                and part["text"].startswith("[context-aware-video:")
                for part in message["content"]
            ) for message in payload.get("messages", [])
        ):
            return payload
        result = {**payload, "messages": copy.deepcopy(payload.get("messages", []))}
        for message in result["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            updated = []
            for part in content:
                token = part.get("text") if isinstance(part, dict) else None
                if isinstance(token, str) and token.startswith("[context-aware-video:"):
                    updated.extend(replacements.get(token, [{"type": "text", "text": UNPARSED}]))
                else:
                    updated.append(part)
            message["content"] = updated
        return result

    @asynccontextmanager
    async def _native_payload(self, payload, plans):
        envelope = len(json.dumps(payload, ensure_ascii=False).encode())
        replacements = {}
        config = getattr(self.provider, "provider_config", {}) or {}
        family = video_family(payload.get("model", ""), config.get("id", ""))
        async with AsyncExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="context-aware-video-"))
            total = envelope
            for index, (token, result) in enumerate(plans.items()):
                if not result.video:
                    raise ValueError("frames-only video plan")
                if self.gemini:
                    suffix = mimetypes.guess_extension(result.mime) or ".mp4"
                    path = Path(directory) / f"video-{index}{suffix}"
                    await _worker(path.write_bytes, result.video)
                    part = await stack.enter_async_context(
                        gemini_video_part(self.provider, str(path), self.options, total)
                    )
                    if "inline_data" in part:
                        total += 4 * ((len(result.video) + 2) // 3)
                    replacements[token] = [{"type": "context_aware_video", "token": token, "part": part}]
                else:
                    if not _inline(len(result.video), self.options, total):
                        raise ValueError("compat inline request limit")
                    total += 4 * ((len(result.video) + 2) // 3)
                    part = await _worker(openai_video_part, result.video, result.mime, family)
                    replacements[token] = [part]
            prepared = self._replace(payload, replacements)
            if family == "glm" and not self.gemini:
                for message in prepared["messages"]:
                    content = message.get("content")
                    if isinstance(content, list):
                        # GLM requires video_url before text within the user content.
                        message["content"] = [p for p in content if p.get("type") == "video_url"] + [p for p in content if p.get("type") != "video_url"]
            yield prepared

    async def conversation(self, payload, *args, **kwargs):
        native = {}
        clean = {**payload, "messages": copy.deepcopy(payload.get("messages", []))}
        for message in clean["messages"]:
            if not isinstance(message.get("content"), list):
                continue
            for index, part in enumerate(message["content"]):
                if isinstance(part, dict) and part.get("type") == "context_aware_video":
                    native[part["token"]] = part["part"]
                    message["content"][index] = {"type": "text", "text": part["token"]}
        conversation = await self.originals["_prepare_conversation"](clean, *args, **kwargs)
        for content in conversation:
            for index, part in enumerate(content.parts or []):
                token = getattr(part, "text", None)
                if token in native:
                    # SDK Part accepts inline_data/file_data; no image decoding.
                    content.parts[index] = type(part)(**native[token])
        return conversation

    async def _frames(self, payload, plans):
        replacements = {}
        for token, result in plans.items():
            try:
                fallback = result
                if result.video:
                    source = await _worker(lambda: f"data:{result.mime};base64," + base64.b64encode(result.video).decode("ascii"))
                    fallback = await understand_video(
                        source, None, replace(self.options, mode="frames"), self.gif_options,
                        capability="frames",
                    )
                parts = [{"type": "text", "text": fallback.text}]
                for data, mime in fallback.images:
                    url = await _worker(lambda: f"data:{mime};base64," + base64.b64encode(data).decode("ascii"))
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                replacements[token] = parts
            except Exception:
                replacements[token] = [{"type": "text", "text": UNPARSED}]
        return self._replace(payload, replacements)

    async def query(self, payload, *args, **kwargs):
        plans = self._find(payload)
        original = self.originals["_query"]
        if not plans:
            return await original(self._replace(payload, {}), *args, **kwargs)
        config = getattr(self.provider, "provider_config", {}) or {}
        capability = detect_video_capability(
            payload.get("model", ""), config.get("id", ""), config.get("modalities"), self.config,
        )
        if capability == "none":
            return await original(self._replace(payload, {
                token: [{"type": "text", "text": NO_VISION}] for token in plans
            }), *args, **kwargs)

        async def native():
            async with self._native_payload(payload, plans) as prepared:
                return await original(prepared, *args, **kwargs)

        if capability == "native":
            try:
                return await asyncio.wait_for(native(), self.options.timeout_sec)
            except Exception:
                pass  # Provider exceptions may include media/credentials; do not log.
        frames = await self._frames(payload, plans)
        try:
            return await original(frames, *args, **kwargs)
        except Exception:
            return await original(self._replace(payload, {}), *args, **kwargs)

    async def query_stream(self, payload, *args, **kwargs):
        plans = self._find(payload)
        original = self.originals["_query_stream"]
        if not plans:
            async for response in original(self._replace(payload, {}), *args, **kwargs):
                yield response
            return
        config = getattr(self.provider, "provider_config", {}) or {}
        capability = detect_video_capability(
            payload.get("model", ""), config.get("id", ""), config.get("modalities"), self.config,
        )
        if capability == "none":
            prepared = self._replace(payload, {
                token: [{"type": "text", "text": NO_VISION}] for token in plans
            })
            async for response in original(prepared, *args, **kwargs):
                yield response
            return

        async def collect(prepared):
            # OpenAIProvider reuses its mutable chunk response between yields.
            responses = [copy.deepcopy(response) async for response in original(prepared, *args, **kwargs)]
            if not responses or (
                all(hasattr(response, "is_chunk") for response in responses)
                and all(response.is_chunk for response in responses)
            ):
                raise ValueError("video stream has no completed response")
            return responses

        async def native():
            async with self._native_payload(payload, plans) as prepared:
                return await collect(prepared)

        try:
            if capability != "native":
                raise ValueError("frames-only request model")
            # Buffer this video's stream so a late rejection cannot duplicate output.
            responses = await asyncio.wait_for(native(), self.options.timeout_sec)
        except Exception:
            frames = await self._frames(payload, plans)
            try:
                responses = await collect(frames)
            except Exception:
                responses = await collect(self._replace(payload, {}))
        for response in responses:
            yield response
