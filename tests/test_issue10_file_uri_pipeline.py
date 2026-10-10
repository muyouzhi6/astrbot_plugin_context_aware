"""Issue #10: real compression with Windows filesystem operations mapped to temp files.

The fixture never decodes URIs. Only correctly converted, registered Windows
paths reach a file; /C:/... and file://... fail as they do on Windows.
"""

import base64
import io
import ntpath
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import ImageContent
from PIL import Image as PILImage

import image_context
import video_input
from image_compression import compress_local_image
from image_context import SEEN_KEY, SNAPSHOT_KEY
from test_gemini_stt_context import FakeContext, FakeEvent, load_plugin_module


class WindowsFiles:
    def __init__(self):
        self.files = {}

    def register(self, windows_path, real_path):
        self.files[ntpath.normcase(windows_path)] = Path(real_path)
        return windows_path

    def local(self, path):
        value = str(path)
        key = ntpath.normcase(value)
        if key in self.files:
            return self.files[key]
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(
            ("file:", "FILE:", "/C:", "\\\\")
        ):
            raise FileNotFoundError(value)
        return Path(value)

    def isfile(self, path):
        try:
            return self.local(path).is_file()
        except OSError:
            return False

    def open(self, path, mode="rb"):
        return self.local(path).open(mode)

    def path(self, value):
        fs = self

        class WindowsPath(PureWindowsPath):
            def resolve(self, **kwargs):
                return self

            def is_file(self):
                return fs.isfile(self)

            def stat(self):
                return fs.local(self).stat()

            def open(self, mode="rb"):
                return fs.open(self, mode)

        if ntpath.isabs(str(value)) and not str(value).startswith("/"):
            return WindowsPath(value)
        return Path(value)

    def compress(self, source, directory, options, gif_options):
        # Run Pillow and the production compressor against actual bytes.
        outcome = compress_local_image(
            str(self.local(source)), directory, options, gif_options
        )
        if not outcome.changed:
            return replace(outcome, source_path=source, output_path=source)
        output = self.register(
            str(
                PureWindowsPath(r"C:\系统\AstrBot\aware")
                / Path(outcome.output_path).name
            ),
            outcome.output_path,
        )
        return replace(outcome, source_path=source, output_path=output)

    def patches(self, mod):
        stack = ExitStack()
        path_ops = SimpleNamespace(
            **{name: getattr(os.path, name) for name in dir(os.path)}
        )
        path_ops.isfile = self.isfile
        path_ops.exists = self.isfile
        path_ops.getsize = lambda path: self.local(path).stat().st_size
        path_ops.isabs = ntpath.isabs
        stack.enter_context(patch.object(mod, "os", SimpleNamespace(path=path_ops)))
        stack.enter_context(patch.object(mod, "Path", self.path))
        stack.enter_context(patch.object(mod, "open", self.open, create=True))
        stack.enter_context(
            patch.object(
                mod,
                "shutil",
                SimpleNamespace(
                    copy2=lambda source, target: shutil.copy2(
                        self.local(source), target
                    ),
                ),
            )
        )
        stack.enter_context(patch.object(mod, "compress_local_image", self.compress))
        stack.enter_context(
            patch.object(image_context, "os", SimpleNamespace(name="nt"))
        )
        stack.enter_context(patch.object(image_context, "Path", self.path))
        stack.enter_context(patch.object(video_input, "Path", self.path))
        return stack


class ImageURLPart(SimpleNamespace):
    type = "image_url"
    ImageURL = SimpleNamespace

    def mark_as_temp(self):
        self.temporary = True
        return self


class WindowsImagePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.mod = load_plugin_module()
        self.plugin = self.mod.Main(
            FakeContext(),
            {
                "image_cache_dir": str(self.root / "cache"),
                "llm_image_compress": {"enable": True, "max_edge": 512},
            },
        )
        self.plugin._image_compress_output_dir = str(self.root / "compressed")
        self.plugin._recall_supports_vision = lambda event: True
        self.plugin._media_capability = lambda event, req=None: ("image", "test")
        self.addAsyncCleanup(self.plugin.terminate)
        source = self.root / "source.jpg"
        with PILImage.effect_noise((1024, 768), 65).convert("RGB") as image:
            image.save(source, "JPEG", quality=100, subsampling=0)
        self.fs = WindowsFiles()
        self.source = self.fs.register(r"C:\系统\AstrBot\图片 # %20.jpg", source)
        self.unc = self.fs.register(r"\\server\share\图片 # %20.jpg", source)
        self.sources = (
            PureWindowsPath(self.source).as_uri(),
            "file://C:\\系统\\AstrBot\\图片%20%23%20%2520.jpg",
            PureWindowsPath(self.unc).as_uri(),
            PureWindowsPath(self.source).as_uri().replace("file:", "FILE:"),
            "file://" + self.source.replace("%", "%25"),
        )

    def event(self, message_id, components=None, text=""):
        event = FakeEvent()
        event.extras = {}
        event.message_obj = SimpleNamespace(message_id=message_id)
        event.message_str = text
        event.is_at_or_wake_command = True
        event.get_messages = lambda: (
            components
            if components is not None
            else [self.mod.At("bot"), self.mod.Plain(text)]
        )
        event.tracked = []
        event.track_temporary_local_file = event.tracked.append
        return event

    def request(self, contexts=None, images=None):
        request = self.mod.ProviderRequest()
        request.image_urls = images or []
        request.contexts = contexts or []
        return request

    async def test_compression_entry_accepts_all_windows_uri_forms(self):
        with self.fs.patches(self.mod):
            for source in self.sources:
                with self.subTest(source=source):
                    local = await self.plugin._materialize_image_for_compression(source)
                    self.assertEqual(
                        local, self.unc if "server" in source else self.source
                    )

    async def test_lazy_download_copies_windows_uri_bytes(self):
        with self.fs.patches(self.mod):
            for source in self.sources:
                with self.subTest(source=source):
                    local = await self.plugin._download_image_to_local(source)
                    self.assertIsNotNone(local)
                    self.assertEqual(
                        Path(local).read_bytes(),
                        self.fs.local(self.source).read_bytes(),
                    )
                    self.assertEqual(Path(local).parent, self.root / "cache")

    async def test_valid_history_is_kept_and_missing_history_is_removed(self):
        with self.fs.patches(self.mod):
            for source in self.sources:
                for dict_value in (False, True):
                    with self.subTest(source=source, dict_value=dict_value):
                        value = (
                            {"url": source, "detail": "high"} if dict_value else source
                        )
                        req = self.request(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "历史"},
                                        {"type": "image_url", "image_url": value},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": "file:///C:/missing.jpg"
                                            },
                                        },
                                    ],
                                }
                            ]
                        )
                        await self.plugin._compress_provider_request_images(
                            self.event("history"), req
                        )
                        parts = req.contexts[0]["content"]
                        self.assertEqual(len(parts), 2)
                        value = parts[1]["image_url"]
                        uri = value["url"] if dict_value else value
                        if dict_value:
                            self.assertEqual(value["detail"], "high")
                        with PILImage.open(
                            io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))
                        ) as image:
                            self.assertLessEqual(max(image.size), 512)

    async def test_history_compression_result_uri_is_serialized(self):
        req = self.request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.source}},
                    ],
                }
            ]
        )
        with (
            self.fs.patches(self.mod),
            patch.object(
                self.plugin,
                "_compress_image_reference",
                AsyncMock(return_value=self.sources[0]),
            ),
        ):
            await self.plugin._compress_provider_request_images(
                self.event("cached"), req
            )
        uri = req.contexts[0]["content"][0]["image_url"]["url"]
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        self.assertEqual(
            base64.b64decode(uri.split(",", 1)[1]),
            self.fs.local(self.source).read_bytes(),
        )

    def test_component_replacement_converts_uri_before_resolving(self):
        with self.fs.patches(self.mod):
            for source in self.sources:
                with self.subTest(source=source):
                    component = self.mod.Image()
                    self.plugin._replace_component_image_ref(component, source)
                    expected = self.unc if "server" in source else self.source
                    self.assertEqual(component.path, expected)
                    self.assertEqual(component.file, PureWindowsPath(expected).as_uri())

    async def test_quoted_uri_and_native_path_are_not_attached_twice(self):
        from image_compression import ImageCompressionOptions

        self.plugin._image_compress_options = ImageCompressionOptions()
        reply = self.mod.Reply()
        reply.chain = [self.mod.Image(file=self.sources[0])]
        req = self.request(images=[self.source])
        with self.fs.patches(self.mod):
            await self.plugin._compress_provider_request_images(
                self.event("quote", [reply]), req
            )
        self.assertEqual(req.image_urls, [self.source])

    def test_gif_signature_uses_decoded_windows_and_posix_uri(self):
        local = self.root / "动画 # %20.bin"
        with PILImage.new("RGB", (8, 8)) as image:
            image.save(local, "GIF")
        source = self.fs.register(r"\\server\share\动画 # %20.bin", local)
        with self.fs.patches(self.mod):
            self.assertTrue(
                self.mod._image_ref_looks_like_gif(PureWindowsPath(source).as_uri())
            )
        self.assertTrue(self.mod._image_ref_looks_like_gif(local.as_uri()))

    async def test_compression_auto_attach_history_and_tool_lifecycle(self):
        await self.check_compression_lifecycle(self.sources[0])

    async def test_native_path_compression_produces_recallable_history_uri(self):
        await self.check_compression_lifecycle(self.source)

    async def test_adapter_literal_hash_survives_compression_and_recall(self):
        await self.check_compression_lifecycle(self.sources[-1])

    async def check_compression_lifecycle(self, source):
        # Keep compression and default automatic recall enabled throughout.
        self.assertTrue(self.plugin._image_compress_options.enabled)
        self.assertTrue(self.plugin._recall_auto)
        component = self.mod.Image(file=source)
        event = self.event("upload", [component])
        original = self.fs.local(self.source).read_bytes()
        with (
            self.fs.patches(self.mod),
            patch.dict(
                sys.modules,
                {
                    "astrbot.core.agent.message": SimpleNamespace(
                        ImageURLPart=ImageURLPart
                    ),
                },
            ),
        ):
            await self.plugin.on_message(event)
            self.assertEqual(self.plugin._image_compress_count, 1)
            self.assertEqual(self.plugin._image_compress_errors, 0)
            self.assertEqual(component.file, PureWindowsPath(component.path).as_uri())
            self.assertIn(component.path, event.tracked)
            self.assertNotEqual(component.path, self.source)
            self.assertEqual(self.fs.local(self.source).read_bytes(), original)
            compressed = self.fs.local(component.path).read_bytes()
            self.assertLess(len(compressed), len(original))
            record = event.get_extra(self.mod.ExtraKeys.CURRENT_MESSAGE_RECORD)
            self.assertEqual(record.image_urls, [component.file])
            ids = tuple(self.plugin._image_index.sessions[event.unified_msg_origin])
            self.assertEqual(len(ids), 1)
            self.assertEqual(
                self.plugin._image_index.resources[ids[0]].data, compressed
            )
            self.assertEqual(self.plugin._image_index.resources[ids[0]].retry_at, 0)

            # A current request reuses the compressed copy; a later request
            # keeps the still-existing URI in history and automatically recalls it.
            await self.plugin.on_llm_request(
                event, self.request(images=[component.file])
            )
            self.assertEqual(self.plugin._image_compress_count, 1)
            question = self.event("question", text="图里是什么？")
            await self.plugin.on_message(question)
            req = self.request(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": component.file}},
                        ],
                    }
                ]
            )
            await self.plugin.on_llm_request(question, req)
            history = req.contexts[0]["content"]
            self.assertEqual(len(history), 1)
            history_uri = history[0]["image_url"]["url"]
            self.assertEqual(base64.b64decode(history_uri.split(",", 1)[1]), compressed)
            self.assertEqual(question.get_extra(SNAPSHOT_KEY), ids)
            attached = [
                p
                for p in req.extra_user_content_parts
                if getattr(p, "type", "") == "image_url"
            ]
            self.assertEqual(len(attached), 1)
            self.assertTrue(attached[0].temporary)
            self.assertEqual(question.get_extra(SEEN_KEY)[ids[0]], "auto")
            with PILImage.open(
                io.BytesIO(base64.b64decode(attached[0].image_url.url.split(",", 1)[1]))
            ) as image:
                self.assertEqual(image.size, (512, 384))

            # Core cleanup deletes originals/copies; retained bytes still serve
            # automatic attachment and a separate request's explicit tool read.
            self.fs.local(self.source).unlink()
            self.fs.local(component.path).unlink()
            after = self.event("after-cleanup", text="图里是什么？")
            await self.plugin.on_message(after)
            after_req = self.request()
            await self.plugin.on_llm_request(after, after_req)
            self.assertEqual(after.get_extra(SEEN_KEY)[ids[0]], "auto")
            retained = [
                p for p in after_req.extra_user_content_parts
                if getattr(p, "type", "") == "image_url"
            ]
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].image_url.url, attached[0].image_url.url)
            tool_event = self.event("tool", text="继续")
            await self.plugin.on_message(tool_event)
            await self.plugin.on_llm_request(tool_event, self.request())
            result = await self.plugin.context_aware_view_images(tool_event, list(ids))
            self.assertEqual([part.type for part in result.content], ["text", "image"])
            self.assertIsInstance(result.content[1], ImageContent)
            with PILImage.open(io.BytesIO(base64.b64decode(result.content[1].data))) as image:
                self.assertEqual(image.size, (512, 384))
            self.assertEqual(self.plugin._image_compress_errors, 0)

    async def test_video_local_consumers_use_the_same_windows_conversion(self):
        local = self.root / "media_video_test.mp4"
        local.write_bytes(b"video fixture")
        path = self.fs.register(r"C:\系统\AstrBot\media_video_test.mp4", local)
        uri = PureWindowsPath(path).as_uri()
        with self.fs.patches(self.mod):
            with self.subTest(consumer="component"):
                self.assertEqual(
                    video_input.video_source(SimpleNamespace(file=uri)), path
                )
            for source in (uri, path):
                with self.subTest(consumer="materialize", source=source):
                    self.assertEqual(
                        await video_input._materialize(
                            source, str(self.root), video_input.VideoOptions()
                        ),
                        path,
                    )
            with self.subTest(consumer="downloaded"):
                req = SimpleNamespace(prompt=f"视频: {uri}")
                self.assertEqual(
                    video_input.downloaded_video_paths(req, r"C:\系统\AstrBot"), [path]
                )

    def test_video_extension_is_detected_from_local_uri(self):
        with self.fs.patches(self.mod):
            for source in (
                r"file://C:\系统\AstrBot\video.mp4",
                "file:///C:/video%2Emp4",
            ):
                with self.subTest(source=source):
                    component = type("File", (), {"file_": source})()
                    self.assertTrue(video_input.is_video_component(component))
