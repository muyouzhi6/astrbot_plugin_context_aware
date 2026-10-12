"""Issue #10: emulate Windows file opening without a Windows host."""

import io
import ntpath
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from image_context import SNAPSHOT_KEY, ImageIndex, _local_image_path
from test_gemini_stt_context import FakeContext, FakeEvent, load_plugin_module
from test_image_context import png


class WindowsRecallTests(unittest.IsolatedAsyncioTestCase):
    def test_windows_path_matrix(self):
        cases = (
            ("file:///C:/images/a%20b.png", r"C:\images\a b.png"),
            (r"file://C:\images\a.png", r"C:\images\a.png"),
            ("file://C:/images/a.png", r"C:\images\a.png"),
            ("FILE:///d:/images/a.png", r"D:\images\a.png"),
            ("file://localhost/C:/images/a.png", r"C:\images\a.png"),
            ("file://LOCALHOST/C:/images/a.png", r"C:\images\a.png"),
            ("file://server/share/a%20b.png", r"\\server\share\a b.png"),
            ("file:////server/share/a.png", r"\\server\share\a.png"),
            ("file:///C:/%E5%9B%BE%E7%89%87/a%23b%25c.png", r"C:\图片\a#b%c.png"),
            ("file:///C:/a%2520b.png", r"C:\a%20b.png"),
            (r"file://C:\图片#1.jpg", r"C:\图片#1.jpg"),
            (r"file://C:\图片?1.jpg", r"C:\图片?1.jpg"),
            (r"FILE://c:\图片%23a#b?c.jpg", r"C:\图片#a#b?c.jpg"),
            ("file:///C:/images/a%23b.jpg?download=1#preview", r"C:\images\a#b.jpg"),
            ("file://server/share/a%23b.jpg?download=1#preview", r"\\server\share\a#b.jpg"),
            (r"C:\images\a%20b.png", r"C:\images\a%20b.png"),
            (r"\\server\share\a.png", r"\\server\share\a.png"),
        )
        with patch("image_context.os", SimpleNamespace(name="nt")):
            for source, expected in cases:
                with self.subTest(source=source):
                    self.assertEqual(_local_image_path(source), expected)

    async def test_windows_file_uris_reach_the_correct_file(self):
        index = ImageIndex()
        self.addAsyncCleanup(index.close)
        raw = png()

        def windows_open(path, mode):
            actual = str(path)
            # Reject the same malformed paths that Windows rejects; do not
            # silently map a dropped drive or authority to a valid fixture.
            if ntpath.normcase(ntpath.normpath(actual)) != ntpath.normcase(expected):
                raise OSError(22, "Invalid argument", actual)
            self.assertEqual(mode, "rb")
            return io.BytesIO(raw)

        for source, expected in (
            ("file:///C:/%E5%9B%BE%E7%89%87/test%20image.png", r"C:\图片\test image.png"),
            (r"file://C:\图片\test image.png", r"C:\图片\test image.png"),
            (r"file://C:\图片#1.png", r"C:\图片#1.png"),
            (r"file://C:\图片?1.png", r"C:\图片?1.png"),
            ("file:///C:/%E5%9B%BE%E7%89%87/a%23b.png?x=1#view", r"C:\图片\a#b.png"),
        ):
            with self.subTest(source=source):
                with (
                    patch("image_context.os", SimpleNamespace(name="nt"), create=True),
                    patch("image_context.Path.open", windows_open),
                ):
                    self.assertEqual(await index._fetch(source), raw)

    async def test_posix_paths_and_uri_decoding_use_real_files(self):
        index = ImageIndex()
        self.addAsyncCleanup(index.close)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "图片 # %20.png"
            raw = png()
            path.write_bytes(raw)
            with patch("image_context.os", SimpleNamespace(name="posix")):
                for source in (str(path), path.as_uri()):
                    with self.subTest(source=source):
                        self.assertEqual(await index._fetch(source), raw)

    async def test_windows_message_llm_and_tool_lifecycle(self):
        mod = load_plugin_module()
        with tempfile.TemporaryDirectory() as directory:
            plugin = mod.Main(FakeContext(), {
                "image_cache_dir": directory, "image_recall_auto": False,
            })
            plugin._recall_supports_vision = AsyncMock(return_value=True)
            try:
                for i, (source, expected) in enumerate((
                    ("file:///C:/images/test%20image.png", r"C:\images\test image.png"),
                    (r"file://C:\images\test image.png", r"C:\images\test image.png"),
                    ("file://server/share/test%20image.png", r"\\server\share\test image.png"),
                    (r"file://C:\images\photo#1.png", r"C:\images\photo#1.png"),
                    (r"file://C:\images\photo?1.png", r"C:\images\photo?1.png"),
                )):
                    def windows_open(path, mode):
                        self.assertEqual(ntpath.normpath(str(path)), expected)
                        return io.BytesIO(png())

                    event = FakeEvent()
                    event.extras = {}
                    event.unified_msg_origin += f":{i}"
                    event.message_obj = SimpleNamespace(message_id=f"image-{i}")
                    event.message_str = "[图片]"
                    event.get_messages = lambda: [mod.Image(file=source)]
                    with (
                        patch("image_context.os", SimpleNamespace(name="nt")),
                        patch("image_context.Path.open", windows_open),
                    ):
                        await plugin.on_message(event)
                    ids = tuple(plugin._image_index.sessions[event.unified_msg_origin])
                    self.assertEqual(len(ids), 1)
                    self.assertEqual(plugin._image_index.resources[ids[0]].data, png())
                    # Simulate Core deleting the original after on_message.
                    with patch("image_context.Path.open", side_effect=FileNotFoundError):
                        question = FakeEvent()
                        question.extras = {}
                        question.unified_msg_origin = event.unified_msg_origin
                        question.message_obj = SimpleNamespace(message_id=f"question-{i}")
                        question.message_str = "看看刚才的图片"
                        question.get_messages = lambda: [mod.Plain(question.message_str)]
                        await plugin.on_message(question)
                        request = mod.ProviderRequest()
                        await plugin.on_llm_request(question, request)
                        self.assertEqual(question.get_extra(SNAPSHOT_KEY), ids)
                        result = await plugin.context_aware_view_images(question, list(ids))
                        self.assertEqual([part.type for part in result.content], ["text", "image"])
                        duplicate = await plugin.context_aware_view_images(question, list(ids))
                        self.assertFalse(any(part.type == "image" for part in duplicate.content))
                        for invalid in ([source], ["ca_other_session"], [], list(ids) * 5):
                            result = await plugin.context_aware_view_images(question, invalid)
                            self.assertFalse(any(part.type == "image" for part in result.content))
                        await plugin._clear_session_context(question, "test reset")
                        result = await plugin.context_aware_view_images(question, list(ids))
                        self.assertFalse(any(part.type == "image" for part in result.content))
                        self.assertNotIn(ids[0], plugin._image_index.resources)
            finally:
                await plugin.terminate()

    async def test_missing_malformed_and_oversized_images_keep_chat_working(self):
        mod = load_plugin_module()
        with tempfile.TemporaryDirectory() as directory:
            plugin = mod.Main(FakeContext(), {
                "image_cache_dir": directory, "image_download_max_bytes": 32,
                "image_recall_auto": False,
            })
            plugin._recall_supports_vision = AsyncMock(return_value=True)
            try:
                for i, source in enumerate(("missing.png", "file:///C:/bad%00.png", "large.png", "broken.png")):
                    if source == "large.png":
                        path = Path(directory) / source
                        path.write_bytes(b"x" * 33)
                        source = str(path)
                    elif source == "broken.png":
                        path = Path(directory) / source
                        path.write_bytes(b"invalid png")
                        source = path.as_uri()
                    event = FakeEvent()
                    event.extras = {}
                    event.unified_msg_origin += f":bad-{i}"
                    event.message_obj = SimpleNamespace(message_id=f"bad-{i}")
                    event.message_str = "继续聊天"
                    event.get_messages = lambda: [mod.Plain("继续聊天"), mod.Image(file=source)]
                    await plugin.on_message(event)
                    request = mod.ProviderRequest()
                    await plugin.on_llm_request(event, request)
                    self.assertTrue(any("<conversation_scene>" in p.text for p in request.extra_user_content_parts))
                    ids = event.get_extra(SNAPSHOT_KEY)
                    self.assertEqual(len(ids), 1)
                    result = await plugin.context_aware_view_images(event, list(ids))
                    self.assertFalse(any(p.type == "image" for p in result.content))
                    self.assertFalse(plugin._image_index.resources[ids[0]].data)
            finally:
                await plugin.terminate()
