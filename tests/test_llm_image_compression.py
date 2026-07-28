from __future__ import annotations

import tempfile
import types
import unittest
import base64
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from image_compression import ImageCompressionOptions, compress_local_image

try:
    from tests.test_gemini_stt_context import FakeContext, load_plugin_module
except ModuleNotFoundError:
    from test_gemini_stt_context import FakeContext, load_plugin_module


def _make_noisy_jpeg(path: Path, size: tuple[int, int] = (1800, 1200)) -> bytes:
    image = Image.effect_noise(size, 65).convert("RGB")
    image.save(path, "JPEG", quality=100, subsampling=0)
    image.close()
    return path.read_bytes()


class ImageCompressionCoreTest(unittest.TestCase):
    def test_options_are_disabled_and_safe_by_default(self):
        options = ImageCompressionOptions.from_mapping({})

        self.assertFalse(options.enabled)
        self.assertEqual(options.max_edge, 2048)
        self.assertEqual(options.quality, 90)
        self.assertEqual(options.min_quality, 75)
        self.assertEqual(options.download_retries, 3)

    def test_invalid_options_are_clamped(self):
        options = ImageCompressionOptions.from_mapping(
            {
                "enable": True,
                "max_edge": 10,
                "quality": 200,
                "min_quality": 99,
                "max_input_size_mb": 0,
                "max_output_size_mb": 999,
                "download_retries": 99,
                "download_timeout": 1,
            }
        )

        self.assertTrue(options.enabled)
        self.assertEqual(options.max_edge, 512)
        self.assertEqual(options.quality, 100)
        self.assertEqual(options.min_quality, 99)
        self.assertEqual(options.max_input_bytes, 1024 * 1024)
        self.assertEqual(options.max_output_bytes, options.max_input_bytes)
        self.assertEqual(options.download_retries, 5)
        self.assertEqual(options.download_timeout, 5)

    def test_large_rgb_image_is_compressed_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "source.jpg"
            original = _make_noisy_jpeg(source)
            options = ImageCompressionOptions.from_mapping(
                {
                    "enable": True,
                    "min_size_mb": 0.1,
                    "max_edge": 1024,
                    "quality": 90,
                    "min_quality": 75,
                    "max_output_size_mb": 1.0,
                    "max_input_size_mb": 20,
                }
            )

            outcome = compress_local_image(str(source), root, options)

            self.assertTrue(outcome.changed, outcome.reason)
            self.assertEqual(source.read_bytes(), original)
            self.assertLess(outcome.output_bytes, outcome.source_bytes)
            self.assertLessEqual(outcome.output_bytes, options.max_output_bytes)
            with Image.open(outcome.output_path) as compressed:
                self.assertLessEqual(max(compressed.size), options.max_edge)
                self.assertEqual(compressed.format, "JPEG")

    def test_transparent_png_preserves_alpha(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "transparent.png"
            image = Image.new("RGBA", (1400, 1000), (255, 0, 0, 100))
            image.save(source, "PNG")
            image.close()
            options = ImageCompressionOptions.from_mapping(
                {
                    "enable": True,
                    "min_size_mb": 100,
                    "max_edge": 700,
                    "max_output_size_mb": 2,
                }
            )

            outcome = compress_local_image(str(source), root, options)

            self.assertTrue(outcome.changed, outcome.reason)
            self.assertTrue(outcome.output_path.endswith(".png"))
            with Image.open(outcome.output_path) as compressed:
                self.assertEqual(compressed.format, "PNG")
                self.assertIn("A", compressed.mode)
                self.assertLessEqual(max(compressed.size), 700)

    def test_animated_image_is_left_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "animated.gif"
            frames = [
                Image.new("RGB", (640, 640), color)
                for color in ((255, 0, 0), (0, 255, 0))
            ]
            frames[0].save(
                source,
                "GIF",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )
            for frame in frames:
                frame.close()
            options = ImageCompressionOptions.from_mapping(
                {"enable": True, "min_size_mb": 0.1, "max_edge": 512}
            )

            outcome = compress_local_image(str(source), root, options)

            self.assertFalse(outcome.changed)
            self.assertEqual(outcome.reason, "animated_image")

    def test_source_over_input_limit_is_left_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "too-large.jpg"
            _make_noisy_jpeg(source)
            options = ImageCompressionOptions.from_mapping(
                {
                    "enable": True,
                    "min_size_mb": 0.1,
                    "max_input_size_mb": 1,
                }
            )

            outcome = compress_local_image(str(source), root, options)

            self.assertFalse(outcome.changed)
            self.assertEqual(outcome.reason, "source_too_large")
            self.assertEqual(outcome.output_path, str(source))


class FakeCompressionEvent:
    def __init__(self, *, private: bool = True):
        self.extras: dict[str, object] = {}
        self.unified_msg_origin = "aiocqhttp:private:100"
        self.is_at_or_wake_command = True
        self.private = private
        self.tracked: list[str] = []

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def is_private_chat(self):
        return self.private

    def track_temporary_local_file(self, path: str):
        self.tracked.append(path)


class LLMImageCompressionIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = load_plugin_module()

    async def test_request_images_compress_even_when_context_feature_is_disabled(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "request.jpg"
            _make_noisy_jpeg(source)
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "only_group_chat": True,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {
                        "enable": True,
                        "min_size_mb": 0.1,
                        "max_edge": 1024,
                        "quality": 90,
                        "max_output_size_mb": 1,
                        "max_input_size_mb": 20,
                    },
                },
            )
            plugin._image_compress_output_dir = str(Path(root) / "output")
            event = FakeCompressionEvent(private=True)
            req = types.SimpleNamespace(
                image_urls=[str(source)],
                extra_user_content_parts=[],
            )

            try:
                await plugin.on_llm_request(event, req)
            finally:
                await plugin.terminate()

            self.assertNotEqual(req.image_urls[0], str(source))
            self.assertTrue(Path(req.image_urls[0]).exists())
            self.assertIn(req.image_urls[0], event.tracked)

    async def test_disabled_compression_keeps_request_image_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "request.jpg"
            _make_noisy_jpeg(source)
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {"enable": False},
                },
            )
            event = FakeCompressionEvent(private=True)
            req = types.SimpleNamespace(
                image_urls=[str(source)],
                extra_user_content_parts=[],
            )

            try:
                await plugin.on_llm_request(event, req)
            finally:
                await plugin.terminate()

            self.assertEqual(req.image_urls, [str(source)])
            self.assertEqual(event.tracked, [])

    async def test_quoted_image_is_compressed_before_main_agent_builds_request(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "quoted.jpg"
            _make_noisy_jpeg(source)
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {
                        "enable": True,
                        "min_size_mb": 0.1,
                        "max_edge": 1024,
                        "max_output_size_mb": 1,
                    },
                },
            )
            plugin._image_compress_output_dir = str(Path(root) / "output")
            image_component = self.mod.Image(url=str(source))
            reply_component = self.mod.Reply()
            reply_component.chain = [image_component]
            event = FakeCompressionEvent(private=True)
            event.get_messages = lambda: [reply_component]

            try:
                await plugin.on_message(event)
            finally:
                await plugin.terminate()

            self.assertNotEqual(image_component.url, str(source))
            self.assertTrue(Path(image_component.url).exists())
            self.assertIn(image_component.url, event.tracked)

    async def test_data_uri_is_materialized_compressed_and_not_processed_twice(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "inline.jpg"
            raw = _make_noisy_jpeg(source)
            data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {
                        "enable": True,
                        "min_size_mb": 0.1,
                        "max_edge": 1024,
                        "max_output_size_mb": 1,
                        "max_input_size_mb": 20,
                    },
                },
            )
            plugin._image_compress_output_dir = str(Path(root) / "output")
            event = FakeCompressionEvent(private=True)
            req = types.SimpleNamespace(
                image_urls=[data_uri],
                extra_user_content_parts=[],
            )

            try:
                await plugin._compress_provider_request_images(event, req)
                first_output = req.image_urls[0]
                req.image_urls = [first_output]
                with patch.object(
                    self.mod,
                    "compress_local_image",
                    side_effect=AssertionError("compressed output was processed twice"),
                ):
                    await plugin._compress_provider_request_images(event, req)
            finally:
                await plugin.terminate()

            self.assertEqual(req.image_urls[0], first_output)
            self.assertTrue(Path(first_output).exists())
            self.assertEqual(plugin._image_compress_count, 1)

    async def test_remote_download_retries_until_complete(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "image_cache_dir": root,
                    "llm_image_compress": {"enable": True},
                },
            )
            calls = 0

            def fake_download(url, local_path, *, max_bytes, timeout):
                nonlocal calls
                calls += 1
                if calls < 3:
                    return None
                Path(local_path).write_bytes(b"complete")
                return len(b"complete")

            try:
                with patch.object(
                    plugin,
                    "_download_remote_image_sync",
                    side_effect=fake_download,
                ):
                    result = await plugin._download_image_to_local(
                        "https://example.com/image.jpg",
                        max_bytes=1024,
                        retries=3,
                        timeout=5,
                    )
            finally:
                await plugin.terminate()

            self.assertEqual(calls, 3)
            self.assertIsNotNone(result)
            self.assertEqual(Path(result).read_bytes(), b"complete")

    async def test_recent_download_failure_is_not_retried_by_second_hook(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "image_cache_dir": root,
                    "llm_image_compress": {"enable": True},
                },
            )

            try:
                with patch.object(
                    plugin,
                    "_download_remote_image_sync",
                    return_value=None,
                ) as download:
                    first = await plugin._download_image_to_local(
                        "https://example.com/incomplete.jpg",
                        max_bytes=1024,
                        retries=3,
                        timeout=5,
                    )
                    second = await plugin._download_image_to_local(
                        "https://example.com/incomplete.jpg",
                        max_bytes=1024,
                        retries=3,
                        timeout=5,
                    )
            finally:
                await plugin.terminate()

            self.assertIsNone(first)
            self.assertIsNone(second)
            self.assertEqual(download.call_count, 3)

    async def test_download_cache_files_are_isolated_by_size_limit(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "image_cache_dir": root,
                    "llm_image_compress": {"enable": True},
                },
            )

            def fake_download(url, local_path, *, max_bytes, timeout):
                Path(local_path).write_bytes(b"image")
                return len(b"image")

            try:
                with patch.object(
                    plugin,
                    "_download_remote_image_sync",
                    side_effect=fake_download,
                ):
                    first = await plugin._download_image_to_local(
                        "https://example.com/image.jpg",
                        max_bytes=1024,
                    )
                    second = await plugin._download_image_to_local(
                        "https://example.com/image.jpg",
                        max_bytes=2048,
                    )
            finally:
                await plugin.terminate()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).exists())
            self.assertTrue(Path(second).exists())

    def test_incomplete_content_length_is_rejected_and_cleaned(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = self.mod.Main(FakeContext(), {"image_cache_dir": root})
            destination = Path(root) / "partial.jpg"

            class FakeResponse:
                status = 200
                headers = {"Content-Length": "10"}

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self, size):
                    if not hasattr(self, "sent"):
                        self.sent = True
                        return b"12345"
                    return b""

            with patch.object(
                self.mod.urllib.request,
                "urlopen",
                return_value=FakeResponse(),
            ):
                result = plugin._download_remote_image_sync(
                    "https://example.com/image.jpg",
                    str(destination),
                    max_bytes=1024,
                    timeout=5,
                )

            self.assertIsNone(result)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
