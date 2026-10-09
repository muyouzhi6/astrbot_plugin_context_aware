from __future__ import annotations

import base64
import tempfile
import types
import unittest
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

    def test_gif_is_converted_to_numbered_grid_png(self):
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

            self.assertTrue(outcome.changed)
            self.assertEqual(outcome.reason, "gif_grid")
            self.assertTrue(outcome.output_path.endswith(".png"))
            with Image.open(outcome.output_path) as converted:
                self.assertEqual(converted.format, "PNG")
                self.assertEqual(getattr(converted, "n_frames", 1), 1)
                self.assertEqual(
                    converted.convert("RGB").getpixel((10, 30)),
                    (255, 0, 0),
                )

    def test_single_frame_gif_is_converted_below_compression_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "single.gif"
            image = Image.new("RGB", (32, 24), (12, 34, 56))
            image.save(source, "GIF")
            image.close()
            options = ImageCompressionOptions.from_mapping(
                {"enable": True, "min_size_mb": 100, "max_edge": 2048}
            )

            outcome = compress_local_image(str(source), root, options)

            self.assertTrue(outcome.changed)
            self.assertEqual(outcome.reason, "gif_static")
            self.assertTrue(outcome.output_path.endswith(".png"))

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

    def get_message_str(self):
        return ""

    def track_temporary_local_file(self, path: str):
        self.tracked.append(path)


class LLMImageCompressionIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = load_plugin_module()

    async def test_nested_quoted_static_image_is_added_once_with_compression_off(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "quoted.png"
            image = Image.new("RGB", (32, 24), "red")
            image.save(source, "PNG")
            image.close()
            plugin = self.mod.Main(FakeContext(), {
                "enable": False, "image_cache_dir": str(Path(root) / "cache"),
                "llm_image_compress": {"enable": False},
            })
            component = self.mod.Image.fromFileSystem(str(source))
            inner, outer = self.mod.Reply(), self.mod.Reply()
            inner.chain = [component, component]
            outer.chain = [inner]
            event = FakeCompressionEvent()
            event.get_messages = lambda: [outer]
            req = types.SimpleNamespace(image_urls=[], extra_user_content_parts=[])
            try:
                with patch.object(plugin, "_download_image_to_local", side_effect=AssertionError):
                    await plugin.on_llm_request(event, req)
                    await plugin.on_llm_request(event, req)
                self.assertEqual(req.image_urls, [str(source)])
                self.assertEqual(plugin._image_compress_count, 0)
            finally:
                await plugin.terminate()

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

            self.assertEqual(image_component.url, "")
            self.assertNotEqual(image_component.path, str(source))
            self.assertTrue(Path(image_component.path).exists())
            self.assertEqual(image_component.file, Path(image_component.path).as_uri())
            self.assertIn(
                Path(image_component.path).resolve(),
                [Path(path).resolve() for path in event.tracked],
            )

    async def test_quoted_image_files_are_promoted_by_content(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {"enable": False},
                },
            )

            try:
                for suffix, image_format in (
                    ("png", "PNG"),
                    ("jpg", "JPEG"),
                    ("webp", "WEBP"),
                    ("gif", "GIF"),
                    ("bmp", "BMP"),
                    ("tiff", "TIFF"),
                ):
                    with self.subTest(image_format=image_format):
                        source = Path(root) / f"quoted.{suffix}"
                        image = Image.new("RGB", (32, 24), (12, 34, 56))
                        image.save(source, image_format)
                        image.close()

                        file_component = self.mod.File(
                            name=source.name,
                            file=str(source),
                        )
                        reply_component = self.mod.Reply()
                        reply_component.chain = [file_component]
                        event = FakeCompressionEvent(private=True)
                        event.get_messages = lambda: [reply_component]

                        await plugin.on_message(event)
                        await plugin.on_message(event)

                        promoted = reply_component.chain[0]
                        self.assertIsInstance(promoted, self.mod.Image)
                        self.assertEqual(
                            Path(promoted.path).resolve(), source.resolve()
                        )
                        self.assertEqual(file_component.get_file_calls, 0)
            finally:
                await plugin.terminate()

    async def test_quoted_file_uses_real_content_instead_of_extension(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {"enable": False},
                },
            )
            real_image = Path(root) / "image.data"
            image = Image.new("RGB", (16, 16), (90, 80, 70))
            image.save(real_image, "PNG")
            image.close()
            fake_image = Path(root) / "not-an-image.png"
            fake_image.write_text("not an image", encoding="utf-8")

            real_file = self.mod.File(name="image.data", file=str(real_image))
            fake_file = self.mod.File(name="not-an-image.png", file=str(fake_image))
            reply_component = self.mod.Reply()
            reply_component.chain = [real_file, fake_file]
            event = FakeCompressionEvent(private=True)
            event.get_messages = lambda: [reply_component]

            try:
                await plugin.on_message(event)
            finally:
                await plugin.terminate()

            self.assertIsInstance(reply_component.chain[0], self.mod.Image)
            self.assertIs(reply_component.chain[1], fake_file)

    async def test_promoted_quoted_file_is_compressed_once_across_hooks(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "quoted-file.jpg"
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
            file_component = self.mod.File(name=source.name, file=str(source))
            reply_component = self.mod.Reply()
            reply_component.chain = [file_component]
            event = FakeCompressionEvent(private=True)
            event.get_messages = lambda: [reply_component]

            try:
                with patch.object(
                    self.mod,
                    "compress_local_image",
                    wraps=self.mod.compress_local_image,
                ) as compress:
                    await plugin.on_message(event)
                    promoted = reply_component.chain[0]
                    req = types.SimpleNamespace(
                        image_urls=[plugin._component_image_ref(promoted)],
                        extra_user_content_parts=[],
                    )
                    await plugin.on_llm_request(event, req)
            finally:
                await plugin.terminate()

            self.assertIsInstance(promoted, self.mod.Image)
            self.assertEqual(file_component.get_file_calls, 0)
            self.assertEqual(compress.call_count, 1)
            self.assertEqual(plugin._image_compress_count, 1)

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

    async def test_history_gif_is_converted_before_provider_request(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "history.gif"
            frames = [
                Image.new("RGB", (48, 32), color)
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
            raw = source.read_bytes()
            data_uri = "data:image/gif;base64," + base64.b64encode(raw).decode("ascii")
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {
                        "enable": True,
                        "min_size_mb": 100,
                        "max_edge": 2048,
                        "max_output_size_mb": 5,
                        "max_input_size_mb": 20,
                    },
                },
            )
            plugin._image_compress_output_dir = str(Path(root) / "output")
            event = FakeCompressionEvent(private=True)
            image_part = {
                "type": "image_url",
                "image_url": {"url": data_uri, "detail": "high"},
            }
            untouched_part = {
                "type": "image_url",
                "image_url": "not-a-readable-image",
            }
            req = types.SimpleNamespace(
                image_urls=[],
                contexts=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "old message"},
                            image_part,
                            untouched_part,
                        ],
                    }
                ],
                extra_user_content_parts=[],
            )

            try:
                await plugin._compress_provider_request_images(event, req)
            finally:
                await plugin.terminate()

            converted_ref = image_part["image_url"]["url"]
            self.assertNotEqual(converted_ref, data_uri)
            self.assertTrue(converted_ref.startswith("data:image/png;base64,"))
            converted_payload = base64.b64decode(converted_ref.partition(",")[2])
            converted_path = Path(root) / "converted.png"
            converted_path.write_bytes(converted_payload)
            with Image.open(converted_path) as converted:
                self.assertEqual(converted.format, "PNG")
                self.assertEqual(getattr(converted, "n_frames", 1), 1)
            self.assertEqual(image_part["image_url"]["detail"], "high")
            self.assertEqual(untouched_part["image_url"], "not-a-readable-image")
            self.assertEqual(len(event.tracked), 1)
            self.assertEqual(plugin._image_compress_count, 1)

    async def test_history_small_data_uri_remains_self_contained(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "history.png"
            image = Image.new("RGB", (32, 24), (12, 34, 56))
            image.save(source, "PNG")
            image.close()
            data_uri = "data:image/png;base64," + base64.b64encode(
                source.read_bytes()
            ).decode("ascii")
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {
                        "enable": True,
                        "min_size_mb": 100,
                        "max_edge": 2048,
                    },
                },
            )
            event = FakeCompressionEvent(private=True)
            image_part = {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
            req = types.SimpleNamespace(
                image_urls=[],
                contexts=[{"role": "user", "content": [image_part]}],
                extra_user_content_parts=[],
            )

            try:
                await plugin._compress_provider_request_images(event, req)
            finally:
                await plugin.terminate()

            prepared_ref = image_part["image_url"]["url"]
            self.assertTrue(prepared_ref.startswith("data:image/png;base64,"))
            self.assertEqual(
                base64.b64decode(prepared_ref.partition(",")[2]),
                source.read_bytes(),
            )

    async def test_stale_local_history_image_is_removed(self):
        with tempfile.TemporaryDirectory() as root:
            plugin = self.mod.Main(
                FakeContext(),
                {
                    "enable": False,
                    "image_cache_dir": str(Path(root) / "cache"),
                    "llm_image_compress": {"enable": True},
                },
            )
            stale_path = str(Path(root) / "cache" / "expired.gif")
            text_part = {"type": "text", "text": "keep me"}
            req = types.SimpleNamespace(
                image_urls=[],
                contexts=[
                    {
                        "role": "user",
                        "content": [
                            text_part,
                            {
                                "type": "image_url",
                                "image_url": {"url": stale_path},
                            },
                            {
                                "type": "image_url",
                                "image_url": f"base64:{stale_path}",
                            },
                        ],
                    }
                ],
                extra_user_content_parts=[],
            )

            try:
                await plugin._compress_provider_request_images(
                    FakeCompressionEvent(private=True),
                    req,
                )
            finally:
                await plugin.terminate()

            self.assertEqual(req.contexts[0]["content"], [text_part])

    def test_component_local_reference_uses_file_uri_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            local_path = str(Path(root) / "compressed.png")
            component = types.SimpleNamespace(file="old", url="old", path="old")

            self.mod.Main._replace_component_image_ref(component, local_path)

            self.assertEqual(component.file, Path(local_path).resolve().as_uri())
            self.assertEqual(component.url, "")
            self.assertEqual(component.path, str(Path(local_path).resolve()))

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


class GifRequestIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_vision_drops_gif_current_history_and_extra_without_sampling(self):
        from test_gif_frames import gif_bytes
        from unittest.mock import AsyncMock
        with tempfile.TemporaryDirectory() as root:
            mod = load_plugin_module()
            plugin = mod.Main(FakeContext(), {"enable": False, "image_cache_dir": root})
            provider = types.SimpleNamespace(provider_config={"id": "text-provider", "modalities": ["text", "tool_use"]}, get_model=lambda: "gemini-3-flash")
            plugin._context.get_using_provider = lambda **kwargs: provider
            source = Path(root) / "animation.data"
            source.write_bytes(gif_bytes())
            static = Path(root) / "ordinary.png"
            with Image.new("RGB", (20, 20), "red") as image:
                image.save(static)
            event = FakeCompressionEvent()
            req = types.SimpleNamespace(
                image_urls=[str(source), str(static)],
                extra_user_content_parts=[types.SimpleNamespace(image_url=types.SimpleNamespace(url=str(source)))],
                contexts=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": str(source)}},
                    {"type": "image_url", "image_url": {"url": str(static)}},
                ]}],
            )
            try:
                with patch.object(mod, "compress_local_image", side_effect=AssertionError), patch.object(mod, "prepare_gif", side_effect=AssertionError):
                    await plugin.on_llm_request(event, req)
                self.assertEqual(req.image_urls, [str(static)])
                self.assertEqual(req.contexts[0]["content"][0], {"type": "text", "text": "[动图]"})
                self.assertEqual(req.contexts[0]["content"][1]["image_url"]["url"], str(static))
                self.assertTrue(all(not hasattr(p, "image_url") for p in req.extra_user_content_parts))
                self.assertFalse(event.tracked)
                self.assertEqual(source.read_bytes(), gif_bytes())
                # Extension-based GIFs do not even materialize a remote source.
                req.image_urls = ["https://example.com/animation.gif"]
                req.contexts = []
                with patch.object(plugin, "_materialize_image_for_compression", new=AsyncMock(side_effect=AssertionError)):
                    await plugin._compress_provider_request_images(event, req)
                self.assertEqual(req.image_urls, [])
            finally:
                await plugin.terminate()

    async def test_default_grid_without_static_compression_and_unchanged_png(self):
        from test_gif_frames import gif_bytes
        with tempfile.TemporaryDirectory() as root:
            mod = load_plugin_module()
            plugin = mod.Main(FakeContext(), {"enable": False, "image_cache_dir": root})
            plugin._image_compress_output_dir = root
            source = Path(root) / "animation.data"
            source.write_bytes(gif_bytes())
            static = Path(root) / "static.png"
            with Image.new("RGB", (32, 24), "red") as image:
                image.save(static)
            event = FakeCompressionEvent(private=True)
            req = types.SimpleNamespace(image_urls=[str(source), str(static)], extra_user_content_parts=[])
            try:
                await plugin.on_llm_request(event, req)
                self.assertNotEqual(req.image_urls[0], str(source))
                self.assertEqual(req.image_urls[1], str(static))
                self.assertTrue(any("3帧" in p.text for p in req.extra_user_content_parts))
                with Image.open(req.image_urls[0]) as image:
                    self.assertEqual(image.getpixel((74, 30)), (0, 128, 0))
                self.assertEqual(source.read_bytes(), gif_bytes())
            finally:
                await plugin.terminate()

    async def test_switch_to_text_model_removes_previously_prepared_gif_frames(self):
        from test_gif_frames import gif_bytes
        with tempfile.TemporaryDirectory() as root:
            mod = load_plugin_module()
            plugin = mod.Main(FakeContext(), {"enable": False, "image_cache_dir": root, "gif_mode": "frames"})
            plugin._image_compress_output_dir = root
            provider = types.SimpleNamespace(provider_config={"modalities": ["text", "image"]}, get_model=lambda: "gpt-4o")
            plugin._context.get_using_provider = lambda **kwargs: provider
            source = Path(root) / "animation.gif"
            source.write_bytes(gif_bytes())
            event = FakeCompressionEvent()
            req = types.SimpleNamespace(
                image_urls=[str(source)],
                extra_user_content_parts=[types.SimpleNamespace(image_url=types.SimpleNamespace(url=str(source)))],
                contexts=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": str(source)}}]}],
            )
            try:
                await plugin._compress_provider_request_images(event, req)
                self.assertEqual(len(req.image_urls), 3)
                provider.provider_config["modalities"] = ["text"]
                with patch.object(mod, "compress_local_image", side_effect=AssertionError):
                    await plugin._compress_provider_request_images(event, req)
                self.assertEqual(req.image_urls, [])
                self.assertTrue(all(p.text == "[动图]" for p in req.extra_user_content_parts))
                self.assertTrue(all(p["type"] == "text" and p["text"] == "[动图]" for p in req.contexts[0]["content"]))
            finally:
                await plugin.terminate()

    async def test_frames_expand_current_history_and_extra_parts_once(self):
        from test_gif_frames import gif_bytes
        with tempfile.TemporaryDirectory() as root:
            mod = load_plugin_module()
            plugin = mod.Main(FakeContext(), {"enable": False, "gif_mode": "frames", "image_cache_dir": root})
            plugin._image_compress_output_dir = root
            source = Path(root) / "animation.gif"
            source.write_bytes(gif_bytes())
            event = FakeCompressionEvent(private=True)
            part = types.SimpleNamespace(image_url=types.SimpleNamespace(url=str(source)), type="image_url")
            req = types.SimpleNamespace(
                image_urls=[str(source)], extra_user_content_parts=[part],
                contexts=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": str(source), "detail": "high"}}]}],
            )
            try:
                await plugin._compress_provider_request_images(event, req)
                self.assertEqual(len(req.image_urls), 3)
                self.assertEqual(len(event.tracked), 3)
                image_parts = [p for p in req.contexts[0]["content"] if p["type"] == "image_url"]
                self.assertEqual(len(image_parts), 3)
                self.assertTrue(all(p["image_url"]["url"].startswith("data:image/") for p in image_parts))
                self.assertTrue(all(p["image_url"]["detail"] == "high" for p in image_parts))
                self.assertEqual(len([p for p in req.extra_user_content_parts if hasattr(p, "image_url")]), 3)
                await plugin._compress_provider_request_images(event, req)
                self.assertEqual(len(req.image_urls), 3)
                self.assertEqual(len(event.tracked), 3)
                self.assertEqual(plugin._image_compress_count, 1)
            finally:
                await plugin.terminate()

    async def test_lazy_caption_gets_all_frames_without_changing_markers(self):
        from unittest.mock import AsyncMock
        from test_gif_frames import gif_bytes
        with tempfile.TemporaryDirectory() as root:
            mod = load_plugin_module()
            plugin = mod.Main(FakeContext(), {
                "enable": False, "image_cache_dir": root, "image_caption": True,
                "image_caption_lazy": True, "show_recent_images_allow_gif": True, "gif_mode": "frames",
            })
            source = Path(root) / "caption.gif"
            source.write_bytes(gif_bytes())
            provider = types.SimpleNamespace(text_chat=AsyncMock(return_value=types.SimpleNamespace(completion_text="颜色依次变换")))
            plugin._context.get_using_provider = lambda: provider
            record = mod.MessageRecord(msg_id="gif", sender_id="a", sender_name="A", content="[图片] [图片]", timestamp=1, image_urls=[str(source), str(source)], has_image=True, image_count=2, has_gif=True, gif_count=2)
            try:
                with patch.object(mod, "Provider", types.SimpleNamespace):
                    result = await plugin._lazy_caption_flow([record])
                self.assertEqual(result[0].content.count("[图片: 颜色依次变换]"), 2)
                self.assertEqual(len(provider.text_chat.call_args.kwargs["image_urls"]), 3)
                self.assertIn("3帧", provider.text_chat.call_args.kwargs["prompt"])
                self.assertEqual(provider.text_chat.await_count, 1)
            finally:
                await plugin.terminate()

    async def test_component_and_reference_original_survive_normalization(self):
        from test_gif_frames import gif_bytes
        with tempfile.TemporaryDirectory() as root:
            mod = load_plugin_module()
            plugin = mod.Main(FakeContext(), {"image_cache_dir": root, "llm_image_compress": {"enable": True}})
            plugin._image_compress_output_dir = root
            path = Path(root) / "original.gif"
            raw = gif_bytes()
            path.write_bytes(raw)
            component = mod.Image.fromFileSystem(str(path))
            event = FakeCompressionEvent(private=True)
            event.get_messages = lambda: [component]
            before = plugin._component_image_ref(component)
            try:
                await plugin._prepare_event_images_for_llm(event)
                self.assertEqual(plugin._component_image_ref(component), before)
                req = types.SimpleNamespace(image_urls=[before], extra_user_content_parts=[])
                await plugin._compress_provider_request_images(event, req)
                self.assertNotEqual(req.image_urls[0], before)
                self.assertEqual(path.read_bytes(), raw)
            finally:
                await plugin.terminate()


class StrictReviewRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_static_paths_do_not_materialize_or_change_references(self):
        from unittest.mock import AsyncMock
        import copy

        mod = load_plugin_module()
        with tempfile.TemporaryDirectory() as root:
            plugin = mod.Main(FakeContext(), {"enable": False, "image_cache_dir": root})
            local = Path(root) / "static.png"
            with Image.new("RGB", (8, 8), "red") as image:
                image.save(local)
            refs = ["https://example.com/image.png", "https://example.com/opaque", str(local),
                    local.as_uri(), "data:image/png;base64," + base64.b64encode(local.read_bytes()).decode(),
                    "base64://" + base64.b64encode(local.read_bytes()).decode(), str(Path(root) / "missing.jpg")]
            parts = [types.SimpleNamespace(image_url=types.SimpleNamespace(url=ref)) for ref in refs]
            history = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ref, "detail": "high"}} for ref in refs]}]
            req = types.SimpleNamespace(image_urls=refs[:], contexts=copy.deepcopy(history), extra_user_content_parts=parts)
            event = FakeCompressionEvent()
            event.get_messages = lambda: [mod.Image(url=ref) for ref in refs]
            try:
                with patch.object(plugin, "_materialize_image_for_compression", new_callable=AsyncMock) as fetch:
                    await plugin._prepare_event_images_for_llm(event)
                    await plugin._compress_provider_request_images(event, req)
                fetch.assert_not_awaited()
                self.assertEqual(req.image_urls, refs)
                self.assertEqual(req.contexts, history)
                self.assertEqual([p.image_url.url for p in parts], refs)
                self.assertEqual(event.tracked, [])
            finally:
                await plugin.terminate()

    async def test_uncompressed_download_is_reused_in_current_and_history(self):
        from unittest.mock import AsyncMock

        mod = load_plugin_module()
        with tempfile.TemporaryDirectory() as root:
            local = Path(root) / "small.png"
            with Image.new("RGB", (8, 8), "red") as image:
                image.save(local)
            plugin = mod.Main(FakeContext(), {"enable": False, "image_cache_dir": root, "llm_image_compress": {"enable": True}})
            plugin._context.get_using_provider = lambda **kw: types.SimpleNamespace(
                provider_config={"modalities": ["text"]}, get_model=lambda: "text-model",
            )
            remote = "https://example.com/signed-image.png"
            extra = types.SimpleNamespace(image_url=types.SimpleNamespace(url=remote))
            req = types.SimpleNamespace(image_urls=[remote], contexts=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": remote, "detail": "high"}}]}], extra_user_content_parts=[extra])
            try:
                with patch.object(plugin, "_download_image_to_local", AsyncMock(return_value=str(local))) as fetch:
                    await plugin._compress_provider_request_images(FakeCompressionEvent(), req)
                fetch.assert_awaited_once()
                self.assertEqual(req.image_urls, [str(local)])
                part = req.contexts[0]["content"][0]["image_url"]
                self.assertEqual(base64.b64decode(part["url"].split(",", 1)[1]), local.read_bytes())
                self.assertEqual(part["detail"], "high")
                self.assertEqual(extra.image_url.url, remote)  # 3.7.1 does not alter extra static ImageURLParts.
            finally:
                await plugin.terminate()

    async def test_cancel_survives_worker_and_cleanup_failures(self):
        import asyncio
        import threading

        mod = load_plugin_module()
        for failure in ("worker", "unlink", "tracking"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                plugin = mod.Main(FakeContext(), {"image_cache_dir": root, "llm_image_compress": {"enable": True}})
                source = Path(root) / "static.png"
                source.write_bytes(b"static")
                started, finish = threading.Event(), threading.Event()

                def worker(*args):
                    started.set()
                    finish.wait(5)
                    if failure == "worker":
                        raise ValueError("worker failed")
                    return types.SimpleNamespace(changed=True, output_paths=(str(Path(root) / "output.png"),))

                event = FakeCompressionEvent()
                try:
                    with patch.object(mod, "compress_local_image", worker), patch.object(Path, "unlink", side_effect=OSError("cleanup failed")), patch.object(plugin, "_track_temporary_image", side_effect=RuntimeError if failure == "tracking" else None):
                        task = asyncio.create_task(plugin._compress_image_reference(event, str(source)))
                        self.assertTrue(await asyncio.to_thread(started.wait, 5))
                        task.cancel()
                        finish.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                finally:
                    finish.set()
                    await plugin.terminate()

    async def test_new_remote_gif_uses_checked_downloader(self):
        from unittest.mock import AsyncMock
        from test_gif_frames import gif_bytes

        mod = load_plugin_module()
        with tempfile.TemporaryDirectory() as root:
            plugin = mod.Main(FakeContext(), {"image_cache_dir": root})
            plugin._image_compress_output_dir = root
            try:
                with patch.object(mod.ImageIndex, "_fetch", AsyncMock(return_value=gif_bytes())) as fetch, patch.object(plugin, "_download_image_to_local", side_effect=AssertionError("legacy downloader")):
                    ref = await plugin._compress_image_reference(FakeCompressionEvent(), "https://example.com/animated.gif")
                fetch.assert_awaited_once_with("https://example.com/animated.gif")
                self.assertTrue(Path(ref).is_file())
            finally:
                await plugin.terminate()
