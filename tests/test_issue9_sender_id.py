"""Issue #9: optional sender IDs through the message and LLM hooks."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from test_gemini_stt_context import FakeContext, FakeEvent, load_plugin_module


class SenderIdTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mod = load_plugin_module()
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)

    def event(self, sender="123456", name="同名群友", mid="m1", private=False):
        event = FakeEvent()
        event.extras = {}
        event.message_str = "你好"
        event.message_obj = SimpleNamespace(message_id=mid)
        event.get_sender_id = lambda: sender
        event.get_sender_name = lambda: name
        event.get_messages = lambda: [self.mod.Plain("你好")]
        event.is_private_chat = lambda: private
        if private:
            event.unified_msg_origin = "aiocqhttp:private:123456"
        return event

    async def scene(self, config, *, private=False):
        plugin = self.mod.Main(
            FakeContext(), {"image_cache_dir": self.root.name, **config}
        )
        try:
            await plugin.on_message(self.event(mid="m1", private=private))
            current = self.event(sender="654321", mid="m2", private=private)
            await plugin.on_message(current)
            request = self.mod.ProviderRequest()
            await plugin.on_llm_request(current, request)
            self.assertEqual(plugin._stats.scenes_injected, 1)
            snapshot = await plugin._sessions.get_snapshot_async(current.unified_msg_origin)
            self.assertEqual([m.sender_name for m in snapshot.messages], ["同名群友"] * 2)
            return "\n".join(p.text for p in request.extra_user_content_parts)
        finally:
            await plugin.terminate()

    async def test_enabled_disambiguates_group_senders(self):
        scene = await self.scene({"show_sender_id": True})
        self.assertIn("同名群友 (123456)", scene)
        self.assertIn("<sender>同名群友 (654321)</sender>", scene)
        tree = ET.fromstring(scene)
        self.assertEqual(len(tree.findall("recent_flow/m")), 2)
        self.assertIn("同名群友 (123456)", tree.findtext("participants"))
        self.assertIn("同名群友 (654321)", tree.findtext("participants"))

    async def test_default_and_explicit_off_keep_existing_scene(self):
        default = await self.scene({})
        self.assertEqual(default, await self.scene({"show_sender_id": False}))
        self.assertNotIn("123456", default)
        self.assertNotIn("654321", default)

    async def test_enabled_keeps_private_scene_unchanged(self):
        config = {"enable_private_chat": True}
        off = await self.scene(config, private=True)
        on = await self.scene({**config, "show_sender_id": True}, private=True)
        self.assertEqual(on, off)

    async def test_invalid_config_does_not_enable_id_display(self):
        for value in (None, "false", "true", "", 0, 1, [], {"enabled": True}):
            with self.subTest(value=value):
                scene = await self.scene({"show_sender_id": value})
                self.assertNotIn("123456", scene)
                self.assertNotIn("654321", scene)

    async def test_images_voice_bot_and_xml_escaping_through_hooks(self):
        plugin = self.mod.Main(FakeContext(), {
            "image_cache_dir": self.root.name,
            "show_sender_id": True,
            "image_recall": False,
        })
        self.addAsyncCleanup(plugin.terminate)
        name, sender = '群友<&"', 'id<&"'
        image_event = self.event(sender, name, "image")
        image_event.message_str = "[图片]"
        image_event.get_messages = lambda: [self.mod.Image(file="missing.png")]
        await plugin.on_message(image_event)
        voice_event = self.event(sender, name, "voice")
        voice_event.set_extra("_gemini_stt_transcript", "语音里的文字")
        await plugin.on_message(voice_event)
        await plugin.on_message(self.event("bot", "机器人", "bot"))
        current = self.event(sender, name, "current")
        await plugin.on_message(current)
        request = self.mod.ProviderRequest()
        await plugin.on_llm_request(current, request)
        text = "\n".join(p.text for p in request.extra_user_content_parts)
        tree = ET.fromstring(text)
        label = f"{name} ({sender})"
        self.assertEqual(tree.findtext("current_message/sender"), label)
        self.assertEqual(tree.find("recent_images/image").get("sender"), label)
        self.assertEqual(tree.find("recent_voice_transcripts/voice").get("sender"), label)
        self.assertTrue(any(m.text.startswith("[你] →") for m in tree.findall("recent_flow/m")))
        self.assertNotIn("机器人 (bot)", text)
        self.assertNotIn("<sender>群友<", text)
        # A repeated LLM hook uses the same stored names, without double suffixes.
        request = self.mod.ProviderRequest()
        await plugin.on_llm_request(current, request)
        self.assertEqual(text, "\n".join(p.text for p in request.extra_user_content_parts))

    async def test_missing_name_id_fallback_and_long_identity(self):
        plugin = self.mod.Main(FakeContext(), {
            "image_cache_dir": self.root.name, "show_sender_id": True,
        })
        self.addAsyncCleanup(plugin.terminate)
        for i, (sender, name, expected) in enumerate((
            ("123456", "", "123456"),
            ("123456", "123456", "123456"),
            ("", "匿名", "匿名"),
            ("0", "零号", "零号 (0)"),
            ("u" * 200, "名" * 500, "名" * 500 + " (" + "u" * 200 + ")"),
        )):
            with self.subTest(case=i):
                event = self.event(sender, name, f"edge-{i}")
                await plugin.on_message(event)
                request = self.mod.ProviderRequest()
                await plugin.on_llm_request(event, request)
                tree = ET.fromstring("\n".join(p.text for p in request.extra_user_content_parts))
                self.assertEqual(tree.findtext("current_message/sender"), expected)

    def test_schema_defaults_off(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
        )
        self.assertIn("show_sender_id", schema)
        self.assertEqual(schema["show_sender_id"]["type"], "bool")
        self.assertIs(schema["show_sender_id"]["default"], False)
