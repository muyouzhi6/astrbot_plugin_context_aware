"""Issue #13: infer_addressee must run when on_llm_request precedes on_message."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

from test_gemini_stt_context import FakeContext, FakeEvent, load_plugin_module


class Issue13AddresseeFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mod = load_plugin_module()
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)

    def _event(
        self,
        *,
        sender_id: str,
        sender_name: str,
        msg_id: str,
        components: list,
        message_str: str,
        is_wake: bool = False,
    ):
        event = FakeEvent()
        event.extras = {}
        event.is_at_or_wake_command = is_wake
        event.message_str = message_str
        event.message_obj = SimpleNamespace(
            message_id=msg_id, message_str=message_str
        )
        event.get_sender_id = lambda: sender_id
        event.get_sender_name = lambda: sender_name
        event.get_messages = lambda: list(components)
        return event

    async def test_llm_request_before_on_message_infers_at_other_and_resolves_name(
        self,
    ):
        plugin = self.mod.Main(
            FakeContext(),
            {"image_cache_dir": self.root.name, "strict_mode": True},
        )
        self.addAsyncCleanup(plugin.terminate)

        # Target user spoke earlier so history knows their display name.
        prior = self._event(
            sender_id="3457374425",
            sender_name="小王",
            msg_id="m_prior",
            components=[self.mod.Plain("刚下课")],
            message_str="刚下课",
        )
        await plugin.on_message(prior)

        # Reproduce issue #13 order: AstrBot runs on_llm_request BEFORE context_aware.on_message.
        trigger_event = self._event(
            sender_id="1731538334",
            sender_name="阿甲",
            msg_id="m_trigger",
            components=[
                self.mod.Reply(sender_id="999999"),
                self.mod.At(qq="3457374425", name=""),
                self.mod.Plain(" 你们期中了吗"),
            ],
            message_str="你们期中了吗",
            is_wake=False,
        )
        req = self.mod.ProviderRequest()
        await plugin.on_llm_request(trigger_event, req)

        current = trigger_event.get_extra(self.mod.ExtraKeys.CURRENT_MESSAGE_RECORD)
        self.assertIsNotNone(current)
        self.assertEqual(current.talking_to, "3457374425")
        self.assertEqual(current.talking_to_name, "小王")

        scene = "\n".join(part.text for part in req.extra_user_content_parts)
        self.assertIn("阿甲 正在和 小王 对话，不是在问你", scene)
        self.assertNotIn("这条消息是说给群里的", scene)

        # When context_aware.on_message runs later in the same pipeline, it must not duplicate.
        recorded_before = plugin._stats.messages_recorded
        await plugin.on_message(trigger_event)
        self.assertEqual(plugin._stats.messages_recorded, recorded_before)
        snapshot = await plugin._sessions.get_snapshot_async(
            trigger_event.unified_msg_origin
        )
        self.assertEqual(len(snapshot.messages), 2)

    async def test_llm_request_before_on_message_infers_reply_to_other(self):
        plugin = self.mod.Main(
            FakeContext(),
            {"image_cache_dir": self.root.name},
        )
        self.addAsyncCleanup(plugin.terminate)

        prior = self._event(
            sender_id="222222",
            sender_name="Aurora",
            msg_id="m1",
            components=[self.mod.Plain("👀")],
            message_str="👀",
        )
        await plugin.on_message(prior)

        reply_event = self._event(
            sender_id="1731538334",
            sender_name="阿甲",
            msg_id="m2",
            components=[
                self.mod.Reply(sender_id="222222"),
                self.mod.Plain("你们期中了吗"),
            ],
            message_str="你们期中了吗",
            is_wake=False,
        )
        req = self.mod.ProviderRequest()
        await plugin.on_llm_request(reply_event, req)

        current = reply_event.get_extra(self.mod.ExtraKeys.CURRENT_MESSAGE_RECORD)
        self.assertEqual(current.talking_to, "222222")
        self.assertEqual(current.talking_to_name, "Aurora")
        scene = "\n".join(part.text for part in req.extra_user_content_parts)
        self.assertIn("阿甲 正在和 Aurora 对话，不是在问你", scene)


if __name__ == "__main__":
    unittest.main()
