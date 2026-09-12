"""Exercise the command and injection hooks against a real temporary SQLite store."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_gemini_stt_context import load_plugin_module


class Event:
    def __init__(self, text="continue", umo="onebot:private:alice", private=True):
        self.text, self.unified_msg_origin, self.private = text, umo, private
        self.message_obj = SimpleNamespace(message_id="current")
        self.extras = {}
        self.stopped = False

    def get_message_str(self):
        return self.text

    def is_private_chat(self):
        return self.private

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text):
        return text


class ShortTermMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mod = load_plugin_module()
        self.mod.get_astrbot_plugin_data_path = lambda: self.tmp.name
        self.manager = SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="a"),
            get_conversation=AsyncMock(
                return_value=SimpleNamespace(history='[{"role":"user"}]')
            ),
        )
        self.config = {
            "image_cache_dir": self.tmp.name,
            "private_topic": {"enabled": True},
        }
        self.plugin = self.mod.Main(
            SimpleNamespace(conversation_manager=self.manager), self.config
        )
        self.memory = self.plugin._topic
        self.memory.closed = True
        self.memory.generate = AsyncMock(side_effect=AssertionError("no model calls"))
        self.memory.candidates = AsyncMock(side_effect=AssertionError("no providers"))
        await self.memory.start(run_worker=False)
        self.log = Mock()
        self.mod.logger = self.log
        self.umo = Event().unified_msg_origin

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.tmp.cleanup()

    async def seed(self, umo=None, cid="a", text="private secret", event_id="seed"):
        receipt = await self.memory.begin(umo or self.umo, cid, event_id, text)
        await self.memory.finish(receipt, "assistant suggestion")
        return receipt

    def request(self, cid="a", history=None):
        return SimpleNamespace(
            conversation=SimpleNamespace(cid=cid, history=history),
            contexts=[],
            extra_user_content_parts=[],
        )

    async def inspect(self, event=None):
        event = event or Event("短期记忆")
        chunks = [chunk async for chunk in self.plugin.show_short_term_memory(event)]
        self.assertTrue(event.stopped)
        self.memory.generate.assert_not_awaited()
        self.memory.candidates.assert_not_awaited()
        return chunks

    async def test_snapshot_matches_actual_parts_and_default_logs_hide_body(self):
        await self.seed()
        event, req = Event(), self.request()
        await self.plugin._topic_request(event, req)
        content = req.extra_user_content_parts[0].text
        self.assertIn("private secret", content)
        self.assertEqual(
            (await self.memory.last_injection(self.umo, "a"))["text"], content
        )
        self.assertIn(content, "".join(await self.inspect()))
        self.assertNotIn("private secret", str(self.log.mock_calls))
        self.assertNotIn(self.umo, str(self.log.mock_calls))

    async def test_log_opt_in_is_single_line_json_and_repeat_hook_is_idempotent(self):
        self.config["private_topic"]["log_injected_content"] = True
        await self.seed(text='secret\nforged log\r\n"quoted"')
        event, req = Event(), self.request()
        for _ in range(2):
            await self.plugin._topic_request(event, req)
        records = [
            c.args[0]
            for c in self.log.info.call_args_list
            if "injection_snapshot=" in c.args[0]
        ]
        self.assertEqual(len(records), 1)
        self.assertNotIn("\n", records[0])
        self.assertNotIn("\r", records[0])
        data = json.loads(records[0].split("injection_snapshot=", 1)[1])
        self.assertEqual(data["content"], req.extra_user_content_parts[0].text)
        self.assertEqual(data["session"], self.umo)
        self.assertEqual(data["conversation_id"], "a")
        self.assertEqual(len(req.extra_user_content_parts), 1)

    async def test_truthy_malformed_logging_values_do_not_disclose_content(self):
        await self.seed()
        for value in ("false", "true", 1, [], {}):
            self.config["private_topic"]["log_injected_content"] = value
            await self.plugin._topic_request(Event(), self.request())
        self.assertNotIn("private secret", str(self.log.mock_calls))
        self.assertNotIn("injection_snapshot=", str(self.log.mock_calls))

    async def test_command_does_not_archive_or_prepare_images(self):
        self.plugin._prepare_event_images_for_llm = AsyncMock()
        for text in ("短期记忆", "/短期记忆", "  /短期记忆  "):
            event, req = Event(text), self.request()
            await self.plugin.on_message(event)
            await self.plugin.on_llm_request(event, req)
            await self.inspect(event)
            self.assertFalse(req.extra_user_content_parts)
        self.plugin._prepare_event_images_for_llm.assert_not_awaited()
        count = await self.memory.store.run(
            lambda db: db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        )
        self.assertEqual(count, 0)
        for text in ("//短期记忆", "短期记忆是什么", "请看短期记忆"):
            self.assertFalse(self.plugin._is_topic_inspect(Event(text)))

    async def test_group_command_never_reads_private_store(self):
        self.memory.last_injection = AsyncMock(side_effect=AssertionError())
        text = "".join(await self.inspect(Event("短期记忆", private=False)))
        self.assertIn("请在私聊", text)
        self.manager.get_curr_conversation_id.assert_not_awaited()
        self.memory.last_injection.assert_not_awaited()

    async def test_disabled_and_empty_states(self):
        self.assertIn("暂无注入记录", "".join(await self.inspect()))
        await self.plugin._topic_request(Event(), self.request())
        self.assertIn("未注入话题资料", "".join(await self.inspect()))
        self.plugin._topic = None
        try:
            self.assertIn("未启用", "".join(await self.inspect()))
        finally:
            self.plugin._topic = self.memory

    async def test_umo_and_conversation_isolation(self):
        for umo, cid, content in (
            (self.umo, "a", "alice-a"),
            (self.umo, "b", "alice-b"),
            ("onebot:private:bob", "a", "bob-a"),
        ):
            receipt = await self.seed(umo, cid)
            await self.memory.save_injection(receipt, content)
        self.assertIn("alice-a", "".join(await self.inspect()))
        self.manager.get_curr_conversation_id.return_value = "b"
        reply = "".join(await self.inspect())
        self.assertIn("alice-b", reply)
        self.assertNotIn("alice-a", reply)
        self.assertNotIn("bob-a", reply)
        self.assertIsNone(await self.memory.last_injection("other-platform:alice", "b"))
        self.manager.get_curr_conversation_id.return_value = None
        self.assertIn("暂无注入记录", "".join(await self.inspect()))

    async def test_reused_event_cannot_inject_previous_conversation(self):
        await self.seed(text="alice-a-secret")
        event = Event()
        first = self.request()
        await self.plugin._topic_request(event, first)
        self.assertIn("alice-a-secret", first.extra_user_content_parts[0].text)
        second = self.request(cid="b")
        await self.plugin._topic_request(event, second)
        self.assertEqual(second.extra_user_content_parts, [])
        event.unified_msg_origin = "onebot:private:bob"
        third = self.request()
        await self.plugin._topic_request(event, third)
        self.assertEqual(third.extra_user_content_parts, [])

    async def test_snapshot_survives_reload_and_not_recomputed_after_summary_update(
        self,
    ):
        receipt = await self.seed()
        await self.memory.save_injection(receipt, "exact old injection")
        await self.memory.store.run(
            lambda db: (
                db.execute(
                    "UPDATE sessions SET checkpoint=?", ('[{"text":"future summary"}]',)
                ).rowcount
            )
        )
        self.assertIn("exact old injection", "".join(await self.inspect()))
        restored = type(self.memory)(
            self.memory.store.path,
            self.memory.settings,
            self.memory.candidates,
            self.memory.generate,
            lambda *a: None,
        )
        try:
            self.assertEqual(
                (await restored.last_injection(self.umo, "a"))["text"],
                "exact old injection",
            )
            self.assertIsNone(restored.task)
        finally:
            await restored.close()

    async def test_out_of_order_and_stale_receipts_cannot_replace_snapshot(self):
        old = await self.seed()
        new = await self.seed(event_id="new")
        self.assertTrue(await self.memory.save_injection(new, "new"))
        self.assertFalse(await self.memory.save_injection(old, "old"))
        self.assertEqual(
            (await self.memory.last_injection(self.umo, "a"))["text"], "new"
        )
        forged = (new[0], new[1], new[2] + 999)
        self.assertFalse(await self.memory.save_injection(forged, "forged"))
        await self.memory.clear(self.umo, "a")
        await self.seed(event_id="after-reset")
        self.assertFalse(await self.memory.save_injection(new, "stale"))
        self.assertIsNone(await self.memory.last_injection(self.umo, "a"))

    async def test_dashboard_reset_hides_and_next_request_clears_snapshot(self):
        receipt = await self.seed()
        await self.memory.save_injection(receipt, "secret before reset")
        self.manager.get_conversation.return_value = SimpleNamespace(history="[]")
        self.assertNotIn("secret before reset", "".join(await self.inspect()))
        req = self.request(history="[]")
        await self.plugin._topic_request(Event(), req)
        self.assertEqual(req.extra_user_content_parts, [])
        self.assertEqual((await self.memory.last_injection(self.umo, "a"))["text"], "")

    async def test_retention_eviction_cascades_to_snapshot(self):
        receipt = await self.seed()
        await self.memory.save_injection(receipt, "expired secret")
        await self.memory.store.run(
            lambda db: db.execute("UPDATE sessions SET updated=0").rowcount
        )
        await self.memory.store.run(self.memory.store.maintain)
        self.assertIsNone(await self.memory.last_injection(self.umo, "a"))

    async def test_long_snapshot_is_chunked_without_content_loss(self):
        receipt = await self.seed()
        content = "sensitive detail " * 350
        await self.memory.save_injection(receipt, content)
        chunks = await self.inspect()
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 1500 for c in chunks))
        self.assertTrue("".join(chunks).endswith(content))

    async def test_errors_do_not_leak_exception_body_and_cancellation_propagates(self):
        self.manager.get_curr_conversation_id.side_effect = RuntimeError("secret token")
        self.assertNotIn("secret token", "".join(await self.inspect()))
        self.assertNotIn("secret token", str(self.log.mock_calls))
        self.manager.get_curr_conversation_id.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.inspect()
        with patch.object(
            self.memory, "render", AsyncMock(side_effect=RuntimeError("secret payload"))
        ):
            await self.plugin._topic_request(Event(), self.request())
        self.assertNotIn("secret payload", str(self.log.mock_calls))

    async def test_group_and_missing_cid_requests_never_open_topic_session(self):
        with patch.object(self.memory, "begin", AsyncMock()) as begin:
            await self.plugin._topic_request(Event(private=False), self.request())
            await self.plugin._topic_request(Event(), self.request(cid=None))
            begin.assert_not_awaited()

    def test_schema_does_not_enable_body_logging_by_default(self):
        schema = json.loads(
            (Path(__file__).parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIs(
            schema["private_topic"]["items"]["log_injected_content"]["default"], False
        )


if __name__ == "__main__":
    unittest.main()
