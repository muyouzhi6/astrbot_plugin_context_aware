from __future__ import annotations

import asyncio
import base64
import io
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from image_context import (
    MIB,
    SEEN_KEY,
    SNAPSHOT_KEY,
    TOOL_NAME,
    ImageIndex,
    encode_image,
    select_automatic,
    strip_tool_images,
)
from test_gemini_stt_context import FakeContext, FakeEvent, load_plugin_module


def png(color="red", size=(20, 20)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def message(msg_id="m1", sender="a", *, count=1, timestamp=None, content="[图片]"):
    source = "data:image/png;base64," + base64.b64encode(png()).decode()
    return SimpleNamespace(
        msg_id=msg_id,
        sender_id=sender,
        sender_name=sender,
        timestamp=time.time() if timestamp is None else timestamp,
        is_bot=False,
        has_gif=False,
        image_urls=[source] * count,
        image_count=count,
        has_image=bool(count),
        content=content,
        reply_to_id=None,
        at_targets=[],
    )


class IndexTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.index = ImageIndex(per_session=3, max_sessions=2)

    async def asyncTearDown(self):
        await self.index.close()

    async def test_scope_watermark_and_snapshot_stability(self):
        first = self.index.add("g1", message(), 1)
        snapshot = self.index.snapshot("g1", 2)
        self.index.add("g1", message("m2"), 3)
        self.assertEqual(snapshot, first)
        self.assertEqual(self.index.snapshot("g1", 2), first)
        self.assertIsNone(await self.index.read("g2", first[0]))

    async def test_dedup_batches_and_capacity(self):
        ids = self.index.add("g1", message(count=2), 1)
        self.assertEqual(self.index.add("g1", message(count=2), 1), ids)
        self.index.add("g1", message("m2", count=2), 2)
        self.assertEqual(len(self.index.sessions["g1"]), 3)
        self.index.add("g2", message(), 3)
        self.index.add("g3", message(), 4)
        self.assertNotIn("g1", self.index.sessions)

    async def test_ttl_and_reset_invalidate_existing_snapshot(self):
        ids = self.index.add("g1", message(), 1)
        self.index.clear("g1")
        self.assertIsNone(await self.index.read("g1", ids[0]))
        self.index.add("g1", message(timestamp=time.time() - 3600), 2)
        self.assertFalse(self.index.snapshot("g1", 3))

    async def test_cache_budget_counts_sources_and_bytes(self):
        self.index.budget = 1000
        self.index.add("g1", message(count=3), 1)
        self.assertLessEqual(self.index.byte_count, 1000)
        for i in self.index.snapshot("g1", 2):
            await self.index.read("g1", i)
        self.assertLessEqual(self.index.byte_count, 1000)

    async def test_prefetch_does_not_block_and_read_shares_fetch(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def fetch(source):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return png()

        self.index._fetch = fetch
        ids = self.index.add("g1", message(), 1)
        self.index.prefetch(ids)
        await entered.wait()
        readers = [asyncio.create_task(self.index.read("g1", ids[0])) for _ in range(2)]
        await asyncio.sleep(0)
        release.set()
        self.assertTrue(all(await asyncio.gather(*readers)))
        self.assertEqual(calls, 1)
        self.assertEqual(self.index.resources[ids[0]].source, "")

    async def test_reset_during_download_does_not_resurrect(self):
        release = asyncio.Event()

        async def fetch(source):
            await release.wait()
            return png()

        self.index._fetch = fetch
        ids = self.index.add("g1", message(), 1)
        reader = asyncio.create_task(self.index.read("g1", ids[0]))
        await asyncio.sleep(0)
        self.index.clear("g1")
        release.set()
        self.assertIsNone(await reader)
        self.assertFalse(self.index.resources)

    async def test_cancelled_reader_keeps_shared_fetch(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def fetch(source):
            entered.set()
            await release.wait()
            return png()

        self.index._fetch = fetch
        ids = self.index.add("g1", message(), 1)
        reader = asyncio.create_task(self.index.read("g1", ids[0]))
        await entered.wait()
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reader
        release.set()
        self.assertIsNotNone(await self.index.read("g1", ids[0]))

    async def test_failure_is_bounded_and_retryable(self):
        self.index._fetch = AsyncMock(side_effect=ValueError("broken"))
        ids = self.index.add("g1", message(), 1)
        self.assertIsNone(await self.index.read("g1", ids[0]))
        self.assertIsNone(await self.index.read("g1", ids[0]))
        self.assertEqual(self.index._fetch.await_count, 1)
        self.index.resources[ids[0]].retry_at = 0
        self.index._fetch = AsyncMock(return_value=png())
        self.assertIsNotNone(await self.index.read("g1", ids[0]))

    async def test_pending_prefetch_tasks_are_bounded(self):
        self.index.per_session = 100
        for n in range(30):
            self.index.prefetch(self.index.add("g1", message(str(n)), n))
        self.assertLessEqual(len(self.index.tasks), 16)

    async def test_private_network_and_oversized_invalid_sources_fail(self):
        for source in (
            "http://127.0.0.1/x",
            "http://169.254.169.254/x",
            "http://[::1]/x",
        ):
            with self.assertRaises(ValueError):
                await self.index._fetch(source)
        self.index.max_download = 3
        with self.assertRaises(ValueError):
            await self.index._fetch("base64://" + base64.b64encode(b"four").decode())


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.index = ImageIndex()
        self.ids = self.index.add("g", message(), 1)
        self.entry = self.index.get("g", self.ids[0]).entry
        self.question = message("q", count=0, content="图里是什么？")

    def test_simple_visual_question_selects_own_unique_image(self):
        self.assertEqual(select_automatic([self.entry], self.question), self.entry)
        for text in ("你看图里是什么？", "你帮我看看刚才那张图"):
            self.question.content = text
            self.assertEqual(select_automatic([self.entry], self.question), self.entry)

    def test_short_deictic_question_is_left_to_model_tool_selection(self):
        self.question.content = "帮我看下这张"
        self.assertIsNone(select_automatic([self.entry], self.question))

    def test_ambiguous_other_sender_requires_explicit_ownership(self):
        other_id = self.index.add("g", message("m2", "b"), 2)[0]
        other = self.index.get("g", other_id).entry
        self.assertIsNone(select_automatic([self.entry, other], self.question))
        self.question.content = "我刚发的图是什么？"
        self.assertEqual(
            select_automatic([self.entry, other], self.question), self.entry
        )

    def test_unrelated_generation_old_multiple_and_foreign_reference_skip(self):
        for text in (
            "今天吃什么",
            "帮我生成一张图",
            "对比这两张图",
            "昨天的图是什么",
            "小王的图是什么",
        ):
            self.question.content = text
            self.assertIsNone(select_automatic([self.entry], self.question), text)
        self.question.content = "图里是什么"
        self.assertIsNone(
            select_automatic([self.entry], self.question, now=time.time() + 200)
        )
        self.assertIsNone(select_automatic([self.entry, self.entry], self.question))

    def test_native_and_quote_are_never_auto_replaced(self):
        self.question.has_image = True
        self.assertIsNone(select_automatic([self.entry], self.question))
        self.question.has_image = False
        self.question.reply_to_id = "a"
        self.assertIsNone(select_automatic([self.entry], self.question))


class HistoryTests(unittest.TestCase):
    def test_exact_tool_image_pair_only_and_no_shared_mutation(self):
        path = "/tmp/tool_image.jpg"
        own = [
            {"type": "text", "text": f"[Image from tool '{TOOL_NAME}', path='{path}']"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA", "id": path},
            },
        ]
        foreign = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "native"}}],
        }
        history = [
            {"role": "assistant", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "image viewed"},
            {"role": "user", "content": own},
            foreign,
        ]
        result, count = strip_tool_images(history)
        self.assertEqual(count, 1)
        self.assertEqual(len(history[2]["content"]), 2)
        self.assertEqual(result[:2], history[:2])
        self.assertEqual(result[3], foreign)
        self.assertNotIn("image_url", str(result[2]))
        self.assertEqual(strip_tool_images(result)[1], 0)

    def test_foreign_tool_marker_and_mismatched_id_keep_image(self):
        for name, image_id in (("another_tool", "/tmp/x"), (TOOL_NAME, "/tmp/y")):
            history = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[Image from tool '{name}', path='/tmp/x']",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "native", "id": image_id},
                        },
                    ],
                }
            ]
            self.assertEqual(strip_tool_images(history)[1], 0)

    def test_payload_resolution_and_size_budget(self):
        raw = png(size=(4500, 1000))
        for detail, edge in (("auto", 1600), ("high", 4096)):
            data, mime = encode_image(raw, detail)
            decoded = base64.b64decode(data)
            self.assertLessEqual(len(decoded), 2 * MIB)
            self.assertEqual(mime, "image/jpeg")
            with Image.open(io.BytesIO(decoded)) as image:
                self.assertEqual(image.width, edge)


class PluginToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mod = load_plugin_module()
        self.plugin = self.mod.Main(FakeContext())
        self.plugin._recall_supports_vision = lambda event: True
        self.event = FakeEvent()
        self.ids = self.plugin._image_index.add(
            self.event.unified_msg_origin, message(), 1
        )
        self.event.set_extra(SNAPSHOT_KEY, self.ids)
        self.event.set_extra(SEEN_KEY, {})

    async def asyncTearDown(self):
        await self.plugin.terminate()

    async def test_tool_returns_real_image_dedups_and_allows_upgrade(self):
        call = self.plugin.context_aware_view_images
        result = await call(self.event, list(self.ids))
        self.assertEqual([p.type for p in result.content], ["text", "image"])
        self.assertTrue(base64.b64decode(result.content[1].data))
        result = await call(self.event, list(self.ids))
        self.assertEqual([p.type for p in result.content], ["text"])
        result = await call(self.event, list(self.ids), "high")
        self.assertEqual([p.type for p in result.content], ["text", "image"])

    async def test_invalid_id_url_and_reset_are_rejected(self):
        for ids in (["https://example.com/image"], ["ca_other_group"], "not-list"):
            result = await self.plugin.context_aware_view_images(self.event, ids)
            self.assertFalse(any(p.type == "image" for p in result.content))
        await self.plugin._clear_session_context(self.event, "test")
        result = await self.plugin.context_aware_view_images(self.event, list(self.ids))
        self.assertFalse(any(p.type == "image" for p in result.content))

    async def test_concurrent_same_event_tool_calls_return_image_once(self):
        results = await asyncio.gather(
            *[
                self.plugin.context_aware_view_images(self.event, list(self.ids))
                for _ in range(2)
            ]
        )
        self.assertEqual(sum(p.type == "image" for r in results for p in r.content), 1)

    async def test_separate_events_can_each_view_same_image(self):
        other = FakeEvent()
        other.set_extra(SNAPSHOT_KEY, self.ids)
        results = await asyncio.gather(
            *[
                self.plugin.context_aware_view_images(event, list(self.ids))
                for event in (self.event, other)
            ]
        )
        self.assertEqual(sum(p.type == "image" for r in results for p in r.content), 2)

    async def test_non_visual_provider_never_claims_image_was_seen(self):
        self.plugin._recall_supports_vision = lambda event: False
        result = await self.plugin.context_aware_view_images(self.event, list(self.ids))
        self.assertEqual(result.content[0].type, "text")
        self.assertIn("未配置视觉能力", result.content[0].text)

    async def test_failed_tool_image_is_attempted_once_per_request(self):
        self.plugin._image_index.read = AsyncMock(return_value=None)
        await self.plugin.context_aware_view_images(self.event, list(self.ids))
        result = await self.plugin.context_aware_view_images(
            self.event, list(self.ids), "high"
        )
        self.assertEqual(self.plugin._image_index.read.await_count, 1)
        self.assertIn("不可用", result.content[0].text)

    async def test_reset_during_message_extraction_revokes_recording(self):
        old_event = FakeEvent()
        self.plugin._stamp_image_event(old_event)
        await self.plugin._clear_session_context(old_event, "test")
        self.plugin._record_recall_images(old_event, message("late"))
        self.assertFalse(self.plugin._image_index.resources)


if __name__ == "__main__":
    unittest.main()
