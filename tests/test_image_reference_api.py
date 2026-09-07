import time
import unittest
from types import SimpleNamespace

from test_gemini_stt_context import FakeContext, FakeEvent, load_plugin_module
from test_image_context import message, png
from image_context import SNAPSHOT_KEY


class ReferenceAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mod = load_plugin_module()
        self.plugin = self.mod.Main(FakeContext())

    async def asyncTearDown(self):
        await self.plugin.terminate()

    def event(self, sender="a", group="g", mid="q", cid="c1"):
        event = FakeEvent()
        event.unified_msg_origin = group
        event.message_obj = SimpleNamespace(message_id=mid)
        event.get_sender_id = lambda: sender
        event.get_sender_name = lambda: sender
        event.set_extra(
            "provider_request", SimpleNamespace(conversation=SimpleNamespace(cid=cid))
        )
        return event

    async def add_source(self, sender="a", group="g", mid="source"):
        event = self.event(sender, group, mid)
        await self.plugin._record_recall_images(event, message(mid, sender))
        return self.plugin.get_image_catalog(event)[0]["id"]

    async def test_same_group_explicit_other_sender_export_and_preview_independence(
        self,
    ):
        image_id = await self.add_source()
        event = self.event("b")
        result = await self.plugin.resolve_reference_images(event, [image_id])
        self.assertEqual(result[0]["data"], png())
        self.assertEqual(result[0]["sender_id"], "a")
        other = self.event("b", "other-group")
        with self.assertRaises(ValueError):
            await self.plugin.resolve_reference_images(other, [image_id])

    async def test_snapshot_rejects_future_and_reset_invalidates_exports(self):
        first = await self.add_source()
        event = self.event()
        self.plugin.get_image_catalog(event)
        later_event = self.event(mid="later")
        await self.plugin._record_recall_images(later_event, message("later"))
        second = self.plugin._image_index.sessions["g"][-1]
        with self.assertRaises(ValueError):
            await self.plugin.resolve_reference_images(event, [second])
        self.plugin._image_index.clear("g")
        with self.assertRaises(ValueError):
            await self.plugin.resolve_reference_images(event, [first])

    async def test_all_or_nothing_missing_reference(self):
        first = await self.add_source()
        event = self.event()
        with self.assertRaises(ValueError):
            await self.plugin.resolve_reference_images(event, [first, "ca_missing"])

    async def test_generated_results_use_request_order_not_completion_order(self):
        a1, a2, b = self.event(mid="a1"), self.event(mid="a2"), self.event("b", mid="b")
        r1 = self.plugin.capture_image_request(a1, "c1")
        r2 = self.plugin.capture_image_request(a2, "c1")
        rb = self.plugin.capture_image_request(b, "c1")
        id2 = await self.plugin.register_generated_image(
            r2, task_id="newer", data=png("blue")
        )
        idb = await self.plugin.register_generated_image(
            rb, task_id="other-user", data=png("green")
        )
        id1 = await self.plugin.register_generated_image(
            r1, task_id="older-slow", data=png()
        )
        catalog = self.plugin.get_image_catalog(self.event(mid="follow-up"))
        latest = [row["id"] for row in catalog if row["is_requester_latest_result"]]
        self.assertEqual(latest, [id2])
        self.assertEqual({row["id"] for row in catalog}, {id1, id2, idb})
        copied = await self.plugin.resolve_reference_images(
            self.event(mid="edit"), [id2]
        )
        self.assertEqual(copied[0]["data"], png("blue"))

    async def test_result_parent_ids_and_reset_receipt(self):
        parent = await self.add_source()
        receipt = self.plugin.capture_image_request(self.event(), "c1")
        image_id = await self.plugin.register_generated_image(
            receipt, task_id="edit", data=png(), parent_ids=[parent]
        )
        row = next(
            row
            for row in self.plugin.get_image_catalog(self.event(mid="next"))
            if row["id"] == image_id
        )
        self.assertEqual(row["parent_ids"], [parent])
        self.plugin._image_index.clear("g")
        self.assertIsNone(
            await self.plugin.register_generated_image(
                receipt, task_id="stale", data=png()
            )
        )

    async def test_conversation_switch_does_not_mark_old_result_as_current(self):
        receipt = self.plugin.capture_image_request(self.event(), "c1")
        await self.plugin.register_generated_image(
            receipt, task_id="old-cid", data=png()
        )
        rows = self.plugin.get_image_catalog(self.event(cid="c2"))
        self.assertFalse(any(row["is_requester_latest_result"] for row in rows))

    async def test_text_only_model_receives_edit_catalog_without_vision(self):
        await self.add_source()
        event = self.event()
        self.plugin._recall_supports_vision = lambda e: False
        request = SimpleNamespace(
            extra_user_content_parts=[],
            system_prompt="",
            image_urls=[],
            func_tool=SimpleNamespace(
                get_tool=lambda name: (
                    SimpleNamespace(
                        parameters={"properties": {"reference_image_ids": {}}}
                    )
                    if name == "aiimg_generate"
                    else None
                )
            ),
        )
        current = self.mod.MessageRecord(
            msg_id="q",
            sender_id="a",
            sender_name="a",
            content="把刚才的图改成水彩",
            timestamp=time.time(),
        )
        await self.plugin._prepare_image_recall(
            event, request, current, self.mod.TRIGGER_AT
        )
        self.assertTrue(event.get_extra(SNAPSHOT_KEY))
        self.assertIn(
            "reference_image_ids",
            str([p.text for p in request.extra_user_content_parts]),
        )

    async def test_sender_burst_does_not_remove_other_members_only_image(self):
        self.plugin._image_index.per_session = 3
        preserved = await self.add_source("b", mid="b")
        for n in range(6):
            event = self.event("a", mid=str(n))
            await self.plugin._record_recall_images(event, message(str(n), "a"))
        self.assertIsNotNone(self.plugin._image_index.get("g", preserved))

    async def test_result_identity_is_immutable_and_registration_idempotent(self):
        receipt = self.plugin.capture_image_request(self.event(), "c1")
        first = await self.plugin.register_generated_image(
            receipt, task_id="same", data=png()
        )
        self.assertEqual(
            first,
            await self.plugin.register_generated_image(
                receipt, task_id="same", data=png()
            ),
        )
        self.assertIsNone(
            await self.plugin.register_generated_image(
                receipt, task_id="same", data=png("blue")
            )
        )
        self.assertEqual(self.plugin._image_index.get("g", first).data, png())

    async def test_parent_must_be_in_receipt_but_can_expire_after_task_handoff(self):
        parent = await self.add_source()
        receipt = self.plugin.capture_image_request(self.event(), "c1")
        self.assertIsNone(
            await self.plugin.register_generated_image(
                receipt, task_id="wrong", data=png(), parent_ids=["ca_other_group"]
            )
        )
        self.plugin._image_index._remove(parent)
        result = await self.plugin.register_generated_image(
            receipt, task_id="valid", data=png(), parent_ids=[parent]
        )
        self.assertIsNotNone(result)

    async def test_invalid_generated_image_does_not_raise(self):
        receipt = self.plugin.capture_image_request(self.event(), "c1")
        self.assertIsNone(
            await self.plugin.register_generated_image(
                receipt, task_id="bad", data=b"not an image"
            )
        )

    async def test_catalog_is_immutable_after_first_read(self):
        event = self.event()
        self.assertEqual(self.plugin.get_image_catalog(event), [])
        await self.plugin._record_recall_images(event, message("q"))
        self.assertEqual(self.plugin.get_image_catalog(event), [])
        self.assertTrue(self.plugin.get_image_catalog(self.event(mid="next")))
