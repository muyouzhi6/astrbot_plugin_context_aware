import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from private_topic import Settings, Store, TopicMemory, tokens, validate_result


def response(prompt, text="预算 3000，不自驾"):
    data = json.loads(prompt)
    return SimpleNamespace(
        role="assistant",
        completion_text=json.dumps(
            {
                "status": "ok",
                "updates": [
                    {
                        "kind": "constraint",
                        "text": text,
                        "sources": [data["messages"][0]["id"]],
                        "replaces": [],
                    }
                ],
            }
        ),
    )


class TopicTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = replace(Settings(), enabled=True, update_after_turns=1)
        self.calls = []

        async def generate(provider, prompt, instruction):
            self.calls.append(provider)
            return response(prompt)

        self.generator = generate
        self.memory = self.make()
        await self.memory.start()

    def make(self, cfg=None, generate=None):
        memory = TopicMemory(
            Path(self.tmp.name) / "topic.db",
            cfg or self.cfg,
            AsyncMock(return_value=["primary", "secondary"]),
            generate or self.generator,
            lambda *args: None,
        )
        memory.closed = True  # Tests drive durable jobs explicitly, no clock races.
        return memory

    async def asyncTearDown(self):
        await self.memory.close()
        self.tmp.cleanup()

    async def turn(self, number=1, cid="a", text="预算 3000，不自驾"):
        receipt = await self.memory.begin("private:u", cid, str(number), text)
        await self.memory.finish(receipt, "可以考虑地铁。")
        return receipt

    async def update(self):
        prepared = await self.memory.store.run(self.memory.store.prepare)
        self.assertIsNotNone(prepared)
        await self.memory.update(*prepared)
        return prepared

    async def test_fallback_and_durable_checkpoint(self):
        async def generate(provider, prompt, instruction):
            self.calls.append(provider)
            if provider == "primary":
                raise ConnectionError("offline")
            return response(prompt)

        self.memory.generate = generate
        await self.turn()
        await self.update()
        restored = self.make()
        await restored.start()
        receipt = await restored.begin("private:u", "a", "next", "继续")
        text = await restored.render(receipt, [])
        self.assertIn("不自驾", text)
        self.assertEqual(self.calls, ["primary", "secondary"])
        self.assertLessEqual(tokens(text), self.cfg.injection_token_budget)

    async def test_refusal_does_not_overwrite_or_retry_same_batch(self):
        await self.turn()
        prepared = await self.update()
        await self.turn(2, text="another turn")
        await self.memory.store.run(
            lambda db: db.execute("UPDATE sessions SET next_try=0").rowcount
        )
        before = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.memory.generate = AsyncMock(
            return_value=SimpleNamespace(
                role="assistant", completion_text='{"status":"refused","updates":[]}'
            )
        )
        await self.update()
        after = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(before["checkpoint"], after["checkpoint"])
        self.assertEqual(before["cursor"], after["cursor"])
        self.assertEqual(after["error"], "refusal")
        self.assertEqual(self.memory.generate.await_count, 1)
        await self.memory.store.run(
            lambda db: db.execute("UPDATE sessions SET next_try=0").rowcount
        )
        self.assertIsNone(await self.memory.store.run(self.memory.store.prepare))
        await self.turn(3, text="new independent information")
        next_batch = await self.memory.store.run(self.memory.store.prepare)
        self.assertTrue(all(s["id"] > prepared[1][-1]["id"] for s in next_batch[1]))

    async def test_reset_discards_inflight_result_and_old_reply(self):
        receipt = await self.turn()
        prepared = await self.memory.store.run(self.memory.store.prepare)

        async def generate(provider, prompt, instruction):
            await self.memory.clear("private:u", "a")
            await self.memory.begin("private:u", "a", "fresh", "新话题")
            return response(prompt)

        self.memory.generate = generate
        await self.memory.update(*prepared)
        self.assertFalse(
            await self.memory.store.run(Store.finish, receipt, "late reply")
        )
        state = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(state["checkpoint"], "[]")

    async def test_switch_isolation_and_duplicate_delivery(self):
        receipt = await self.turn()
        await self.memory.finish(receipt, "duplicate")
        repeated = await self.memory.begin("private:u", "a", "1", "预算 3000，不自驾")
        self.assertEqual(receipt, repeated)
        count = await self.memory.store.run(
            lambda db: db.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        )
        self.assertEqual(count, 2)
        await self.update()
        other = await self.memory.begin("private:u", "b", "1", "other")
        self.assertEqual(await self.memory.render(other, []), "")
        back = await self.memory.begin("private:u", "a", "2", "继续")
        self.assertIn("不自驾", await self.memory.render(back, []))

    async def test_long_message_segment_coverage_and_no_new_message_loss(self):
        await self.turn(text="甲" * 9000)
        prepared = await self.memory.store.run(self.memory.store.prepare)
        selected = []

        async def generate(provider, prompt, instruction):
            selected.extend(x["id"] for x in json.loads(prompt)["messages"])
            await self.turn(2, text="预算改为 2000")
            return response(prompt)

        self.memory.generate = generate
        await self.memory.update(*prepared)
        state = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(state["cursor"], max(selected))
        pending = await self.memory.store.run(
            lambda db: [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM segments WHERE id>?", (state["cursor"],)
                )
            ]
        )
        self.assertTrue(any("2000" in s["text"] for s in pending))
        self.assertTrue(any("甲" in s["text"] for s in pending))

    async def test_invalid_evidence_then_good_fallback(self):
        await self.turn()

        async def generate(provider, prompt, instruction):
            self.calls.append(provider)
            if provider == "primary":
                return SimpleNamespace(
                    role="assistant",
                    completion_text='{"status":"ok","updates":[{"kind":"constraint","text":"invented","sources":[999999]}]}',
                )
            return response(prompt)

        self.memory.generate = generate
        await self.update()
        self.assertEqual(self.calls, ["primary", "secondary"])

    async def test_budget_counts_fallback_and_survives_restart(self):
        self.memory.store.settings = replace(self.cfg, daily_request_budget=1)
        self.memory.generate = AsyncMock(side_effect=ConnectionError())
        await self.turn()
        await self.update()
        self.assertEqual(self.memory.generate.await_count, 1)
        state = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(state["error"], "budget")
        restored = self.make(replace(self.cfg, daily_request_budget=1))
        await restored.start()
        self.assertFalse(await restored.store.run(restored.store.reserve, 1))

    async def test_timeout_fallback_and_current_request_does_not_wait(self):
        self.memory.settings = replace(
            self.cfg, attempt_timeout_sec=0.02, job_timeout_sec=0.1
        )

        async def generate(provider, prompt, instruction):
            if provider == "primary":
                await asyncio.Event().wait()
            return response(prompt)

        self.memory.generate = generate
        await self.turn()
        prepared = await self.memory.store.run(self.memory.store.prepare)
        task = asyncio.create_task(self.memory.update(*prepared))
        current = await asyncio.wait_for(
            self.memory.begin("private:u", "a", "next", "继续"), 0.5
        )
        await asyncio.wait_for(self.memory.render(current, []), 0.5)
        await asyncio.wait_for(task, 0.5)

    async def test_bridge_not_duplicate_existing_history(self):
        await self.turn(text="预算3000元")
        current = await self.memory.begin("private:u", "a", "next", "继续")
        text = await self.memory.render(
            current,
            [
                {"role": "user", "content": "预算3000元"},
                {"role": "assistant", "content": "可以考虑地铁。"},
            ],
        )
        self.assertEqual(text, "")
        text = await self.memory.render(current, [])
        self.assertIn("预算3000元", text)

    async def test_checkpoint_update_preserves_unmentioned_constraints(self):
        old = [
            {"id": "a", "kind": "constraint", "text": "不自驾", "sources": [1]},
            {"id": "b", "kind": "constraint", "text": "预算3000", "sources": [2]},
        ]
        result = validate_result(
            json.dumps(
                {
                    "status": "ok",
                    "updates": [
                        {
                            "kind": "correction",
                            "text": "预算2000",
                            "sources": [3],
                            "replaces": ["b"],
                        }
                    ],
                }
            ),
            old,
            [{"id": 3}],
        )
        self.assertTrue(any(x["text"] == "不自驾" for x in result))
        self.assertFalse(any(x["text"] == "预算3000" for x in result))

    async def test_threshold_not_every_turn_after_first_summary(self):
        await self.turn()
        await self.update()
        self.memory.store.settings = replace(self.cfg, update_after_turns=4)
        await self.turn(2)
        await self.memory.store.run(
            lambda db: db.execute("UPDATE sessions SET next_try=0").rowcount
        )
        self.assertIsNone(await self.memory.store.run(self.memory.store.prepare))

    async def test_retention_marks_unprocessed_gap(self):
        await self.turn()
        await self.memory.store.run(
            lambda db: db.execute("UPDATE turns SET created=0").rowcount
        )
        await self.memory.store.run(self.memory.store.maintain)
        state = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(state["gap"], 1)
        self.assertEqual(
            await self.memory.store.run(
                lambda db: db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            ),
            0,
        )

    async def test_whole_chain_deadline_does_not_commit(self):
        self.memory.settings = replace(
            self.cfg, attempt_timeout_sec=1, job_timeout_sec=0.02
        )

        async def generate(*args):
            await asyncio.Event().wait()

        self.memory.generate = generate
        await self.turn()
        await asyncio.wait_for(self.update(), 0.5)
        state = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(state["cursor"], 0)
        self.assertEqual(state["error"], "job_timeout")

    async def test_cancel_does_not_trigger_fallback_or_commit(self):
        entered = asyncio.Event()

        async def generate(*args):
            entered.set()
            await asyncio.Event().wait()

        self.memory.generate = AsyncMock(side_effect=generate)
        await self.turn()
        prepared = await self.memory.store.run(self.memory.store.prepare)
        task = asyncio.create_task(self.memory.update(*prepared))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.memory.generate.await_count, 1)
        state = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(state["cursor"], 0)
        self.assertEqual(state["failures"], 0)

    async def test_chinese_panel_settings_and_disabled_default(self):
        cfg = Settings.parse(
            {
                "provider_mode": "单独指定",
                "content_refusal_policy": "尝试备用模型",
                "injection_token_budget": -1,
            }
        )
        self.assertEqual(cfg.provider_mode, "custom")
        self.assertEqual(cfg.content_refusal_policy, "try_next")
        self.assertEqual(cfg.injection_token_budget, 300)
        self.assertFalse(cfg.enabled)

    async def test_older_request_does_not_receive_future_checkpoint(self):
        earlier = await self.memory.begin("private:u", "a", "older", "先问")
        await self.turn(2, text="稍后决定预算3000元")
        await self.update()
        self.assertEqual(await self.memory.render(earlier, []), "")

    async def test_refusal_can_fallback_when_explicitly_selected(self):
        self.memory.settings = replace(self.cfg, content_refusal_policy="try_next")

        async def generate(provider, prompt, instruction):
            self.calls.append(provider)
            if provider == "primary":
                return SimpleNamespace(role="err", completion_text="prompt_blocked")
            return response(prompt)

        self.memory.generate = generate
        await self.turn()
        await self.update()
        self.assertEqual(self.calls, ["primary", "secondary"])

    async def test_plain_text_refusal_stops_default_chain(self):
        await self.turn()
        self.memory.generate = AsyncMock(
            return_value=SimpleNamespace(
                role="assistant", completion_text="抱歉，我无法总结这段内容。"
            )
        )
        await self.update()
        self.assertEqual(self.memory.generate.await_count, 1)
        state = await self.memory.store.run(
            lambda db: dict(db.execute("SELECT * FROM sessions").fetchone())
        )
        self.assertEqual(state["error"], "refusal")
        self.assertEqual(state["cursor"], 0)


if __name__ == "__main__":
    unittest.main()
