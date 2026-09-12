"""Run with real AstrBot on PYTHONPATH. Isolated data, no network or QQ sends."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astrbot.api.message_components import Plain  # noqa: E402
from astrbot.core.agent.message import Message, dump_messages_with_checkpoints  # noqa: E402
from astrbot.core.agent.context.manager import ContextManager  # noqa: E402
from astrbot.core.agent.context.config import ContextConfig  # noqa: E402
from astrbot.core.platform.astr_message_event import AstrMessageEvent  # noqa: E402
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember  # noqa: E402
from astrbot.core.platform.message_type import MessageType  # noqa: E402
from astrbot.core.platform.platform_metadata import PlatformMetadata  # noqa: E402
from astrbot.core.provider.entities import LLMResponse, ProviderRequest  # noqa: E402

import main  # noqa: E402
from private_topic import Store, RECEIPT  # noqa: E402


def event(n, text):
    raw = AstrBotMessage()
    raw.type = MessageType.FRIEND_MESSAGE
    raw.self_id = "bot"
    raw.sender = MessageMember("isolated", "Test")
    raw.message_id = str(n)
    raw.group_id = ""
    raw.message = [Plain(text)]
    raw.message_str = text
    raw.raw_message = {}
    return AstrMessageEvent(
        text, raw, PlatformMetadata("aiocqhttp", "test", "test"), "isolated"
    )


async def verify():
    with tempfile.TemporaryDirectory() as directory:
        main.get_astrbot_plugin_data_path = lambda: directory
        context = SimpleNamespace(get_config=lambda **kw: {"wake_prefix": []})
        plugin = main.Main(
            context, {"image_cache_dir": directory, "private_topic": {"enabled": True}}
        )
        plugin._topic.closed = True
        plugin._topic.candidates = lambda umo: asyncio.sleep(0, result=["fixture"])

        async def generate(provider, prompt, instruction):
            source = json.loads(prompt)["messages"][0]["id"]
            return LLMResponse(
                role="assistant",
                completion_text=json.dumps(
                    {
                        "status": "ok",
                        "updates": [
                            {
                                "kind": "constraint",
                                "text": "本次旅行不自驾，预算3000元",
                                "sources": [source],
                                "replaces": [],
                            }
                        ],
                    }
                ),
            )

        plugin._topic.generate = generate
        history = []
        manager = ContextManager(ContextConfig(enforce_max_turns=12))
        try:
            for n in range(30):
                text = "旅行预算3000元，不自驾" if n == 0 else f"继续讨论第{n}个细节"
                ev = event(n, text)
                req = ProviderRequest(prompt=text, contexts=history)
                req.conversation = SimpleNamespace(
                    cid="test-a", history=json.dumps(history)
                )
                await plugin.on_llm_request(ev, req)
                first_parts = len(req.extra_user_content_parts)
                await plugin.on_llm_request(ev, req)
                assert len(req.extra_user_content_parts) == first_parts, (
                    "duplicate topic injection"
                )
                current = Message.model_validate(await req.assemble_context())
                messages = [Message.model_validate(m) for m in history] + [current]
                messages = await manager.process(messages)
                messages.append(Message(role="assistant", content=f"建议{n}"))
                history = dump_messages_with_checkpoints(messages)
                assert "context_aware_private_topic" not in json.dumps(history)
                await plugin.on_llm_response(
                    ev, LLMResponse(role="assistant", completion_text=f"建议{n}")
                )
                if n == 3:
                    prepared = await plugin._topic.store.run(
                        plugin._topic.store.prepare
                    )
                    assert prepared
                    await plugin._topic.update(*prepared)
                if n == 29:
                    assert any("不自驾" in p.text for p in req.extra_user_content_parts)
                    context.conversation_manager = SimpleNamespace(
                        get_curr_conversation_id=AsyncMock(return_value="test-a"),
                        get_conversation=AsyncMock(
                            return_value=SimpleNamespace(history=json.dumps(history))
                        ),
                    )
                    expected = await plugin._topic.last_injection(
                        ev.unified_msg_origin, "test-a"
                    )
                    inspect_event = event(30, "短期记忆")
                    results = [
                        r async for r in plugin.show_short_term_memory(inspect_event)
                    ]
                    actual = "".join(
                        part.text
                        for r in results
                        for part in r.chain
                        if isinstance(part, Plain)
                    )
                    assert expected["text"] in actual
                    assert inspect_event.is_stopped()
            ev = event(31, "换个会话")
            req = ProviderRequest(prompt=ev.message_str)
            req.conversation = SimpleNamespace(cid="test-b", history="[]")
            await plugin.on_llm_request(ev, req)
            assert not req.extra_user_content_parts
            # Dashboard clears same CID, without command marker.
            ev = event(32, "重新开始")
            req = ProviderRequest(prompt=ev.message_str)
            req.conversation = SimpleNamespace(cid="test-a", history="[]")
            await plugin.on_llm_request(ev, req)
            view = await plugin._topic.store.run(Store.view, ev.get_extra(RECEIPT))
            assert view[0]["checkpoint"] == "[]"
            print(
                "PASS: 30 turns, Core 12-turn truncation, checkpoint retained, temporary injection never persisted, duplicate hook, conversation isolation, dashboard reset, real command result"
            )
        finally:
            await plugin.terminate()


if __name__ == "__main__":
    asyncio.run(verify())
