from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _decorator(*args: Any, **kwargs: Any):
    def wrap(func):
        return func

    return wrap


def install_astrbot_stubs() -> dict[str, types.ModuleType]:
    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    event_mod = types.ModuleType("astrbot.api.event")
    provider_mod = types.ModuleType("astrbot.api.provider")
    message_components_mod = types.ModuleType("astrbot.api.message_components")
    core_mod = types.ModuleType("astrbot.core")
    agent_mod = types.ModuleType("astrbot.core.agent")
    agent_message_mod = types.ModuleType("astrbot.core.agent.message")

    class Logger:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    class Star:
        def __init__(self, context):
            self.context = context

    StarNamespace = types.SimpleNamespace(Star=Star, Context=object)

    class PlatformAdapterType:
        ALL = object()

    FilterNamespace = types.SimpleNamespace(
        PlatformAdapterType=PlatformAdapterType,
        platform_adapter_type=_decorator,
        on_llm_request=_decorator,
        on_llm_response=_decorator,
        after_message_sent=_decorator,
    )

    class TextPart:
        def __init__(self, text: str):
            self.text = text

        def mark_as_temp(self):
            return self

    class Plain:
        def __init__(self, text: str):
            self.text = text

    class Image:
        def __init__(self, url: str = "", file: str = ""):
            self.url = url
            self.file = file

    class At:
        def __init__(self, qq: str, name: str = ""):
            self.qq = qq
            self.name = name

    class AtAll:
        pass

    class Reply:
        def __init__(self, sender_id: str = ""):
            self.sender_id = sender_id

    astrbot_mod.logger = Logger()
    api_mod.star = StarNamespace
    event_mod.AstrMessageEvent = object
    event_mod.filter = FilterNamespace
    class ProviderRequest:
        def __init__(self):
            self.system_prompt = ""
            self.extra_user_content_parts = []

    provider_mod.LLMResponse = object
    provider_mod.Provider = object
    provider_mod.ProviderRequest = ProviderRequest
    agent_message_mod.TextPart = TextPart
    message_components_mod.Plain = Plain
    message_components_mod.Image = Image
    message_components_mod.At = At
    message_components_mod.AtAll = AtAll
    message_components_mod.Reply = Reply

    return {
        "astrbot": astrbot_mod,
        "astrbot.api": api_mod,
        "astrbot.api.event": event_mod,
        "astrbot.api.provider": provider_mod,
        "astrbot.api.message_components": message_components_mod,
        "astrbot.core": core_mod,
        "astrbot.core.agent": agent_mod,
        "astrbot.core.agent.message": agent_message_mod,
    }


def load_plugin_module():
    with patch.dict(sys.modules, install_astrbot_stubs()):
        spec = importlib.util.spec_from_file_location("context_aware_main", PLUGIN_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules["context_aware_main"] = module
        spec.loader.exec_module(module)
        sys.modules.pop("context_aware_main", None)
        return module


class FakeContext:
    def get_config(self, umo: str | None = None):
        return {"wake_prefix": [], "provider_ltm_settings": {"group_icl_enable": False}}


class FakeMessageObj:
    message_id = "m1"
    message = []


class FakeEvent:
    def __init__(self):
        self.message_obj = FakeMessageObj()
        self.message_str = ""
        self.unified_msg_origin = "aiocqhttp:group:200"
        self.extras = {
            "_gemini_stt_transcript": "大家等会儿看一下这个配置",
            "_gemini_stt_cache_only": True,
        }

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_sender_id(self):
        return "100"

    def get_sender_name(self):
        return "Alice"

    def get_self_id(self):
        return "bot"

    def get_messages(self):
        return []

    def get_message_outline(self):
        return ""

    def get_message_str(self):
        return ""

    def is_private_chat(self):
        return False


class ContextAwareGeminiSTTTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mod = load_plugin_module()

    async def test_gemini_stt_transcript_is_recorded_as_message(self):
        plugin = self.mod.Main(FakeContext(), {"enable": True, "only_group_chat": True})
        event = FakeEvent()

        await plugin.on_message(event)

        messages = plugin.get_recent_messages(event.unified_msg_origin, count=1)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "[语音转写] 大家等会儿看一下这个配置")

    async def test_voice_transcript_counts_as_content(self):
        plugin = self.mod.Main(FakeContext(), {"enable": True, "only_group_chat": True})
        event = FakeEvent()

        await plugin.on_message(event)

        self.assertTrue(plugin._sessions.has_session(event.unified_msg_origin))

    async def test_llm_request_fallback_and_message_handler_do_not_duplicate_voice(self):
        plugin = self.mod.Main(FakeContext(), {"enable": True, "only_group_chat": True})
        event = FakeEvent()
        req = self.mod.ProviderRequest()

        await plugin.on_llm_request(event, req)
        await plugin.on_message(event)

        messages = plugin.get_recent_messages(event.unified_msg_origin, count=5)
        voice_messages = [
            msg
            for msg in messages
            if msg["content"] == "[语音转写] 大家等会儿看一下这个配置"
        ]
        self.assertEqual(len(voice_messages), 1)
        self.assertEqual(plugin._stats.messages_recorded, 1)

    def test_scene_generator_injects_recent_voice_transcripts_outside_dialogue_flow(self):
        generator = self.mod.SceneGenerator()
        current = self.mod.MessageRecord(
            msg_id="current",
            sender_id="100",
            sender_name="Alice",
            content="@bot 更早的一条呢",
            timestamp=100.0,
            at_bot=True,
            talking_to="bot",
        )
        voice = self.mod.MessageRecord(
            msg_id="voice",
            sender_id="100",
            sender_name="Alice",
            content="[语音转写] 我想测一下唱歌识别",
            timestamp=1.0,
        )
        recent = self.mod.MessageRecord(
            msg_id="recent",
            sender_id="101",
            sender_name="Bob",
            content="普通聊天",
            timestamp=99.0,
        )

        scene = generator.generate(
            trigger_type=self.mod.TRIGGER_AT,
            trigger_desc="被@",
            current=current,
            flow=[recent, current],
            voice_flow=[voice, recent, current],
            bot_status={},
            participants=["Alice", "Bob"],
        )

        self.assertIn("<recent_voice_transcripts>", scene)
        self.assertIn("[语音转写] 我想测一下唱歌识别", scene)

    async def test_on_llm_request_keeps_voice_transcript_from_larger_window(self):
        plugin = self.mod.Main(
            FakeContext(),
            {
                "enable": True,
                "only_group_chat": True,
                "dialogue_window": 3,
                "voice_context_window": 20,
            },
        )
        event = FakeEvent()
        req = self.mod.ProviderRequest()

        records = [
            self.mod.MessageRecord(
                msg_id="voice",
                sender_id="100",
                sender_name="Alice",
                content="[语音转写] 我想测一下唱歌识别",
                timestamp=1.0,
            )
        ]
        for i in range(10):
            records.append(
                self.mod.MessageRecord(
                    msg_id=f"filler-{i}",
                    sender_id=str(200 + i),
                    sender_name=f"User{i}",
                    content=f"普通聊天 {i}",
                    timestamp=2.0 + i,
                )
            )
        records.append(
            self.mod.MessageRecord(
                msg_id="current",
                sender_id="100",
                sender_name="Alice",
                content="@bot 更早的一条呢",
                timestamp=20.0,
                at_bot=True,
                talking_to="bot",
            )
        )
        for record in records:
            await plugin._sessions.add_message_async(event.unified_msg_origin, record)

        await plugin.on_llm_request(event, req)

        self.assertTrue(req.extra_user_content_parts)
        scene = req.extra_user_content_parts[-1].text
        self.assertIn("<recent_voice_transcripts>", scene)
        self.assertIn("[语音转写] 我想测一下唱歌识别", scene)


if __name__ == "__main__":
    unittest.main()
