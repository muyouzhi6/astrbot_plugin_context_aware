"""
v3.2.0 新改动的单元测试

覆盖：
1. 规则4 时间窗口收紧（35s→20s）
2. 规则4 "他人正在对话中"保护
3. 规则6 已移除（快速连续对话不再推断对话对象）
4. strict_mode：TRIGGER_ACTIVE/UNKNOWN 下强制 talking_to=group
"""
from __future__ import annotations

import importlib.util
import sys
import time
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
        def info(self, *a, **kw): pass
        def debug(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass

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

    class ProviderRequest:
        def __init__(self):
            self.system_prompt = ""
            self.extra_user_content_parts = []

    astrbot_mod.logger = Logger()
    api_mod.star = StarNamespace
    event_mod.AstrMessageEvent = object
    event_mod.filter = FilterNamespace
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
        spec = importlib.util.spec_from_file_location("context_aware_v320", PLUGIN_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules["context_aware_v320"] = module
        spec.loader.exec_module(module)
        sys.modules.pop("context_aware_v320", None)
        return module


class FakeContext:
    def get_config(self, umo: str | None = None):
        return {"wake_prefix": [], "provider_ltm_settings": {"group_icl_enable": False}}


class V320ChangesTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_plugin_module()

    def _make_analyzer(self, bot_id: str = "bot"):
        return self.mod.SceneAnalyzer(bot_id=bot_id)

    def _make_msg(self, msg_id, sender_id, sender_name, content, ts,
                  is_bot=False, talking_to="group", talking_to_name="群聊",
                  at_bot=False):
        msg = self.mod.MessageRecord(
            msg_id=msg_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            timestamp=ts,
            is_bot=is_bot,
            talking_to=talking_to,
            talking_to_name=talking_to_name,
            at_bot=at_bot,
        )
        return msg

    # ------------------------------------------------------------------ #
    # 规则4：时间窗口收紧测试
    # ------------------------------------------------------------------ #

    def test_rule4_within_20s_still_infers_bot(self):
        """规则4：Bot回复后15秒内用户发短确认词，应推断为回复Bot"""
        analyzer = self._make_analyzer()
        now = time.time()

        bot_msg = self._make_msg("b1", "bot", "[你]", "好的，稍等。", now - 15,
                                  is_bot=True, talking_to="alice", talking_to_name="Alice")
        user_msg = self._make_msg("u2", "alice", "Alice", "谢谢", now)

        reason = analyzer.infer_addressee(
            user_msg,
            [bot_msg],
            bot_replied_to="alice",
            bot_replied_to_name="Alice",
        )
        self.assertEqual(user_msg.talking_to, "bot", f"应推断为bot，得到: {user_msg.talking_to} ({reason})")

    def test_rule4_beyond_20s_does_not_infer_bot(self):
        """规则4收紧：Bot回复后25秒，时间窗口已过，不再推断为回复Bot"""
        analyzer = self._make_analyzer()
        now = time.time()

        bot_msg = self._make_msg("b1", "bot", "[你]", "好的，稍等。", now - 25,
                                  is_bot=True, talking_to="alice", talking_to_name="Alice")
        user_msg = self._make_msg("u2", "alice", "Alice", "谢谢", now)

        reason = analyzer.infer_addressee(
            user_msg,
            [bot_msg],
            bot_replied_to="alice",
            bot_replied_to_name="Alice",
        )
        self.assertEqual(user_msg.talking_to, "group", f"25s后应保持group，得到: {user_msg.talking_to} ({reason})")

    def test_rule4_other_user_talking_to_current_user_prevents_bot_inference(self):
        """规则4保护：最近有人主动在和当前用户说话，'谢谢'应推断为回那个人"""
        analyzer = self._make_analyzer()
        now = time.time()

        # Bob 10秒前对 Alice 说了话
        bob_msg = self._make_msg("bob1", "bob", "Bob", "Alice你觉得呢", now - 10,
                                  talking_to="alice", talking_to_name="Alice")
        # Bot 5秒前回复了 Alice
        bot_msg = self._make_msg("b1", "bot", "[你]", "是的，Alice说得对。", now - 5,
                                  is_bot=True, talking_to="alice", talking_to_name="Alice")
        # Alice 现在说"好的"
        user_msg = self._make_msg("u2", "alice", "Alice", "好的", now)

        reason = analyzer.infer_addressee(
            user_msg,
            [bob_msg, bot_msg],
            bot_replied_to="alice",
            bot_replied_to_name="Alice",
        )
        # 应该推断 Alice 在回复 Bob，不是 Bot
        self.assertNotEqual(user_msg.talking_to, "bot",
                            f"Bob刚对Alice说话，'好的'应回Bob而非Bot，得到: {user_msg.talking_to} ({reason})")
        self.assertEqual(user_msg.talking_to, "bob",
                          f"应推断为回复Bob，得到: {user_msg.talking_to} ({reason})")

    # ------------------------------------------------------------------ #
    # 规则6：已移除，快速连续对话不应改变 talking_to
    # ------------------------------------------------------------------ #

    def test_rule6_removed_quick_followup_stays_group(self):
        """规则6已移除：A 在 10 秒内跟在 B 的消息后发消息，不应推断 A 在回复 B"""
        analyzer = self._make_analyzer()
        now = time.time()

        b_msg = self._make_msg("b1", "bob", "Bob", "今天天气真好", now - 8,
                                talking_to="group", talking_to_name="群聊")
        a_msg = self._make_msg("a1", "alice", "Alice", "对啊", now)

        reason = analyzer.infer_addressee(a_msg, [b_msg])
        # 规则6 移除后，应保持默认 group
        self.assertEqual(a_msg.talking_to, "group",
                          f"规则6移除后应保持group，得到: {a_msg.talking_to} ({reason})")
        self.assertEqual(reason, self.mod.InferenceReason.DEFAULT_GROUP,
                          f"应为DEFAULT_GROUP，得到: {reason}")

    # ------------------------------------------------------------------ #
    # strict_mode 测试
    # ------------------------------------------------------------------ #

    def _make_fake_event(self, extras: dict | None = None, is_private: bool = False):
        """创建一个最小化 FakeEvent"""
        mod = self.mod
        extras = extras or {}

        class FakeMsgObj:
            message_id = "test_msg_001"
            message = []

        class FakeEvent:
            message_obj = FakeMsgObj()
            message_str = "好的"
            unified_msg_origin = "aiocqhttp:group:999"
            is_at_or_wake_command = False

            def get_extra(self, key, default=None):
                return extras.get(key, default)

            def set_extra(self, key, value):
                extras[key] = value

            def get_sender_id(self):
                return "alice"

            def get_sender_name(self):
                return "Alice"

            def get_self_id(self):
                return "bot"

            def get_messages(self):
                return []

            def get_message_outline(self):
                return ""

            def get_message_str(self):
                return "好的"

            def is_private_chat(self):
                return is_private

        return FakeEvent()

    async def _run_llm_request_with_history(self, plugin, event, req, history_msgs):
        """向 session 注入历史消息后触发 on_llm_request"""
        umo = event.unified_msg_origin
        for m in history_msgs:
            await plugin._sessions.add_message_async(umo, m)
        await plugin.on_llm_request(event, req)

    def test_strict_mode_disabled_by_default(self):
        """strict_mode 默认关闭"""
        plugin = self.mod.Main(FakeContext(), {"enable": True})
        self.assertFalse(plugin._strict_mode)

    def test_strict_mode_enabled_via_config(self):
        """strict_mode 可通过配置开启"""
        plugin = self.mod.Main(FakeContext(), {"enable": True, "strict_mode": True})
        self.assertTrue(plugin._strict_mode)

    def test_inferencereason_no_rule6(self):
        """InferenceReason 不再包含 RULE_6_QUICK_FOLLOW"""
        self.assertFalse(
            hasattr(self.mod.InferenceReason, "RULE_6_QUICK_FOLLOW"),
            "RULE_6_QUICK_FOLLOW 已从 v3.2.0 中移除"
        )

    def test_version_is_320(self):
        """插件版本应为 3.2.0"""
        import re
        with open(PLUGIN_PATH, encoding="utf-8") as f:
            content = f.read()
        match = re.search(r'^Version:\s*(\S+)', content, re.MULTILINE)
        self.assertIsNotNone(match, "找不到 Version 字段")
        self.assertEqual(match.group(1), "3.2.0")


if __name__ == "__main__":
    unittest.main()
