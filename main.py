"""
AstrBot 上下文场景感知增强插件 v2.4.0 (Context-Aware Enhancement)

为 LLM 提供结构化的群聊场景描述，增强其对对话情境的理解能力。
重点解决：主动回复时 Bot 误以为别人在问自己的问题。

核心功能:
- 触发类型检测: 被@、被回复、唤醒词、主动搭话
- 对话对象推断: 谁在和谁说话（关键功能）
- 对话流分析: 最近的对话结构
- Bot 状态追踪: 上次发言时间和内容
- 图像转述: 将群友发送的图片转为文字描述（可选）

设计原则:
- 只做加法，不修改框架原有信息
- 可完全替代框架内置 LTM 的群聊记录功能
- 轻量高效，图像转述为可选功能

v2.4.0 更新:
- 新增图像转述功能：可将群友发送的图片转为文字描述
- 支持自定义图像转述提供商和提示词
- 完善文档：明确说明需关闭框架内置 LTM 以避免重复

v2.3.0 更新:
- 修复回复词推断误判：现在只有当 Bot 之前确实在回复当前用户时，才会推断用户在回复 Bot
- 增加 Bot 回复对象追踪：记录 Bot 每次回复的目标用户
- 优化 TRIGGER_UNKNOWN 处理：未知触发时采用更保守的策略
- 收紧上下文推断条件：减少误判风险

Author: 木有知
Version: 2.4.0
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from astrbot import logger
from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart

if TYPE_CHECKING:
    from astrbot.core.config import AstrBotConfig


# ============================================================================
# Constants
# ============================================================================

# 触发类型常量
TRIGGER_PRIVATE: Final = "private_chat"
TRIGGER_AT: Final = "at_bot"
TRIGGER_REPLY: Final = "reply_to_bot"
TRIGGER_WAKE: Final = "wake_word"
TRIGGER_MENTION: Final = "mention"
TRIGGER_ACTIVE: Final = "active"
TRIGGER_UNKNOWN: Final = "unknown"

# 触发类型中文名（用于日志）
TRIGGER_NAMES: Final = {
    TRIGGER_PRIVATE: "私聊",
    TRIGGER_AT: "@Bot",
    TRIGGER_REPLY: "回复Bot",
    TRIGGER_WAKE: "唤醒词",
    TRIGGER_MENTION: "提及Bot",
    TRIGGER_ACTIVE: "主动触发",
    TRIGGER_UNKNOWN: "未知",
}

# 回复特征词（用于判断是否在回复 Bot）
REPLY_STARTERS: Final = frozenset({
    "好的", "好", "嗯", "是的", "对", "谢谢", "感谢", "收到",
    "明白", "知道了", "了解", "可以", "行", "没问题",
    "ok", "OK", "Ok", "好滴", "好哒", "好嘞", "okok",
})


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(slots=True)
class MessageRecord:
    """轻量级消息记录"""

    msg_id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: float  # Unix timestamp
    is_bot: bool = False
    at_bot: bool = False
    reply_to_id: str | None = None
    talking_to: str = "group"
    talking_to_name: str = "群聊"
    at_targets: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class SessionState:
    """会话状态 - 每个群聊/私聊一个"""

    messages: list[MessageRecord] = field(default_factory=list)
    bot_last_spoke_at: float = 0.0
    bot_last_content: str = ""
    bot_last_replied_to: str = ""  # Bot 上次回复的对象 ID
    bot_last_replied_to_name: str = ""  # Bot 上次回复的对象名称


@dataclass
class PluginStats:
    """插件统计信息"""

    messages_recorded: int = 0
    scenes_injected: int = 0
    bot_responses_recorded: int = 0
    trigger_counts: dict[str, int] = field(default_factory=dict)

    def record_trigger(self, trigger_type: str) -> None:
        self.trigger_counts[trigger_type] = self.trigger_counts.get(trigger_type, 0) + 1


# ============================================================================
# Session Manager (LRU Cache)
# ============================================================================


class SessionManager:
    """会话管理器 - 带 LRU 淘汰机制"""

    __slots__ = ("_sessions", "_max_messages", "_max_sessions")

    def __init__(self, max_messages: int = 50, max_sessions: int = 100) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._max_messages = max(10, max_messages)
        self._max_sessions = max(10, max_sessions)

    def get(self, session_id: str) -> SessionState:
        """获取或创建会话状态"""
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        while len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)

        state = SessionState()
        self._sessions[session_id] = state
        return state

    def add_message(self, session_id: str, msg: MessageRecord) -> None:
        """添加消息到会话"""
        state = self.get(session_id)
        state.messages.append(msg)
        if len(state.messages) > self._max_messages:
            state.messages = state.messages[-self._max_messages:]

    def record_bot_response(
        self,
        session_id: str,
        content: str,
        ts: float,
        replied_to_id: str = "",
        replied_to_name: str = "",
    ) -> None:
        """记录 Bot 回复"""
        state = self.get(session_id)
        state.bot_last_spoke_at = ts
        state.bot_last_content = content[:100] if content else ""
        state.bot_last_replied_to = replied_to_id
        state.bot_last_replied_to_name = replied_to_name

    def has_session(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return session_id in self._sessions

    def get_session_count(self) -> int:
        """获取当前会话数量"""
        return len(self._sessions)

    def get_message_count(self, session_id: str) -> int:
        """获取会话消息数量"""
        if session_id in self._sessions:
            return len(self._sessions[session_id].messages)
        return 0


# ============================================================================
# Scene Analyzer
# ============================================================================


class SceneAnalyzer:
    """场景分析器 - 负责所有分析逻辑"""

    __slots__ = ("_bot_id", "_bot_names")

    def __init__(self, bot_id: str, bot_names: list[str] | None = None) -> None:
        self._bot_id = bot_id
        self._bot_names: tuple[str, ...] = tuple(
            n.lower() for n in (bot_names or []) if n
        )

    def extract_message(self, event: AstrMessageEvent) -> MessageRecord:
        """从事件提取消息记录"""
        sender_id = event.get_sender_id()

        # 提取消息内容，拼接所有文本和图片描述
        content = event.message_str or ""
        if not content:
            # message_str 为空时，从消息组件中拼接
            parts: list[str] = []
            for comp in event.get_messages():
                if isinstance(comp, Plain) and comp.text:
                    parts.append(comp.text)
                elif isinstance(comp, Image):
                    parts.append("[图片]")
            content = "".join(parts) if parts else "[消息]"

        msg = MessageRecord(
            msg_id=str(event.message_obj.message_id),
            sender_id=sender_id,
            sender_name=event.get_sender_name() or sender_id,
            content=content[:500],
            timestamp=time.time(),
            is_bot=(sender_id == self._bot_id),
        )

        for comp in event.get_messages():
            if isinstance(comp, At):
                qq_str = str(comp.qq)
                msg.at_targets.append((qq_str, comp.name or qq_str))
                if qq_str == self._bot_id:
                    msg.at_bot = True
            elif isinstance(comp, Reply):
                if comp.sender_id:
                    msg.reply_to_id = str(comp.sender_id)

        return msg

    def detect_trigger(
        self, event: AstrMessageEvent, msg: MessageRecord
    ) -> tuple[str, str]:
        """检测触发类型"""
        sender = msg.sender_name

        if event.is_private_chat():
            return TRIGGER_PRIVATE, f"私聊对话，{sender} 在直接和你交流"

        if msg.at_bot:
            return TRIGGER_AT, f"{sender} @了你，需要你回应"

        if msg.reply_to_id == self._bot_id:
            return TRIGGER_REPLY, f"{sender} 回复了你之前的消息"

        if event.is_at_or_wake_command and not msg.at_bot:
            return TRIGGER_WAKE, f"{sender} 使用唤醒词呼叫你"

        if self._bot_names:
            msg_lower = msg.content.lower()
            for name in self._bot_names:
                if name in msg_lower:
                    return TRIGGER_MENTION, f"{sender} 在消息中提到了你"

        if event.get_extra("_active_trigger") or event.get_extra("active_reply_triggered"):
            return TRIGGER_ACTIVE, "你是主动加入这个对话的，没有人在叫你"

        return TRIGGER_UNKNOWN, "触发原因未知"

    def infer_addressee(
        self,
        msg: MessageRecord,
        history: list[MessageRecord],
        bot_replied_to: str = "",
        bot_replied_to_name: str = "",
    ) -> None:
        """
        推断消息的对话对象
        
        核心原则：宁可保守（判定为群聊），不可激进（误判为和Bot说话）
        只有高置信度时才判定 talking_to = "bot"
        """
        # ===== 规则1: 明确的 @ Bot（高置信度）=====
        if msg.at_bot:
            msg.talking_to, msg.talking_to_name = "bot", "你"
            return

        # ===== 规则2: @ 其他人（高置信度）=====
        if msg.at_targets:
            target_id, target_name = msg.at_targets[0]
            if target_id != self._bot_id:
                msg.talking_to, msg.talking_to_name = target_id, target_name
                return

        # ===== 规则3: 引用回复消息（高置信度）=====
        if msg.reply_to_id:
            if msg.reply_to_id == self._bot_id:
                msg.talking_to, msg.talking_to_name = "bot", "你"
            else:
                msg.talking_to = msg.reply_to_id
                for m in reversed(history):
                    if m.sender_id == msg.reply_to_id:
                        msg.talking_to_name = m.sender_name
                        break
                else:
                    msg.talking_to_name = msg.reply_to_id
            return

        # ===== 以下是上下文推断，需要更保守 =====
        if not history:
            # 没有历史，保持默认 "group"
            return

        recent = [m for m in history[-5:] if m.sender_id != msg.sender_id]
        if not recent:
            return

        last = recent[-1]
        time_gap = msg.timestamp - last.timestamp

        # ===== 规则4: Bot 刚回复过当前用户，且用户像在回应（中置信度）=====
        # 关键修复：必须是 Bot 之前在回复"当前这个用户"，才能推断用户在回复 Bot
        if last.is_bot and time_gap < 45:
            # 检查 Bot 上次是否在回复当前发言者
            if bot_replied_to == msg.sender_id:
                if self._looks_like_reply(msg.content):
                    msg.talking_to, msg.talking_to_name = "bot", "你"
                    return
            # 如果 Bot 不是在回复这个人，则这个人的"谢谢"大概率不是对 Bot 说的
            # 保持 talking_to = "group"
            return

        # ===== 规则5: A-B-A 对话模式（低置信度，需要更多条件）=====
        # 只有当上一条消息明确是对当前用户说的，才推断当前用户在回复
        if last.talking_to == msg.sender_id and time_gap < 60:
            # 额外检查：上一条不是 Bot 发的（Bot 场景已在规则4处理）
            if not last.is_bot:
                msg.talking_to, msg.talking_to_name = last.sender_id, last.sender_name
                return

        # ===== 规则6: 快速连续对话（最低置信度，收紧条件）=====
        # 只有非常短的时间间隔 + 非 Bot 消息 + 上一条是对群说的，才推断是延续对话
        if time_gap < 15 and not last.is_bot:
            # 如果上一条是某人对群说的，当前消息可能是在回应那个人
            if last.talking_to == "group":
                msg.talking_to, msg.talking_to_name = last.sender_id, last.sender_name
                return
        
        # 默认：保持 talking_to = "group"，表示无法确定具体对话对象

    @staticmethod
    def _looks_like_reply(content: str) -> bool:
        """判断是否像回复"""
        stripped = content.strip()
        return any(stripped.startswith(s) for s in REPLY_STARTERS)


# ============================================================================
# Scene Generator - 核心：生成清晰的场景描述
# ============================================================================


class SceneGenerator:
    """场景描述生成器 - 生成清晰有力的场景描述"""

    __slots__ = ()

    @staticmethod
    def _escape(text: str) -> str:
        """XML 转义"""
        if not text:
            return ""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def generate(
        self,
        trigger_type: str,
        trigger_desc: str,
        current: MessageRecord,
        flow: list[MessageRecord],
        bot_status: dict[str, float | str | bool],
        participants: list[str],
        *,
        show_flow: bool = True,
    ) -> str:
        """生成场景描述，重点强调对话对象"""
        esc = self._escape
        parts: list[str] = ["<conversation_scene>"]

        # ===== 1. 触发类型 =====
        parts.append(f'  <trigger type="{trigger_type}">{esc(trigger_desc)}</trigger>')

        # ===== 2. 当前消息分析（最重要的部分）=====
        is_talking_to_bot = current.talking_to == "bot"
        is_talking_to_group = current.talking_to == "group"

        if is_talking_to_bot:
            addressee_desc = "你（Bot）"
        elif is_talking_to_group:
            addressee_desc = "群里所有人（非特定对象）"
        else:
            addressee_desc = current.talking_to_name

        parts.append(
            f'  <current_message>'
            f'\n    <sender>{esc(current.sender_name)}</sender>'
            f'\n    <talking_to>{esc(addressee_desc)}</talking_to>'
            f'\n    <content>{esc(current.content[:80])}</content>'
            f'\n  </current_message>'
        )

        # ===== 3. 关键行为指导（重点！）=====
        instruction = self._generate_instruction(
            trigger_type, current, is_talking_to_bot, is_talking_to_group
        )
        if instruction:
            parts.append(f'  <instruction>{instruction}</instruction>')

        # ===== 4. 对话流（简化）=====
        if show_flow and len(flow) > 1:
            flow_lines: list[str] = []
            for m in flow[-5:]:
                to_name = "你" if m.talking_to == "bot" else (
                    "群" if m.talking_to == "group" else m.talking_to_name
                )
                sender = "[你]" if m.is_bot else m.sender_name
                preview = m.content[:20] + ("..." if len(m.content) > 20 else "")
                flow_lines.append(f'    <m>{esc(sender)} → {esc(to_name)}: {esc(preview)}</m>')
            parts.append('  <recent_flow>')
            parts.extend(flow_lines)
            parts.append('  </recent_flow>')

        # ===== 5. Bot 状态 =====
        if bot_status.get("active"):
            mins = bot_status.get("minutes_ago", 0)
            if isinstance(mins, (int, float)) and mins > 0:
                parts.append(f'  <your_last_message minutes_ago="{mins:.1f}"/>')

        # ===== 6. 参与者 =====
        if len(participants) > 1:
            parts.append(f'  <participants>{esc(", ".join(participants[:5]))}</participants>')

        parts.append("</conversation_scene>")
        return "\n".join(parts)

    @staticmethod
    def _generate_instruction(
        trigger: str,
        msg: MessageRecord,
        is_talking_to_bot: bool,
        is_talking_to_group: bool,
    ) -> str:
        """
        生成行为指导 - 这是解决"误以为在问自己"问题的关键
        
        核心原则：
        - 明确触发（@、回复、唤醒词、私聊）→ 正常回应
        - 主动触发 → 必须明确告知 Bot 它是主动插入的
        - 未知触发 → 最保守处理
        """
        # ===== 被明确呼叫 - 正常回复 =====
        if trigger in (TRIGGER_AT, TRIGGER_REPLY, TRIGGER_WAKE, TRIGGER_PRIVATE):
            return "用户在和你对话，请正常回应。"

        if trigger == TRIGGER_MENTION:
            return "用户提到了你，可以适当回应。"

        # ===== 主动触发 - 需要特别小心 =====
        if trigger == TRIGGER_ACTIVE:
            if is_talking_to_bot:
                # 即使推断用户在和 Bot 说话，也要提醒这是主动触发
                return (
                    "你是主动加入对话的。根据上下文分析，用户可能在回应你之前的消息。"
                    "请谨慎判断，如果不确定，宁可保持观望。"
                )

            if is_talking_to_group:
                return (
                    "【注意】你是主动加入对话的，这条消息是说给群里的，不是在问你。"
                    "不要把这当作向你提问。"
                    "合适的做法：1)发表自己的看法 2)补充相关信息 3)保持沉默。"
                )

            # A 在和 B 说话，Bot 主动插话
            return (
                f"【重要】你是主动加入对话的！{msg.sender_name} 正在和 {msg.talking_to_name} 对话，不是在问你。"
                f"不要把别人的对话当成问你的。"
                f"合适的做法：1)以旁观者身份补充 2)等待被问到再回答 3)保持沉默。"
            )

        # ===== 未知触发 - 最保守处理 =====
        if trigger == TRIGGER_UNKNOWN:
            # 触发原因未知时，无论推断结果如何，都要非常保守
            if is_talking_to_bot:
                return (
                    "【谨慎】触发原因不明确。虽然上下文分析显示用户可能在和你说话，"
                    "但请仔细判断这是否真的是对你说的。如果不确定，请保持沉默或简短回应。"
                )
            
            if is_talking_to_group:
                return (
                    "【注意】触发原因不明确，这条消息是说给群里的。"
                    "在不确定的情况下，建议保持沉默或仅在有价值时简短补充。"
                )
            
            return (
                f"【注意】触发原因不明确。{msg.sender_name} 似乎在和 {msg.talking_to_name} 对话。"
                f"在不确定的情况下，建议保持沉默，避免误入他人对话。"
            )

        return ""


# ============================================================================
# Main Plugin
# ============================================================================


class Main(star.Star):
    """
    上下文场景感知插件

    通过分析群聊消息结构，为 LLM 提供结构化的场景描述，
    帮助 Bot 更好地理解对话情境并做出恰当回应。

    v2.4.0 更新：新增图像转述功能，可完全替代框架内置 LTM。
    """

    def __init__(
        self,
        context: star.Context,
        config: AstrBotConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config
        self._context = context  # 保存 context 用于获取 provider

        self._enabled = bool(self._cfg("enable", True))
        self._group_only = bool(self._cfg("only_group_chat", True))

        # 图像转述配置
        self._image_caption_enabled = bool(self._cfg("image_caption", False))
        self._image_caption_provider_id = str(self._cfg("image_caption_provider_id", "") or "")
        self._image_caption_prompt = str(
            self._cfg("image_caption_prompt", "请用中文简洁描述这张图片的内容，不超过50字。") or ""
        )

        self._sessions = SessionManager(
            max_messages=int(self._cfg("max_history", 50) or 50),
            max_sessions=int(self._cfg("max_groups", 100) or 100),
        )
        self._scene_generator = SceneGenerator()
        self._stats = PluginStats()

        self._bot_id: str | None = None
        self._analyzer: SceneAnalyzer | None = None

        # 图像转述统计
        self._image_caption_count = 0
        self._image_caption_errors = 0

        version = "2.4.0"
        caption_status = "已启用" if self._image_caption_enabled else "未启用"
        logger.info(f"[ContextAware] 插件 v{version} 已加载 | 图像转述: {caption_status}")

    def _cfg(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        if self._config is None:
            return default
        return self._config.get(key, default)

    def _should_process(self, event: AstrMessageEvent) -> bool:
        """判断是否应该处理此事件"""
        if not self._enabled:
            return False
        if self._group_only and event.is_private_chat():
            return False
        return True

    def _ensure_initialized(self, event: AstrMessageEvent) -> bool:
        """确保组件已初始化"""
        if self._analyzer is not None:
            return True

        self._bot_id = event.get_self_id()
        if not self._bot_id:
            logger.warning("[ContextAware] 无法获取 Bot ID，跳过处理")
            return False

        bot_names_raw = self._cfg("bot_names", [])
        bot_names: list[str] = []
        if isinstance(bot_names_raw, list):
            bot_names = [str(n) for n in bot_names_raw if n]

        self._analyzer = SceneAnalyzer(self._bot_id, bot_names)
        logger.info(f"[ContextAware] 初始化完成，Bot ID: {self._bot_id}")
        return True

    async def _get_image_caption(self, image_url: str) -> str | None:
        """获取图片描述"""
        if not self._image_caption_enabled:
            return None

        try:
            # 获取 provider
            provider = None
            if self._image_caption_provider_id:
                provider = self._context.get_provider_by_id(self._image_caption_provider_id)
                if not provider:
                    logger.warning(
                        f"[ContextAware] 找不到指定的图像转述提供商: {self._image_caption_provider_id}"
                    )
                    return None
            else:
                provider = self._context.get_using_provider()

            if not provider or not isinstance(provider, Provider):
                logger.warning("[ContextAware] 无法获取有效的 Provider 进行图像转述")
                return None

            # 调用 LLM 获取图片描述
            response = await provider.text_chat(
                prompt=self._image_caption_prompt,
                image_urls=[image_url],
            )

            if response and response.completion_text:
                self._image_caption_count += 1
                caption = response.completion_text.strip()
                # 限制长度
                if len(caption) > 100:
                    caption = caption[:97] + "..."
                logger.debug(f"[ContextAware] 图像转述成功: {caption[:30]}...")
                return caption

        except Exception as e:
            self._image_caption_errors += 1
            logger.error(f"[ContextAware] 图像转述失败: {e}")

        return None

    async def _extract_message_with_caption(
        self, event: AstrMessageEvent
    ) -> MessageRecord:
        """从事件提取消息记录，支持图像转述"""
        assert self._analyzer is not None

        sender_id = event.get_sender_id()
        parts: list[str] = []

        # 提取消息内容
        for comp in event.get_messages():
            if isinstance(comp, Plain) and comp.text:
                parts.append(comp.text)
            elif isinstance(comp, Image):
                # 尝试图像转述
                if self._image_caption_enabled:
                    image_url = comp.url if comp.url else comp.file
                    if image_url:
                        caption = await self._get_image_caption(image_url)
                        if caption:
                            parts.append(f"[图片: {caption}]")
                        else:
                            parts.append("[图片]")
                    else:
                        parts.append("[图片]")
                else:
                    parts.append("[图片]")

        content = "".join(parts) if parts else (event.message_str or "[消息]")

        msg = MessageRecord(
            msg_id=str(event.message_obj.message_id),
            sender_id=sender_id,
            sender_name=event.get_sender_name() or sender_id,
            content=content[:500],
            timestamp=time.time(),
            is_bot=(sender_id == self._analyzer._bot_id),
        )

        # 提取 @ 和回复信息
        for comp in event.get_messages():
            if isinstance(comp, At):
                qq_str = str(comp.qq)
                msg.at_targets.append((qq_str, comp.name or qq_str))
                if qq_str == self._analyzer._bot_id:
                    msg.at_bot = True
            elif isinstance(comp, Reply):
                if comp.sender_id:
                    msg.reply_to_id = str(comp.sender_id)

        return msg

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent, *args: Any, **kwargs: Any) -> None:
        """监听所有消息，记录到历史"""
        if not self._should_process(event):
            return

        has_content = any(
            isinstance(c, (Plain, Image)) for c in event.get_messages()
        )
        if not has_content:
            return

        if not self._ensure_initialized(event):
            return

        assert self._analyzer is not None

        # 使用支持图像转述的方法提取消息
        msg = await self._extract_message_with_caption(event)
        state = self._sessions.get(event.unified_msg_origin)
        self._analyzer.infer_addressee(
            msg,
            state.messages,
            bot_replied_to=state.bot_last_replied_to,
            bot_replied_to_name=state.bot_last_replied_to_name,
        )

        self._sessions.add_message(event.unified_msg_origin, msg)
        self._stats.messages_recorded += 1

        # 每记录 50 条消息输出一次统计
        if self._stats.messages_recorded % 50 == 0:
            caption_info = ""
            if self._image_caption_enabled:
                caption_info = f", 图像转述 {self._image_caption_count} 次"
            logger.info(
                f"[ContextAware] 统计: 已记录 {self._stats.messages_recorded} 条消息, "
                f"已注入 {self._stats.scenes_injected} 次场景, "
                f"活跃会话 {self._sessions.get_session_count()} 个{caption_info}"
            )

    @filter.on_llm_request(priority=-10)
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """在 LLM 请求前注入场景描述"""
        if not self._should_process(event):
            return

        if not self._ensure_initialized(event):
            return

        assert self._analyzer is not None

        umo = event.unified_msg_origin
        if not self._sessions.has_session(umo):
            # 使用支持图像转述的方法
            msg = await self._extract_message_with_caption(event)
            self._sessions.add_message(umo, msg)

        try:
            state = self._sessions.get(umo)
            if not state.messages:
                return

            current = state.messages[-1]
            trigger_type, trigger_desc = self._analyzer.detect_trigger(event, current)

            window = int(self._cfg("dialogue_window", 8) or 8)
            flow = state.messages[-window:]

            now = time.time()
            bot_status: dict[str, float | str | bool] = {}
            if state.bot_last_spoke_at > 0:
                mins = (now - state.bot_last_spoke_at) / 60
                bot_status = {
                    "active": True,
                    "minutes_ago": round(mins, 1),
                    "content": state.bot_last_content,
                }

            participants = list({m.sender_name for m in flow if not m.is_bot})

            scene = self._scene_generator.generate(
                trigger_type=trigger_type,
                trigger_desc=trigger_desc,
                current=current,
                flow=flow,
                bot_status=bot_status,
                participants=participants,
                show_flow=bool(self._cfg("enable_dialogue_flow", True)),
            )

            # 注入场景描述到请求
            # 优先使用 extra_user_content_parts，如果不存在则回退到 system_prompt
            try:
                if hasattr(req, 'extra_user_content_parts') and req.extra_user_content_parts is not None:
                    req.extra_user_content_parts.append(TextPart(text=scene))
                else:
                    # 回退方案：添加到 system_prompt
                    req.system_prompt = (req.system_prompt or "") + "\n\n" + scene
            except AttributeError:
                # 兼容旧版本 AstrBot
                req.system_prompt = (req.system_prompt or "") + "\n\n" + scene

            self._stats.scenes_injected += 1
            self._stats.record_trigger(trigger_type)

            # 关键日志：每次场景注入都输出
            trigger_name = TRIGGER_NAMES.get(trigger_type, trigger_type)
            talking_to_display = (
                "Bot" if current.talking_to == "bot"
                else ("群聊" if current.talking_to == "group" else current.talking_to_name)
            )
            logger.info(
                f"[ContextAware] ✓ 场景注入 #{self._stats.scenes_injected} | "
                f"触发: {trigger_name} | "
                f"{current.sender_name} → {talking_to_display} | "
                f"历史: {len(flow)} 条"
            )

        except Exception as e:
            logger.error(f"[ContextAware] 场景分析失败: {e}")

    @filter.on_llm_response()
    async def on_llm_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        """记录 Bot 回复"""
        if not self._should_process(event):
            return

        if not resp.completion_text:
            return

        if not self._ensure_initialized(event):
            return

        now = time.time()
        umo = event.unified_msg_origin
        
        # 获取当前消息的发送者（Bot 正在回复的人）
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name() or sender_id

        self._sessions.record_bot_response(
            umo,
            resp.completion_text,
            now,
            replied_to_id=sender_id,
            replied_to_name=sender_name,
        )

        bot_msg = MessageRecord(
            msg_id=f"bot_{now}",
            sender_id=self._bot_id or "bot",
            sender_name="[你]",
            content=resp.completion_text[:200],
            timestamp=now,
            is_bot=True,
            talking_to=sender_id,  # 记录 Bot 在回复谁
            talking_to_name=sender_name,
        )
        self._sessions.add_message(umo, bot_msg)
        self._stats.bot_responses_recorded += 1

        logger.debug(
            f"[ContextAware] Bot 回复已记录 (回复给: {sender_name}, 共 {self._stats.bot_responses_recorded} 次)"
        )

    async def terminate(self) -> None:
        """清理资源"""
        # 输出最终统计
        trigger_summary = ", ".join(
            f"{TRIGGER_NAMES.get(k, k)}: {v}"
            for k, v in sorted(self._stats.trigger_counts.items(), key=lambda x: -x[1])
        )
        caption_info = ""
        if self._image_caption_enabled:
            caption_info = f", 图像转述 {self._image_caption_count} 次"
            if self._image_caption_errors > 0:
                caption_info += f" (失败 {self._image_caption_errors})"
        logger.info(
            f"[ContextAware] 插件已终止 | "
            f"统计: 消息 {self._stats.messages_recorded}, "
            f"场景注入 {self._stats.scenes_injected}, "
            f"Bot回复 {self._stats.bot_responses_recorded}{caption_info} | "
            f"触发类型: {trigger_summary or '无'}"
        )
