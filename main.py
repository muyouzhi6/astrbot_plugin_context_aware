"""
AstrBot 上下文场景感知增强插件 v3.1.6 (Context-Aware Enhancement)

为 LLM 提供结构化的群聊场景描述，增强其对对话情境的理解能力。
重点解决：主动回复时 Bot 误以为别人在问自己的问题。

核心功能:
- 触发类型检测: 被@、被回复、唤醒词、主动搭话、戳一戳
- 对话对象推断: 谁在和谁说话（关键功能）
- 对话流分析: 最近的对话结构
- Bot 状态追踪: 上次发言时间和内容
- 图像转述: 将群友发送的图片转为文字描述（可选）

设计原则:
- 只做加法，不修改框架原有信息
- 可完全替代框架内置 LTM 的群聊记录功能
- 轻量高效，图像转述为可选功能

v3.1.6 更新:
- [FIX] 语音转写独立上下文窗口，避免高频群聊把未回复语音挤出最近对话流

v3.1.5 更新:
- [FIX] 兼容 Gemini_STT 语音转写上下文，记录为普通群聊消息
- [FIX] 按消息 ID 幂等写入，避免 LLM 请求兜底记录和消息 handler 重复记录同一条语音

v3.1.4 更新:
- [FIX] 最近图片上下文支持过滤 GIF，避免不支持 image/gif 的模型在后续请求中报错
- [CONFIG] 新增 show_recent_images_allow_gif 开关，默认不将 GIF 加入 <recent_images>

v3.1.3 更新:
- [FIX] 图片上下文记录只扩展到图片概要，避免纯表情/@/引用等占位消息污染对话流
- [CHANGE] 最低 AstrBot 版本提高到 4.24.0，确保临时注入不会写入会话历史

v3.1.2 更新:
- [FIX] 唤醒词判定改为显式匹配 wake_prefix，避免把异常触发误判成 wake_word
- [FIX] 多个 @ 对象时保留完整对话目标，避免只显示第一个人

v3.1.1 更新:
- [FIX] 修复 SessionState 字段重复定义问题
- [FIX] 修复超时配置代码(300s)与schema(600s)不一致
- [FIX] 修复 Bot 消息 ID 使用时间戳可能冲突，改用 uuid

v3.0.0 更新 (重大重构):
- [CRITICAL] 修复并发竞态: SessionManager 添加异步锁 + deque 替代 list
- [HIGH] 图像转述优化: 并发限流(Semaphore) + 超时控制 + URL缓存
- [HIGH] 修复封装破坏: SceneAnalyzer 添加 bot_id 只读属性
- [HIGH] 消除魔法字符串: 集中定义 ExtraKeys 常量类
- [HIGH] 安全注入场景: 防止重复注入 + 兼容处理
- [HIGH] 对话推断增强: 关键锚点分离 + 推断原因追踪
- [MEDIUM] 配置工具方法: _cfg_int/_cfg_bool/_cfg_list
- [MEDIUM] 回复特征词可配置化
- [MEDIUM] 增强可观测性: 推断规则日志
- [LOW] 修复时间戳精度: 使用 uuid

v2.5.1 更新:
- 新增戳一戳触发类型（TRIGGER_POKE）
- 支持 poke_to_llm 插件的 _poke_trigger 标记
- 戳一戳时正确显示戳一戳用户信息

Author: 木有知
Version: 3.1.6
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, final

from astrbot import logger
from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, AtAll, Image, Plain, Reply
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart

if TYPE_CHECKING:
    from astrbot.core.config import AstrBotConfig


# ============================================================================
# Extra Keys - 消除魔法字符串
# ============================================================================


@final
class ExtraKeys:
    """框架 extra 字段键名常量，集中管理避免魔法字符串"""
    
    POKE_TRIGGER: Final[str] = "_poke_trigger"
    POKE_SENDER_ID: Final[str] = "_poke_sender_id"
    POKE_SENDER_NAME: Final[str] = "_poke_sender_name"
    ACTIVE_TRIGGER: Final[str] = "_active_trigger"
    ACTIVE_REPLY_TRIGGERED: Final[str] = "active_reply_triggered"
    CURRENT_MESSAGE_RECORD: Final[str] = "_context_aware_current_message_record"
    GEMINI_STT_TRANSCRIPT: Final[str] = "_gemini_stt_transcript"
    GEMINI_STT_RAW_TEXT: Final[str] = "_gemini_stt_raw_text"
    GEMINI_STT_CACHE_ONLY: Final[str] = "_gemini_stt_cache_only"
    GEMINI_STT_SHOULD_REPLY: Final[str] = "_gemini_stt_should_reply"
    GEMINI_STT_REPLY_REASON: Final[str] = "_gemini_stt_reply_reason"
    
    # 场景注入标记，防止重复注入
    SCENE_INJECTED_MARKER: Final[str] = "<!-- context_aware_scene_v3 -->"


# ============================================================================
# Constants
# ============================================================================

# 触发类型常量
TRIGGER_PRIVATE: Final = "private_chat"
TRIGGER_AT: Final = "at_bot"
TRIGGER_AT_ALL: Final = "at_all"
TRIGGER_REPLY: Final = "reply_to_bot"
TRIGGER_WAKE: Final = "wake_word"
TRIGGER_MENTION: Final = "mention"
TRIGGER_ACTIVE: Final = "active"
TRIGGER_POKE: Final = "poke"
TRIGGER_UNKNOWN: Final = "unknown"

# 触发类型中文名（用于日志）
TRIGGER_NAMES: Final = {
    TRIGGER_PRIVATE: "私聊",
    TRIGGER_AT: "@Bot",
    TRIGGER_AT_ALL: "@全体",
    TRIGGER_REPLY: "回复Bot",
    TRIGGER_WAKE: "唤醒词",
    TRIGGER_MENTION: "提及Bot",
    TRIGGER_ACTIVE: "主动触发",
    TRIGGER_POKE: "戳一戳",
    TRIGGER_UNKNOWN: "未知",
}

# 回复特征词（用于判断是否在回复 Bot）- 可通过配置覆盖
DEFAULT_REPLY_STARTERS: Final = frozenset({
    "好的", "好", "嗯", "是的", "对", "谢谢", "感谢", "收到",
    "明白", "知道了", "了解", "可以", "行", "没问题",
    "ok", "OK", "Ok", "好滴", "好哒", "好嘞", "okok",
})


# ============================================================================
# Inference Reasons - 推断原因追踪
# ============================================================================


@final
class InferenceReason:
    """对话对象推断原因常量"""
    
    RULE_1_AT_BOT: Final[str] = "rule_1_at_bot"           # 明确 @Bot
    RULE_2_AT_OTHER: Final[str] = "rule_2_at_other"       # @其他人
    RULE_3_REPLY: Final[str] = "rule_3_reply"             # 引用回复
    RULE_4_BOT_REPLIED: Final[str] = "rule_4_bot_replied" # Bot 刚回复过此人
    RULE_4B_BOT_INTERRUPTED: Final[str] = "rule_4b_bot_interrupted" # Bot 插话导致误判，回退给上一位对话者
    RULE_5_ABA_PATTERN: Final[str] = "rule_5_aba_pattern" # A-B-A 对话模式
    RULE_6_QUICK_FOLLOW: Final[str] = "rule_6_quick_follow"  # 快速连续对话
    DEFAULT_GROUP: Final[str] = "default_group"           # 默认群聊


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
    at_all: bool = False
    reply_to_id: str | None = None
    talking_to: str = "group"
    talking_to_name: str = "群聊"
    at_targets: list[tuple[str, str]] = field(default_factory=list)
    message_outline: str = ""
    has_image: bool = False
    image_count: int = 0
    has_gif: bool = False
    gif_count: int = 0


def _normalize_at_target(
    bot_id: str,
    target_id: str,
    target_name: str | None,
) -> tuple[str, str]:
    """统一 @ 目标的表示，Bot 使用稳定标识避免后续判断分裂。"""
    normalized_id = str(target_id or "").strip()
    if normalized_id == bot_id:
        return "bot", "你"

    normalized_name = str(target_name or normalized_id).strip() or normalized_id
    return normalized_id, normalized_name


def _unique_targets(targets: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """按 target_id 去重，保留首次出现顺序。"""
    seen: set[str] = set()
    normalized: list[tuple[str, str]] = []
    for target_id, target_name in targets:
        normalized_id = str(target_id or "").strip()
        if not normalized_id or normalized_id in seen:
            continue
        seen.add(normalized_id)
        normalized_name = str(target_name or normalized_id).strip() or normalized_id
        normalized.append((normalized_id, normalized_name))
    return normalized


def _format_name_list(names: list[str]) -> str:
    """把多个名字拼成更自然的中文列举。"""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}和{names[1]}"
    return f"{'、'.join(names[:-1])}和{names[-1]}"


def _clean_one_line(value: Any) -> str:
    """压缩消息概要为单行文本，避免注入上下文时破坏结构。"""
    text = "" if value is None else str(value)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _event_message_outline(event: AstrMessageEvent) -> str:
    """优先使用 AstrBot 消息概要，以保留图片/语音等非文本消息占位。"""
    transcript = _event_voice_transcript(event)
    if transcript:
        return transcript

    outline = ""
    try:
        outline = event.get_message_outline()
    except Exception:
        outline = ""
    if not outline:
        try:
            outline = event.get_message_str()
        except Exception:
            outline = ""
    if not outline:
        outline = str(getattr(event.message_obj, "message_str", "") or event.message_str or "")
    return _clean_one_line(outline)


def _event_voice_transcript(event: AstrMessageEvent) -> str:
    """读取 Gemini_STT 导出的语音转写，作为群聊上下文普通消息记录。"""
    getter = getattr(event, "get_extra", None)
    if not callable(getter):
        return ""
    try:
        transcript = getter(ExtraKeys.GEMINI_STT_TRANSCRIPT, "") or getter(
            ExtraKeys.GEMINI_STT_RAW_TEXT, ""
        )
    except Exception:
        return ""
    transcript = _clean_one_line(transcript)
    if not transcript:
        return ""
    return f"[语音转写] {transcript}"


def _looks_like_voice_transcript(text: str) -> bool:
    return _clean_one_line(text).startswith("[语音转写]")


def _looks_like_image_outline(text: str) -> bool:
    """识别平台概要中的图片占位，兼容不同适配器的文案。"""
    lowered = text.lower()
    return any(token in lowered for token in ("[图片", "图片", "照片", "[image", "image", "photo"))


_GIF_BASE64_PREFIXES: Final[tuple[str, str]] = ("R0lGODlh", "R0lGODdh")


def _image_ref_looks_like_gif(image_ref: str) -> bool:
    """尽量在投递给视觉模型前识别 GIF，避免不支持 image/gif 的模型报错。"""
    ref = (image_ref or "").strip()
    if not ref:
        return False

    lowered = ref.lower()
    if "image/gif" in lowered:
        return True

    ref_without_query = lowered.split("?", 1)[0].split("#", 1)[0]
    if ref_without_query.endswith(".gif"):
        return True

    local_path = ref
    if lowered.startswith("file:///"):
        local_path = ref[8:]
    elif lowered.startswith("file://"):
        local_path = ref[7:]
    if "://" not in local_path and not lowered.startswith(("data:", "base64:")):
        try:
            with open(local_path, "rb") as f:
                return f.read(6) in (b"GIF87a", b"GIF89a")
        except OSError:
            pass

    payload = ref
    lowered_payload = payload.lower()
    if lowered_payload.startswith("base64://"):
        payload = payload[len("base64://"):]
    elif lowered_payload.startswith("base64:"):
        payload = payload[len("base64:"):]

    if payload.lower().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    payload = payload.lstrip()
    return payload.startswith(_GIF_BASE64_PREFIXES)


def _explicit_addressees(
    msg: MessageRecord,
    *,
    bot_label: str = "你",
) -> list[tuple[str, str]]:
    """提取消息里显式出现的 @ 目标。"""
    targets = _unique_targets(msg.at_targets)
    if not targets:
        return []
    return [
        (target_id, bot_label if target_id == "bot" else target_name)
        for target_id, target_name in targets
    ]


def _other_explicit_target_names(msg: MessageRecord) -> list[str]:
    """获取除 Bot 外其他被显式点名的对象名称。"""
    return [
        target_name
        for target_id, target_name in _explicit_addressees(msg)
        if target_id != "bot"
    ]


def _describe_addressee(
    msg: MessageRecord,
    *,
    bot_label: str = "你（Bot）",
    group_label: str = "群聊",
    multi_target_bot_label: str = "你",
) -> str:
    """根据显式目标和推断结果生成更贴近真实群聊的对话对象描述。"""
    explicit_targets = _explicit_addressees(msg, bot_label=multi_target_bot_label)
    if len(explicit_targets) > 1:
        return _format_name_list([target_name for _, target_name in explicit_targets])
    if msg.talking_to == "bot":
        return bot_label
    if msg.talking_to == "group":
        return group_label
    if explicit_targets:
        return explicit_targets[0][1]
    return msg.talking_to_name or msg.talking_to


@dataclass(slots=True)
class SessionState:
    """会话状态 - 每个群聊/私聊一个

    v3.0.0: 使用 deque 替代 list，避免手动裁剪的非原子操作
    v3.1.1: 修复字段重复定义问题
    """

    messages: deque[MessageRecord] = field(default_factory=lambda: deque(maxlen=50))
    bot_last_spoke_at: float = 0.0
    bot_last_content: str = ""
    bot_last_replied_to: str = ""  # Bot 上次回复的对象 ID
    bot_last_replied_to_name: str = ""  # Bot 上次回复的对象名称
    # 关键锚点分离，不随消息淘汰
    last_user_interaction: dict[str, float] = field(default_factory=dict)  # user_id -> timestamp
    # 会话摘要（用于上下文压缩）
    summary: str = ""
    summary_updated_at: float = 0.0
    summary_message_count: int = 0
    compressing: bool = False


@dataclass
class PluginStats:
    """插件统计信息"""

    messages_recorded: int = 0
    scenes_injected: int = 0
    bot_responses_recorded: int = 0
    trigger_counts: dict[str, int] = field(default_factory=dict)

    def record_trigger(self, trigger_type: str) -> None:
        self.trigger_counts[trigger_type] = self.trigger_counts.get(trigger_type, 0) + 1


@dataclass(slots=True)
class SessionSnapshot:
    """会话快照（避免在锁外直接读写 SessionState 导致竞态）"""

    messages: list[MessageRecord]
    bot_last_spoke_at: float
    bot_last_content: str
    bot_last_replied_to: str
    bot_last_replied_to_name: str
    summary: str
    summary_updated_at: float
    summary_message_count: int


# ============================================================================
# Session Manager (LRU Cache)
# ============================================================================


class SessionManager:
    """会话管理器 - 带 LRU 淘汰机制和异步锁保护
    
    v3.0.0 重构:
    - 添加 asyncio.Lock 防止并发竞态
    - 使用 deque 自动裁剪，避免非原子操作
    - 淘汰会话时同时清理关联的锁
    
    v3.0.1 增强:
    - 添加缓存级别锁保护 LRU 的 move_to_end/popitem
    - 废弃同步写方法的直接使用（保留向后兼容但加警告）
    
    并发模型说明:
    - _cache_lock: 保护 _sessions (OrderedDict) 和 _locks (dict) 的结构性修改
    - 每会话锁: 保护单个会话的 messages/state 修改
    - 所有写操作应使用 async 版本
    """

    __slots__ = ("_sessions", "_locks", "_max_messages", "_max_sessions", "_cache_lock")

    def __init__(self, max_messages: int = 50, max_sessions: int = 100) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache_lock = asyncio.Lock()  # 缓存级别锁，保护 LRU 操作
        self._max_messages = max(10, max_messages)
        self._max_sessions = max(10, max_sessions)

    def _has_message_id(self, state: SessionState, msg_id: str) -> bool:
        normalized = str(msg_id or "").strip()
        if not normalized:
            return False
        return any(existing.msg_id == normalized for existing in state.messages)

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """获取会话锁（惰性创建，使用 setdefault 保证原子性）"""
        # setdefault 是原子操作，避免竞态条件
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def _get_or_create_session(self, session_id: str) -> SessionState:
        """获取或创建会话状态（异步，带缓存锁保护）
        
        这是并发安全的核心方法，保护 LRU 的 move_to_end 和 popitem。
        """
        async with self._cache_lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]

            while len(self._sessions) >= self._max_sessions:
                evicted_id, _ = self._sessions.popitem(last=False)
                # 清理关联的锁
                self._locks.pop(evicted_id, None)

            # 创建新会话时设置 deque 的 maxlen
            state = SessionState()
            state.messages = deque(maxlen=self._max_messages)
            self._sessions[session_id] = state
            return state

    def get(self, session_id: str) -> SessionState:
        """获取或创建会话状态（同步方法，用于读取）
        
        警告：此方法在并发场景下可能存在竞态。
        推荐在异步上下文中使用 _get_or_create_session()。
        """
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        while len(self._sessions) >= self._max_sessions:
            evicted_id, _ = self._sessions.popitem(last=False)
            self._locks.pop(evicted_id, None)

        state = SessionState()
        state.messages = deque(maxlen=self._max_messages)
        self._sessions[session_id] = state
        return state

    async def add_message_async(self, session_id: str, msg: MessageRecord) -> bool:
        """异步添加消息到会话（推荐使用，完全并发安全）"""
        async with self._get_lock(session_id):
            state = await self._get_or_create_session(session_id)
            if self._has_message_id(state, msg.msg_id):
                return False
            state.messages.append(msg)
            if not msg.is_bot:
                state.last_user_interaction[msg.sender_id] = msg.timestamp
            return True

    async def get_snapshot_async(self, session_id: str) -> SessionSnapshot:
        """获取会话快照（带会话锁）"""
        async with self._get_lock(session_id):
            state = await self._get_or_create_session(session_id)
            return SessionSnapshot(
                messages=list(state.messages),
                bot_last_spoke_at=state.bot_last_spoke_at,
                bot_last_content=state.bot_last_content,
                bot_last_replied_to=state.bot_last_replied_to,
                bot_last_replied_to_name=state.bot_last_replied_to_name,
                summary=state.summary,
                summary_updated_at=state.summary_updated_at,
                summary_message_count=state.summary_message_count,
            )

    async def mark_compressing_async(self, session_id: str) -> bool:
        """尝试标记会话正在压缩（避免并发重复压缩）。成功返回 True。"""
        async with self._get_lock(session_id):
            state = await self._get_or_create_session(session_id)
            if state.compressing:
                return False
            state.compressing = True
            return True

    async def clear_compressing_async(self, session_id: str) -> None:
        async with self._get_lock(session_id):
            if session_id in self._sessions:
                self._sessions[session_id].compressing = False

    async def set_summary_and_trim_async(
        self,
        session_id: str,
        *,
        summary: str,
        keep_recent: int,
        summarized_count: int,
        updated_at: float,
    ) -> None:
        """设置摘要并裁剪历史（带会话锁）"""
        keep_recent = max(5, keep_recent)
        async with self._get_lock(session_id):
            state = await self._get_or_create_session(session_id)
            msgs = list(state.messages)
            recent = msgs[-keep_recent:] if msgs else []
            state.messages = deque(recent, maxlen=state.messages.maxlen)
            state.summary = summary
            state.summary_updated_at = updated_at
            state.summary_message_count = max(state.summary_message_count, summarized_count)
            state.compressing = False

    async def remove_session_async(self, session_id: str) -> int:
        """移除整个会话（用于 reset/new/switch 等清空场景）"""
        async with self._cache_lock:
            state = self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
            if not state:
                return 0
            return len(state.messages)

    def add_message(self, session_id: str, msg: MessageRecord) -> bool:
        """同步添加消息（向后兼容，但不推荐在并发场景使用）
        
        注意：此方法不提供完整的并发保护，仅用于向后兼容。
        """
        state = self.get(session_id)
        if self._has_message_id(state, msg.msg_id):
            return False
        state.messages.append(msg)
        if not msg.is_bot:
            state.last_user_interaction[msg.sender_id] = msg.timestamp
        return True

    async def record_bot_response_async(
        self,
        session_id: str,
        content: str,
        ts: float,
        replied_to_id: str = "",
        replied_to_name: str = "",
    ) -> None:
        """异步记录 Bot 回复（推荐使用，完全并发安全）"""
        async with self._get_lock(session_id):
            state = await self._get_or_create_session(session_id)
            state.bot_last_spoke_at = ts
            state.bot_last_content = content[:100] if content else ""
            state.bot_last_replied_to = replied_to_id
            state.bot_last_replied_to_name = replied_to_name

    def record_bot_response(
        self,
        session_id: str,
        content: str,
        ts: float,
        replied_to_id: str = "",
        replied_to_name: str = "",
    ) -> None:
        """同步记录 Bot 回复（向后兼容）"""
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

    def get_messages_list(self, session_id: str) -> list[MessageRecord]:
        """获取消息列表（将 deque 转为 list，统一入口避免到处转换）"""
        if session_id in self._sessions:
            return list(self._sessions[session_id].messages)
        return []

    async def remove_message_by_id_async(self, session_id: str, msg_id: str) -> bool:
        """异步删除指定消息（带锁保护）
        
        供 recall_cancel 等插件调用，在消息撤回时清理记录。
        """
        async with self._get_lock(session_id):
            if session_id not in self._sessions:
                return False
            
            state = self._sessions[session_id]
            original_count = len(state.messages)
            new_messages: deque[MessageRecord] = deque(
                (m for m in state.messages if m.msg_id != msg_id),
                maxlen=state.messages.maxlen
            )
            state.messages = new_messages
            
            return original_count - len(state.messages) > 0

    def remove_message_by_id(self, session_id: str, msg_id: str) -> bool:
        """同步删除指定消息（向后兼容）"""
        if session_id not in self._sessions:
            return False
        
        state = self._sessions[session_id]
        original_count = len(state.messages)
        new_messages: deque[MessageRecord] = deque(
            (m for m in state.messages if m.msg_id != msg_id),
            maxlen=state.messages.maxlen
        )
        state.messages = new_messages
        
        return original_count - len(state.messages) > 0

    async def remove_last_bot_message_async(self, session_id: str) -> bool:
        """异步删除最后一条 Bot 消息（带锁保护）"""
        async with self._get_lock(session_id):
            if session_id not in self._sessions:
                return False
            
            state = self._sessions[session_id]
            if not state.messages:
                return False
            
            messages_list = list(state.messages)
            for i in range(len(messages_list) - 1, -1, -1):
                if messages_list[i].is_bot:
                    del messages_list[i]
                    state.messages = deque(messages_list, maxlen=state.messages.maxlen)
                    return True
            
            return False

    def remove_last_bot_message(self, session_id: str) -> bool:
        """同步删除最后一条 Bot 消息（向后兼容）"""
        if session_id not in self._sessions:
            return False
        
        state = self._sessions[session_id]
        if not state.messages:
            return False
        
        messages_list = list(state.messages)
        for i in range(len(messages_list) - 1, -1, -1):
            if messages_list[i].is_bot:
                del messages_list[i]
                state.messages = deque(messages_list, maxlen=state.messages.maxlen)
                return True
        
        return False


# ============================================================================
# Scene Analyzer
# ============================================================================


class SceneAnalyzer:
    """场景分析器 - 负责所有分析逻辑
    
    v3.0.0: 添加 bot_id 只读属性，支持自定义回复特征词
    """

    __slots__ = (
        "_bot_id",
        "_bot_names",
        "_bot_name_patterns",
        "_reply_starters",
        "_wake_prefixes",
    )

    def __init__(
        self, 
        bot_id: str, 
        bot_names: list[str] | None = None,
        reply_starters: frozenset[str] | None = None,
        wake_prefixes: list[str] | None = None,
    ) -> None:
        self._bot_id = bot_id
        names = [n.lower() for n in (bot_names or []) if n]
        self._bot_names: tuple[str, ...] = tuple(names)
        # 为英文/数字类名字做边界匹配，降低误触发（如 “robot” 包含 “bot”）
        compiled: list[tuple[str, re.Pattern[str] | None]] = []
        for n in names:
            if re.fullmatch(r"[a-z0-9_]+", n):
                compiled.append(
                    (n, re.compile(rf"(?<![\\w]){re.escape(n)}(?![\\w])", re.IGNORECASE))
                )
            else:
                compiled.append((n, None))
        self._bot_name_patterns: tuple[tuple[str, re.Pattern[str] | None], ...] = tuple(compiled)
        self._reply_starters = reply_starters or DEFAULT_REPLY_STARTERS
        self._wake_prefixes: tuple[str, ...] = tuple(
            str(prefix).strip()
            for prefix in (wake_prefixes or [])
            if str(prefix).strip()
        )

    @property
    def bot_id(self) -> str:
        """Bot ID 只读属性（v3.0.0: 修复封装破坏）"""
        return self._bot_id

    @staticmethod
    def _image_ref_from_component(comp: Image) -> str:
        return comp.url if comp.url else (comp.file or "")

    def extract_message(self, event: AstrMessageEvent) -> MessageRecord:
        """从事件提取消息记录"""
        sender_id = event.get_sender_id()
        message_outline = _event_message_outline(event)
        voice_transcript = _event_voice_transcript(event)
        image_count = 0
        gif_count = 0
        has_plain_text = False

        # 提取消息内容，拼接所有文本和图片描述
        content = voice_transcript or event.message_str or ""
        if not content:
            # message_str 为空时，从消息组件中拼接
            parts: list[str] = []
            for comp in event.get_messages():
                if isinstance(comp, Plain) and comp.text:
                    has_plain_text = True
                    parts.append(comp.text)
                elif isinstance(comp, Image):
                    image_count += 1
                    if _image_ref_looks_like_gif(self._image_ref_from_component(comp)):
                        gif_count += 1
                    parts.append("[图片]")
            content = "".join(parts) if parts else (message_outline or "[消息]")
        else:
            has_plain_text = True
            for comp in event.get_messages():
                if isinstance(comp, Image):
                    image_count += 1
                    if _image_ref_looks_like_gif(self._image_ref_from_component(comp)):
                        gif_count += 1
        has_image = image_count > 0 or (not has_plain_text and _looks_like_image_outline(message_outline))

        msg = MessageRecord(
            msg_id=str(event.message_obj.message_id),
            sender_id=sender_id,
            sender_name=event.get_sender_name() or sender_id,
            content=content[:500],
            timestamp=time.time(),
            is_bot=(sender_id == self._bot_id),
            message_outline=message_outline,
            has_image=has_image,
            image_count=max(image_count, 1 if has_image else 0),
            has_gif=gif_count > 0,
            gif_count=gif_count,
        )

        for comp in event.get_messages():
            if isinstance(comp, At):
                qq_str = str(comp.qq)
                msg.at_targets.append(
                    _normalize_at_target(self._bot_id, qq_str, comp.name or qq_str)
                )
                if qq_str == self._bot_id:
                    msg.at_bot = True
            elif isinstance(comp, AtAll):
                msg.at_all = True
            elif isinstance(comp, Reply):
                if comp.sender_id:
                    msg.reply_to_id = str(comp.sender_id)

        return msg

    def _is_explicit_wake_trigger(self, event: AstrMessageEvent) -> bool:
        """仅在能明确观察到 wake_prefix 时才判定为 wake_word。"""
        if not self._wake_prefixes:
            return False

        original_text = str(
            getattr(event.message_obj, "message_str", "") or event.message_str or ""
        ).strip()
        if not original_text:
            return False

        messages = event.get_messages()
        for wake_prefix in self._wake_prefixes:
            if not original_text.startswith(wake_prefix):
                continue
            if (
                not event.is_private_chat()
                and messages
                and isinstance(messages[0], At)
                and str(messages[0].qq) not in {self._bot_id, "all"}
            ):
                return False
            return True
        return False

    def detect_trigger(
        self, event: AstrMessageEvent, msg: MessageRecord
    ) -> tuple[str, str]:
        """检测触发类型"""
        sender = msg.sender_name

        # 检查是否为戳一戳触发（由 poke_to_llm 插件设置）
        if event.get_extra(ExtraKeys.POKE_TRIGGER):
            poke_sender_name = event.get_extra(ExtraKeys.POKE_SENDER_NAME) or sender
            return TRIGGER_POKE, f"{poke_sender_name} 戳了戳你，可能想让你回应之前的内容或想和你聊天"

        if event.is_private_chat():
            return TRIGGER_PRIVATE, f"私聊对话，{sender} 在直接和你交流"

        if msg.at_bot:
            return TRIGGER_AT, f"{sender} @了你，需要你回应"

        if msg.at_all:
            return TRIGGER_AT_ALL, f"{sender} @了全体成员（包含你），可能希望你回应"

        if msg.reply_to_id == self._bot_id:
            return TRIGGER_REPLY, f"{sender} 回复了你之前的消息"

        if self._is_explicit_wake_trigger(event):
            other_targets = _other_explicit_target_names(msg)
            if other_targets:
                return (
                    TRIGGER_WAKE,
                    f"{sender} 使用唤醒词呼叫你，并同时在和 {_format_name_list(other_targets)} 对话",
                )
            return TRIGGER_WAKE, f"{sender} 使用唤醒词呼叫你"

        if self._bot_names:
            msg_lower = msg.content.lower()
            for name, pat in self._bot_name_patterns:
                if pat:
                    if pat.search(msg_lower):
                        return TRIGGER_MENTION, f"{sender} 在消息中提到了你"
                else:
                    if name and name in msg_lower:
                        return TRIGGER_MENTION, f"{sender} 在消息中提到了你"

        if event.get_extra(ExtraKeys.ACTIVE_TRIGGER) or event.get_extra(ExtraKeys.ACTIVE_REPLY_TRIGGERED):
            return TRIGGER_ACTIVE, "你是主动加入这个对话的，没有人在叫你"

        if event.is_at_or_wake_command:
            other_targets = _other_explicit_target_names(msg)
            if other_targets and not msg.at_bot and not msg.at_all and msg.reply_to_id != self._bot_id:
                return (
                    TRIGGER_ACTIVE,
                    f"{sender} 明确在和 {_format_name_list(other_targets)} 对话，你是被动卷入的",
                )
            return TRIGGER_UNKNOWN, "存在触发信号，但不是在明确呼叫你"

        # 如果没有任何显式唤醒条件但仍触发了 LLM 请求，通常属于“主动回复/主动搭话”类场景
        # （例如 AstrBot 的主动回复功能或其他插件主动调用 request_llm）
        if not event.is_at_or_wake_command and not event.is_private_chat():
            return TRIGGER_ACTIVE, "你是主动加入这个对话的，没有人在叫你"

        return TRIGGER_UNKNOWN, "触发原因未知"

    def infer_addressee(
        self,
        msg: MessageRecord,
        history: list[MessageRecord] | deque[MessageRecord],
        bot_replied_to: str = "",
        bot_replied_to_name: str = "",
    ) -> str:
        """
        推断消息的对话对象
        
        核心原则：宁可保守（判定为群聊），不可激进（误判为和Bot说话）
        只有高置信度时才判定 talking_to = "bot"
        
        v3.0.0: 返回推断原因，用于可观测性
        
        Returns:
            推断原因常量 (InferenceReason.*)
        """
        explicit_targets = _explicit_addressees(msg)

        # ===== 规则1: 明确的 @ Bot（高置信度）=====
        if msg.at_bot:
            msg.talking_to = "bot"
            msg.talking_to_name = _format_name_list(
                [target_name for _, target_name in explicit_targets]
            ) or "你"
            return InferenceReason.RULE_1_AT_BOT

        # ===== 规则2: @ 其他人（高置信度）=====
        non_bot_targets = [
            (target_id, target_name)
            for target_id, target_name in explicit_targets
            if target_id != "bot"
        ]
        if non_bot_targets:
            target_id, _ = non_bot_targets[0]
            msg.talking_to = target_id
            msg.talking_to_name = _format_name_list(
                [target_name for _, target_name in non_bot_targets]
            )
            return InferenceReason.RULE_2_AT_OTHER

        # ===== 规则3: 引用回复消息（高置信度）=====
        if msg.reply_to_id:
            if msg.reply_to_id == self._bot_id:
                msg.talking_to, msg.talking_to_name = "bot", "你"
            else:
                msg.talking_to = msg.reply_to_id
                # 将 deque 转为可迭代的反向列表
                history_list = list(history) if isinstance(history, deque) else history
                for m in reversed(history_list):
                    if m.sender_id == msg.reply_to_id:
                        msg.talking_to_name = m.sender_name
                        break
                else:
                    msg.talking_to_name = msg.reply_to_id
            return InferenceReason.RULE_3_REPLY

        # ===== 以下是上下文推断，需要更保守 =====
        if not history:
            # 没有历史，保持默认 "group"
            return InferenceReason.DEFAULT_GROUP

        # 将 deque 转为 list 以支持切片
        history_list = list(history) if isinstance(history, deque) else history
        recent = [m for m in history_list[-5:] if m.sender_id != msg.sender_id]
        if not recent:
            return InferenceReason.DEFAULT_GROUP

        last = recent[-1]
        time_gap = msg.timestamp - last.timestamp

        # ===== 规则4: Bot 刚回复过当前用户，且用户像在回应（中置信度）=====
        # 关键修复：必须是 Bot 之前在回复"当前这个用户"，才能推断用户在回复 Bot
        if last.is_bot and time_gap < 35:
            # 检查 Bot 上次是否在回复当前发言者
            if bot_replied_to == msg.sender_id:
                stripped = msg.content.strip()
                # 保守：只对“短确认/致谢类”做推断，避免把用户对他人的“好的/嗯”等当成回复 Bot
                if stripped and len(stripped) <= 20 and self._looks_like_reply(stripped):
                    # 若 Bot 插话前，群里有人刚刚在和该用户说话，则优先认为用户在回那个人
                    history_list = list(history) if isinstance(history, deque) else history
                    prev_to_user: MessageRecord | None = None
                    for m in reversed(history_list[:-1]):
                        if msg.timestamp - m.timestamp > 90:
                            break
                        if m.is_bot or m.sender_id == msg.sender_id:
                            continue
                        if m.talking_to == msg.sender_id:
                            prev_to_user = m
                            break
                    if prev_to_user and (last.timestamp - prev_to_user.timestamp) < 60:
                        msg.talking_to, msg.talking_to_name = (
                            prev_to_user.sender_id,
                            prev_to_user.sender_name,
                        )
                        return InferenceReason.RULE_4B_BOT_INTERRUPTED

                    msg.talking_to, msg.talking_to_name = "bot", "你"
                    return InferenceReason.RULE_4_BOT_REPLIED
            # 如果 Bot 不是在回复这个人，则这个人的"谢谢"大概率不是对 Bot 说的
            # 保持 talking_to = "group"
            return InferenceReason.DEFAULT_GROUP

        # ===== 规则5: A-B-A 对话模式（低置信度，需要更多条件）=====
        # 只有当上一条消息明确是对当前用户说的，才推断当前用户在回复
        if last.talking_to == msg.sender_id and time_gap < 60:
            # 额外检查：上一条不是 Bot 发的（Bot 场景已在规则4处理）
            if not last.is_bot:
                msg.talking_to, msg.talking_to_name = last.sender_id, last.sender_name
                return InferenceReason.RULE_5_ABA_PATTERN

        # ===== 规则6: 快速连续对话（最低置信度，收紧条件）=====
        # 只有非常短的时间间隔 + 非 Bot 消息 + 上一条是对群说的，才推断是延续对话
        if time_gap < 15 and not last.is_bot:
            # 如果上一条是某人对群说的，当前消息可能是在回应那个人
            if last.talking_to == "group":
                msg.talking_to, msg.talking_to_name = last.sender_id, last.sender_name
                return InferenceReason.RULE_6_QUICK_FOLLOW
        
        # 默认：保持 talking_to = "group"，表示无法确定具体对话对象
        return InferenceReason.DEFAULT_GROUP

    def _looks_like_reply(self, content: str) -> bool:
        """判断是否像回复（v3.0.0: 使用可配置的回复特征词）"""
        stripped = content.strip()
        return any(stripped.startswith(s) for s in self._reply_starters)


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
        summary: str = "",
        *,
        show_flow: bool = True,
        show_recent_images: bool = True,
        show_recent_gifs: bool = True,
        image_flow: list[MessageRecord] | None = None,
        voice_flow: list[MessageRecord] | None = None,
    ) -> str:
        """生成场景描述，重点强调对话对象"""
        esc = self._escape
        parts: list[str] = ["<conversation_scene>"]

        # ===== 1. 触发类型 =====
        parts.append(f'  <trigger type="{trigger_type}">{esc(trigger_desc)}</trigger>')

        # ===== 2. 当前消息分析（最重要的部分）=====
        is_talking_to_bot = current.talking_to == "bot"
        is_talking_to_group = current.talking_to == "group"

        addressee_desc = _describe_addressee(
            current,
            bot_label="你（Bot）",
            group_label="群里所有人（非特定对象）",
            multi_target_bot_label="你",
        )

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
        if summary:
            parts.append(f'  <history_summary>{esc(summary[:600])}</history_summary>')

        if show_flow and len(flow) > 1:
            flow_lines: list[str] = []
            for m in flow[-5:]:
                to_name = _describe_addressee(
                    m,
                    bot_label="你",
                    group_label="群",
                    multi_target_bot_label="你",
                )
                sender = "[你]" if m.is_bot else m.sender_name
                preview = m.content[:20] + ("..." if len(m.content) > 20 else "")
                flow_lines.append(f'    <m>{esc(sender)} → {esc(to_name)}: {esc(preview)}</m>')
            parts.append('  <recent_flow>')
            parts.extend(flow_lines)
            parts.append('  </recent_flow>')

        if show_recent_images:
            image_lines: list[str] = []
            image_source = image_flow if image_flow is not None else flow
            for m in image_source:
                content = m.content or ""
                if not m.has_image and "[图片" not in content:
                    continue
                visible_image_count = max(m.image_count - m.gif_count, 0)
                if m.has_gif and not show_recent_gifs and visible_image_count <= 0:
                    continue
                to_name = _describe_addressee(
                    m,
                    bot_label="你",
                    group_label="群",
                    multi_target_bot_label="你",
                )
                sender = "[你]" if m.is_bot else m.sender_name
                preview_source = content or m.message_outline or "[图片]"
                preview = preview_source[:120] + ("..." if len(preview_source) > 120 else "")
                display_count = visible_image_count if m.has_gif and not show_recent_gifs else m.image_count
                count_attr = f' count="{display_count}"' if display_count > 1 else ""
                image_lines.append(
                    f'    <image sender="{esc(sender)}" talking_to="{esc(to_name)}"{count_attr}>'
                    f"{esc(preview)}</image>"
                )
            if image_lines:
                parts.append("  <recent_images>")
                parts.extend(image_lines)
                parts.append("  </recent_images>")

        voice_source = voice_flow if voice_flow is not None else flow
        voice_lines: list[str] = []
        for m in voice_source:
            content = m.content or ""
            if not _looks_like_voice_transcript(content):
                continue
            to_name = _describe_addressee(
                m,
                bot_label="你",
                group_label="群",
                multi_target_bot_label="你",
            )
            sender = "[你]" if m.is_bot else m.sender_name
            preview = content[:200] + ("..." if len(content) > 200 else "")
            voice_lines.append(
                f'    <voice sender="{esc(sender)}" talking_to="{esc(to_name)}">'
                f"{esc(preview)}</voice>"
            )
        if voice_lines:
            parts.append("  <recent_voice_transcripts>")
            parts.extend(voice_lines[-5:])
            parts.append("  </recent_voice_transcripts>")

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
        - 明确触发（@、回复、唤醒词、私聊、戳一戳）→ 正常回应
        - 主动触发 → 必须明确告知 Bot 它是主动插入的
        - 未知触发 → 最保守处理
        """
        shared_targets = _other_explicit_target_names(msg)

        # ===== 被明确呼叫，且同时点名了其他对象 =====
        if trigger in (TRIGGER_AT, TRIGGER_WAKE, TRIGGER_REPLY) and shared_targets:
            others_text = _format_name_list(shared_targets)
            return (
                f"用户正在同时对你和{others_text}说话，你是被共同点名的对象之一。"
                "请正常回应，但不要把这理解成只针对你一个人的单独提问。"
            )

        # ===== 被明确呼叫 - 正常回复 =====
        if trigger in (TRIGGER_AT, TRIGGER_AT_ALL, TRIGGER_REPLY, TRIGGER_WAKE, TRIGGER_PRIVATE):
            return "用户在和你对话，请正常回应。"

        # ===== 戳一戳触发 - 用户主动找你 =====
        if trigger == TRIGGER_POKE:
            return (
                "用户戳了戳你，这通常意味着希望你回应上下文中的内容。"
                "【优先级】1)回应用户最近的消息 2)继续之前的话题 3)只有上下文完全为空时才回应戳一戳本身。"
                "不要主动开新话题，不要撒娇卖萌。"
            )

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

    v3.0.0 重大更新：
    - 并发安全：SessionManager 添加异步锁
    - 图像转述优化：并发限流 + 超时 + 缓存
    - 封装修复：SceneAnalyzer 添加 bot_id 只读属性
    - 配置工具：_cfg_int/_cfg_bool/_cfg_list
    - 可观测性：推断规则日志
    """

    def __init__(
        self,
        context: star.Context,
        config: AstrBotConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config
        self._context = context  # 保存 context 用于获取 provider

        self._enabled = self._cfg_bool("enable", True)
        self._group_only = self._cfg_bool("only_group_chat", True)
        self._warn_builtin_ltm = self._cfg_bool("warn_builtin_ltm", True)
        self._show_recent_images = self._cfg_bool("show_recent_images", True)
        self._show_recent_images_allow_gif = self._cfg_bool("show_recent_images_allow_gif", False)
        self._image_context_window = max(1, self._cfg_int("image_context_window", 20))
        self._voice_context_window = max(0, self._cfg_int("voice_context_window", 50))
        self._builtin_ltm_warned: set[str] = set()

        # 图像转述配置
        self._image_caption_enabled = self._cfg_bool("image_caption", False)
        self._image_caption_provider_id = str(self._cfg("image_caption_provider_id", "") or "")
        self._image_caption_prompt = str(
            self._cfg("image_caption_prompt", "请用中文简洁描述这张图片的内容，不超过50字。") or ""
        )

        # v3.0.0: 图像转述并发控制
        self._image_caption_semaphore = asyncio.Semaphore(3)  # 最多并发3个
        self._image_caption_cache: OrderedDict[str, str] = OrderedDict()  # URL -> caption (LRU)
        self._image_caption_cache_max = 100  # 硬上限
        # 用户可配置超时（范围校验：10-600秒，与 schema 对齐）
        _timeout_cfg = self._cfg_int("image_caption_timeout", 60)
        if _timeout_cfg < 10 or _timeout_cfg > 600:
            logger.warning(
                f"[ContextAware] image_caption_timeout={_timeout_cfg} 超出合理范围(10-600)，已回退为60秒"
            )
            _timeout_cfg = 60
        self._image_caption_timeout = float(_timeout_cfg)

        # v3.1.0: 历史压缩（可选，默认关闭以避免额外 LLM 调用）
        self._history_compress_semaphore = asyncio.Semaphore(1)

        self._sessions = SessionManager(
            max_messages=self._cfg_int("max_history", 50),
            max_sessions=self._cfg_int("max_groups", 100),
        )
        self._scene_generator = SceneGenerator()
        self._stats = PluginStats()

        self._bot_id: str | None = None
        self._analyzer: SceneAnalyzer | None = None

        # 图像转述统计
        self._image_caption_count = 0
        self._image_caption_errors = 0
        self._image_caption_cache_hits = 0

        version = "3.1.6"
        caption_status = "已启用" if self._image_caption_enabled else "未启用"
        logger.info(f"[ContextAware] 插件 v{version} 已加载 | 图像转述: {caption_status}")

    def _cfg(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        if self._config is None:
            return default
        return self._config.get(key, default)

    def _cfg_int(self, key: str, default: int) -> int:
        """获取整数配置项（v3.0.0）"""
        val = self._cfg(key, default)
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        """获取布尔配置项（v3.0.0）"""
        val = self._cfg(key, default)
        if val is None:
            return default
        return bool(val)

    def _cfg_list(self, key: str, default: list[str] | None = None) -> list[str]:
        """获取列表配置项（v3.0.0）"""
        val = self._cfg(key, default or [])
        if isinstance(val, list):
            return [str(v) for v in val if v]
        return default or []

    def _inject_scene(self, req: ProviderRequest, scene: str) -> None:
        """安全注入场景描述到请求（v3.0.0: 防止重复注入 + 兼容处理）"""
        marker = ExtraKeys.SCENE_INJECTED_MARKER
        
        # 检查是否已注入（防止重复）
        if hasattr(req, 'system_prompt') and req.system_prompt and marker in req.system_prompt:
            logger.debug("[ContextAware] 场景已注入，跳过重复注入")
            return
        
        # 优先使用 extra_user_content_parts
        try:
            extra_parts = getattr(req, 'extra_user_content_parts', None)
            if extra_parts is not None and isinstance(extra_parts, list):
                part = TextPart(text=scene)
                mark_as_temp = getattr(part, "mark_as_temp", None)
                if callable(mark_as_temp):
                    temp_part = mark_as_temp()
                    if temp_part is not None:
                        part = temp_part
                extra_parts.append(part)
                return
        except Exception:
            pass
        
        # 回退方案：添加到 system_prompt（带标记）
        try:
            req.system_prompt = (req.system_prompt or "") + f"\n\n{marker}\n{scene}"
        except Exception as e:
            logger.error(f"[ContextAware] 场景注入失败: {e}")

    def _should_process(self, event: AstrMessageEvent) -> bool:
        """判断是否应该处理此事件"""
        if not self._enabled:
            return False
        if self._group_only and event.is_private_chat():
            return False
        return True

    def _builtin_ltm_enabled(self, event: AstrMessageEvent) -> bool:
        """检测 AstrBot 内置群聊上下文感知是否启用，避免重复注入。"""
        try:
            cfg = self._context.get_config(umo=event.unified_msg_origin)
        except TypeError:
            cfg = self._context.get_config()
        except Exception:
            return False
        if not cfg:
            return False
        try:
            settings = cfg.get("provider_ltm_settings", {})
            return bool(settings.get("group_icl_enable", False))
        except Exception:
            return False

    def _warn_if_builtin_ltm_enabled(self, event: AstrMessageEvent) -> None:
        if not self._warn_builtin_ltm:
            return
        umo = event.unified_msg_origin
        if umo in self._builtin_ltm_warned:
            return
        if not self._builtin_ltm_enabled(event):
            return
        self._builtin_ltm_warned.add(umo)
        logger.warning(
            "[ContextAware] 检测到 AstrBot 内置群聊上下文感知已启用，"
            "建议关闭 provider_ltm_settings.group_icl_enable，避免重复注入群聊历史。"
        )

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

        # v3.0.0: 支持自定义回复特征词
        custom_starters = self._cfg_list("reply_starters", None)
        reply_starters = frozenset(custom_starters) if custom_starters else None
        astrbot_config = self._context.get_config()
        wake_prefixes_raw = astrbot_config.get("wake_prefix", []) if astrbot_config else []
        wake_prefixes = [str(prefix) for prefix in wake_prefixes_raw if prefix]

        self._analyzer = SceneAnalyzer(
            bot_id=self._bot_id, 
            bot_names=bot_names,
            reply_starters=reply_starters,
            wake_prefixes=wake_prefixes,
        )
        logger.info(f"[ContextAware] 初始化完成，Bot ID: {self._bot_id}")
        return True

    # -------------------------------------------------------------------------
    # History Compression (Optional)
    # -------------------------------------------------------------------------

    def _history_compress_cfg(self) -> dict[str, Any]:
        """读取历史压缩配置（插件内置；默认关闭以避免额外 LLM 调用）"""
        strategy = str(self._cfg("history_compress_strategy", "off") or "off")
        return {
            "strategy": strategy,  # off | llm_summary
            "trigger_count": self._cfg_int("history_compress_trigger_count", 48),
            "keep_recent": self._cfg_int("history_compress_keep_recent", 16),
            "min_interval_sec": self._cfg_int("history_compress_min_interval_sec", 300),
            "provider_id": str(self._cfg("history_compress_provider_id", "") or ""),
            "instruction": str(self._cfg("history_compress_instruction", "") or ""),
            "timeout_sec": float(self._cfg_int("history_compress_timeout", 60)),
            "max_input_chars": self._cfg_int("history_compress_max_input_chars", 5000),
            "max_summary_chars": self._cfg_int("history_compress_max_summary_chars", 800),
        }

    def _build_summary_input(self, msgs: list[MessageRecord], *, max_chars: int) -> str:
        lines: list[str] = []
        for m in msgs:
            sender = "[你]" if m.is_bot else m.sender_name
            to = _describe_addressee(
                m,
                bot_label="[你]",
                group_label="群聊",
                multi_target_bot_label="[你]",
            )
            content = (m.content or "").replace("\n", " ").strip()
            if len(content) > 120:
                content = content[:117] + "..."
            lines.append(f"{sender} -> {to}: {content}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        # 输入过长时保留末尾（更贴近当前主题）
        return text[-max_chars:]

    async def _maybe_compress_history(self, umo: str, snapshot: SessionSnapshot) -> SessionSnapshot:
        cfg = self._history_compress_cfg()
        if cfg["strategy"] != "llm_summary":
            return snapshot

        trigger_count = max(10, int(cfg["trigger_count"]))
        keep_recent = max(5, int(cfg["keep_recent"]))
        if len(snapshot.messages) < trigger_count or len(snapshot.messages) <= keep_recent + 5:
            return snapshot

        now = time.time()
        if snapshot.summary_updated_at and (now - snapshot.summary_updated_at) < float(cfg["min_interval_sec"]):
            return snapshot

        # 避免同一会话并发重复压缩
        if not await self._sessions.mark_compressing_async(umo):
            return snapshot

        try:
            async with self._history_compress_semaphore:
                provider: Provider | None = None
                provider_id = str(cfg["provider_id"] or "")
                if provider_id:
                    p = self._context.get_provider_by_id(provider_id)
                    if isinstance(p, Provider):
                        provider = p
                else:
                    p = self._context.get_using_provider(umo)
                    if isinstance(p, Provider):
                        provider = p

                if not provider:
                    await self._sessions.clear_compressing_async(umo)
                    return snapshot

                instruction = cfg["instruction"].strip()
                if not instruction:
                    instruction = (
                        "你是“群聊上下文压缩器”。请将下面这段群聊/机器人对话历史压缩成一段简洁中文摘要，要求：\n"
                        "1) 保留关键事实、结论、已达成的决定、正在讨论的话题、未解决的问题。\n"
                        "2) 尽量保留人物关系与称呼（谁在对谁说什么），但不要逐条复述。\n"
                        "3) 输出长度控制在 200-600 字，避免空话套话。\n"
                    )

                to_summarize = snapshot.messages[:-keep_recent]
                input_text = self._build_summary_input(
                    to_summarize, max_chars=int(cfg["max_input_chars"])
                )

                prompt_parts = []
                if snapshot.summary:
                    prompt_parts.append(f"已有摘要（可在此基础上更新）：\n{snapshot.summary}\n")
                prompt_parts.append(f"需要压缩的历史：\n{input_text}\n")
                prompt_parts.append("请输出新的摘要：")
                prompt = "\n".join(prompt_parts)

                try:
                    resp = await asyncio.wait_for(
                        provider.text_chat(
                            prompt=prompt,
                            system_prompt=instruction,
                            session_id=uuid.uuid4().hex,
                            persist=False,
                        ),
                        timeout=float(cfg["timeout_sec"]),
                    )
                except asyncio.TimeoutError:
                    await self._sessions.clear_compressing_async(umo)
                    return snapshot

                if not resp or not resp.completion_text:
                    await self._sessions.clear_compressing_async(umo)
                    return snapshot

                summary = resp.completion_text.strip()
                max_summary_chars = int(cfg["max_summary_chars"])
                if len(summary) > max_summary_chars:
                    summary = summary[: max_summary_chars - 3] + "..."

                await self._sessions.set_summary_and_trim_async(
                    umo,
                    summary=summary,
                    keep_recent=keep_recent,
                    summarized_count=snapshot.summary_message_count + len(to_summarize),
                    updated_at=now,
                )

                return await self._sessions.get_snapshot_async(umo)
        except Exception as e:
            logger.error(f"[ContextAware] 历史压缩失败: {e}")
            await self._sessions.clear_compressing_async(umo)
            return snapshot

    async def _get_image_caption(self, image_url: str) -> str | None:
        """获取图片描述（v3.0.0: 并发限流 + 超时 + 缓存）"""
        if not self._image_caption_enabled:
            return None

        # 缓存命中检查
        if image_url in self._image_caption_cache:
            self._image_caption_cache_hits += 1
            # 移动到末尾（LRU 更新）
            self._image_caption_cache.move_to_end(image_url)
            return self._image_caption_cache[image_url]

        try:
            # 并发限流
            async with self._image_caption_semaphore:
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

                # 调用 LLM 获取图片描述（带超时）
                try:
                    response = await asyncio.wait_for(
                        provider.text_chat(
                            prompt=self._image_caption_prompt,
                            image_urls=[image_url],
                        ),
                        timeout=self._image_caption_timeout
                    )
                except asyncio.TimeoutError:
                    self._image_caption_errors += 1
                    logger.warning(f"[ContextAware] 图像转述超时 ({self._image_caption_timeout}s)")
                    return None

                if response and response.completion_text:
                    self._image_caption_count += 1
                    caption = response.completion_text.strip()
                    # 限制长度
                    if len(caption) > 100:
                        caption = caption[:97] + "..."
                    # 缓存结果（使用 OrderedDict 实现 LRU）
                    self._image_caption_cache[image_url] = caption
                    # LRU 淘汰：超过硬上限时移除最旧的
                    while len(self._image_caption_cache) > self._image_caption_cache_max:
                        self._image_caption_cache.popitem(last=False)
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
        message_outline = _event_message_outline(event)
        voice_transcript = _event_voice_transcript(event)
        image_count = 0
        gif_count = 0
        has_plain_text = False

        if voice_transcript:
            parts.append(voice_transcript)
            has_plain_text = True

        # 提取消息内容
        for comp in event.get_messages():
            if isinstance(comp, Plain) and comp.text and not voice_transcript:
                has_plain_text = True
                parts.append(comp.text)
            elif isinstance(comp, Image):
                image_count += 1
                image_url = SceneAnalyzer._image_ref_from_component(comp)
                is_gif = _image_ref_looks_like_gif(image_url)
                if is_gif:
                    gif_count += 1
                # 尝试图像转述
                if self._image_caption_enabled and (
                    not is_gif or self._show_recent_images_allow_gif
                ):
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

        has_image = image_count > 0 or (not has_plain_text and _looks_like_image_outline(message_outline))
        content = "".join(parts) if parts else (message_outline or "[消息]")
        if has_image and image_count == 0 and "[图片" not in content:
            content = f"[图片] {content}".strip()

        msg = MessageRecord(
            msg_id=str(event.message_obj.message_id),
            sender_id=sender_id,
            sender_name=event.get_sender_name() or sender_id,
            content=content[:500],
            timestamp=time.time(),
            is_bot=(sender_id == self._analyzer.bot_id),
            message_outline=message_outline,
            has_image=has_image,
            image_count=max(image_count, 1 if has_image else 0),
            has_gif=gif_count > 0,
            gif_count=gif_count,
        )

        # 提取 @ 和回复信息
        for comp in event.get_messages():
            if isinstance(comp, At):
                qq_str = str(comp.qq)
                msg.at_targets.append(
                    _normalize_at_target(self._analyzer.bot_id, qq_str, comp.name or qq_str)
                )
                if qq_str == self._analyzer.bot_id:
                    msg.at_bot = True
            elif isinstance(comp, AtAll):
                msg.at_all = True
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

        message_outline = _event_message_outline(event)
        messages = event.get_messages()
        has_content = any(isinstance(c, (Plain, Image)) for c in messages)
        has_content = has_content or _looks_like_image_outline(message_outline)
        has_content = has_content or bool(_event_voice_transcript(event))
        if not has_content:
            return

        if not self._ensure_initialized(event):
            return

        assert self._analyzer is not None

        # 使用支持图像转述的方法提取消息
        msg = await self._extract_message_with_caption(event)
        snapshot = await self._sessions.get_snapshot_async(event.unified_msg_origin)
        inference_reason = self._analyzer.infer_addressee(
            msg,
            snapshot.messages,
            bot_replied_to=snapshot.bot_last_replied_to,
            bot_replied_to_name=snapshot.bot_last_replied_to_name,
        )

        # v3.0.0: 推断规则日志（可观测性增强）
        # 绑定当前事件的消息记录，供 on_llm_request 精确取 current/flow（避免并发取错最后一条）
        try:
            event.set_extra(ExtraKeys.CURRENT_MESSAGE_RECORD, msg)
        except Exception:
            pass

        if self._cfg_bool("debug_inference", False):
            talking_to_display = _describe_addressee(
                msg,
                bot_label="Bot",
                group_label="群聊",
                multi_target_bot_label="Bot",
            )
            logger.debug(
                f"[ContextAware] 推断: {msg.sender_name} → {talking_to_display} "
                f"(规则: {inference_reason})"
            )

        # v3.0.0: 使用异步方法确保并发安全
        added = await self._sessions.add_message_async(event.unified_msg_origin, msg)
        if added:
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
        self._warn_if_builtin_ltm_enabled(event)

        umo = event.unified_msg_origin
        if not self._sessions.has_session(umo):
            # 使用支持图像转述的方法
            msg = await self._extract_message_with_caption(event)
            added = await self._sessions.add_message_async(umo, msg)
            if added:
                self._stats.messages_recorded += 1

        try:
            snapshot = await self._sessions.get_snapshot_async(umo)
            if not snapshot.messages:
                return

            # 检查是否为戳一戳触发
            is_poke_trigger = bool(event.get_extra(ExtraKeys.POKE_TRIGGER))
            
            if is_poke_trigger:
                # 戳一戳触发时，创建虚拟的 current 消息表示戳一戳用户
                poke_sender_id = event.get_extra(ExtraKeys.POKE_SENDER_ID) or event.get_sender_id()
                poke_sender_name = event.get_extra(ExtraKeys.POKE_SENDER_NAME) or event.get_sender_name() or poke_sender_id
                current = MessageRecord(
                    msg_id=f"poke_{uuid.uuid4().hex[:12]}",
                    sender_id=str(poke_sender_id),
                    sender_name=str(poke_sender_name),
                    content=f"[戳了戳你]",
                    timestamp=time.time(),
                    is_bot=False,
                    talking_to="bot",
                    talking_to_name="你",
                )
                flow_source = snapshot.messages
            else:
                current_from_extra = event.get_extra(ExtraKeys.CURRENT_MESSAGE_RECORD, None)
                current = (
                    current_from_extra
                    if isinstance(current_from_extra, MessageRecord)
                    else snapshot.messages[-1]
                )

                # 并发保护：flow 只截取到 current 为止，避免把其他并发消息带进来
                flow_source = snapshot.messages
                try:
                    idx = next(
                        (i for i, m in enumerate(flow_source) if m.msg_id == current.msg_id),
                        -1,
                    )
                    if idx >= 0:
                        flow_source = flow_source[: idx + 1]
                except Exception:
                    pass
                
            # 可选：压缩历史（会裁剪 flow_source 对应的底层会话）
            snapshot2 = await self._maybe_compress_history(umo, snapshot)
            if snapshot2 is not snapshot:
                snapshot = snapshot2
                flow_source = snapshot.messages
                if not is_poke_trigger:
                    try:
                        idx2 = next(
                            (i for i, m in enumerate(flow_source) if m.msg_id == current.msg_id),
                            -1,
                        )
                        if idx2 >= 0:
                            flow_source = flow_source[: idx2 + 1]
                    except Exception:
                        pass

            trigger_type, trigger_desc = self._analyzer.detect_trigger(event, current)

            window = self._cfg_int("dialogue_window", 8)
            flow = flow_source[-window:] if window > 0 else flow_source
            image_flow = (
                flow_source[-self._image_context_window:]
                if self._image_context_window > 0
                else flow_source
            )
            voice_flow = (
                flow_source[-self._voice_context_window:]
                if self._voice_context_window > 0
                else []
            )

            now = time.time()
            bot_status: dict[str, float | str | bool] = {}
            if snapshot.bot_last_spoke_at > 0:
                mins = (now - snapshot.bot_last_spoke_at) / 60
                bot_status = {
                    "active": True,
                    "minutes_ago": round(mins, 1),
                    "content": snapshot.bot_last_content,
                }

            participants = list({m.sender_name for m in flow if not m.is_bot})

            scene = self._scene_generator.generate(
                trigger_type=trigger_type,
                trigger_desc=trigger_desc,
                current=current,
                flow=flow,
                bot_status=bot_status,
                participants=participants,
                summary=snapshot.summary,
                show_flow=bool(self._cfg("enable_dialogue_flow", True)),
                show_recent_images=self._show_recent_images,
                show_recent_gifs=self._show_recent_images_allow_gif,
                image_flow=image_flow,
                voice_flow=voice_flow,
            )

            # 注入场景描述到请求（v3.0.0: 防止重复注入）
            self._inject_scene(req, scene)

            self._stats.scenes_injected += 1
            self._stats.record_trigger(trigger_type)

            # 关键日志：每次场景注入都输出
            trigger_name = TRIGGER_NAMES.get(trigger_type, trigger_type)
            talking_to_display = _describe_addressee(
                current,
                bot_label="Bot",
                group_label="群聊",
                multi_target_bot_label="Bot",
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

        await self._sessions.record_bot_response_async(
            umo,
            resp.completion_text,
            now,
            replied_to_id=sender_id,
            replied_to_name=sender_name,
        )

        bot_msg = MessageRecord(
            msg_id=f"bot_{uuid.uuid4().hex[:12]}",
            sender_id=self._bot_id or "bot",
            sender_name="[你]",
            content=resp.completion_text[:200],
            timestamp=now,
            is_bot=True,
            talking_to=sender_id,  # 记录 Bot 在回复谁
            talking_to_name=sender_name,
        )
        await self._sessions.add_message_async(umo, bot_msg)
        self._stats.bot_responses_recorded += 1

        logger.debug(
            f"[ContextAware] Bot 回复已记录 (回复给: {sender_name}, 共 {self._stats.bot_responses_recorded} 次)"
        )

    # -------------------------------------------------------------------------
    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        """跟随系统 reset/new/switch 清空本插件会话上下文（不注册新指令，避免冲突）"""
        try:
            if event.get_extra("_clean_ltm_session", False):
                removed = await self._sessions.remove_session_async(event.unified_msg_origin)
                if removed:
                    logger.info(
                        f"[ContextAware] 检测到会话清空标记，已清理 {event.unified_msg_origin} 的 {removed} 条上下文记录"
                    )
        except Exception as e:
            logger.error(f"[ContextAware] 清理会话失败: {e}")

    # Public API - 供其他插件调用
    # -------------------------------------------------------------------------

    def get_recent_messages(
        self,
        unified_msg_origin: str,
        count: int = 10,
    ) -> list[dict[str, Any]]:
        """获取指定会话的最近消息历史

        供其他插件（如 poke_to_llm）调用，获取群聊上下文。

        Args:
            unified_msg_origin: 会话标识 (event.unified_msg_origin)
            count: 获取的消息数量，默认 10

        Returns:
            消息列表，每条消息包含:
            - sender_name: 发送者名称
            - content: 消息内容
            - timestamp: 时间戳
            - is_bot: 是否为 Bot 消息
            - talking_to: 对话对象
        """
        if not self._sessions.has_session(unified_msg_origin):
            return []

        state = self._sessions.get(unified_msg_origin)
        # v3.0.0: 将 deque 转为 list 以支持切片
        messages_list = list(state.messages)
        messages = messages_list[-count:] if count > 0 else messages_list

        return [
            {
                "sender_name": msg.sender_name,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "is_bot": msg.is_bot,
                "talking_to": _describe_addressee(
                    msg,
                    bot_label="你",
                    group_label="群聊",
                    multi_target_bot_label="你",
                ),
                "has_image": msg.has_image,
                "image_count": msg.image_count,
                "has_gif": msg.has_gif,
                "gif_count": msg.gif_count,
                "message_outline": msg.message_outline,
            }
            for msg in messages
        ]

    def get_formatted_context(
        self,
        unified_msg_origin: str,
        count: int = 10,
    ) -> str:
        """获取格式化的群聊上下文字符串

        供其他插件调用，直接获取可注入 LLM 的上下文文本。

        Args:
            unified_msg_origin: 会话标识
            count: 获取的消息数量

        Returns:
            格式化的对话上下文字符串
        """
        messages = self.get_recent_messages(unified_msg_origin, count)
        if not messages:
            return ""

        lines: list[str] = []
        if self._sessions.has_session(unified_msg_origin):
            state = self._sessions.get(unified_msg_origin)
            summary = getattr(state, "summary", "") or ""
            if summary:
                lines.append("[历史摘要]")
                lines.append(summary)
                lines.append("")

        lines.append("[最近的群聊消息]")
        for msg in messages:
            name = "[你]" if msg["is_bot"] else msg["sender_name"]
            lines.append(f"{name}: {msg['content']}")

        return "\n".join(lines)

    def has_session(self, unified_msg_origin: str) -> bool:
        """检查是否有该会话的消息记录

        Args:
            unified_msg_origin: 会话标识

        Returns:
            是否存在该会话
        """
        return self._sessions.has_session(unified_msg_origin)

    def remove_message(self, unified_msg_origin: str, msg_id: str) -> bool:
        """删除指定会话中的指定消息
        
        供 recall_cancel 等插件调用，在消息撤回时清理记录。
        
        Args:
            unified_msg_origin: 会话标识
            msg_id: 要删除的消息ID
            
        Returns:
            是否成功删除
        """
        result = self._sessions.remove_message_by_id(unified_msg_origin, msg_id)
        if result:
            logger.debug(f"[ContextAware] 已删除消息记录 msg_id={msg_id}")
        return result

    def remove_last_bot_response(self, unified_msg_origin: str) -> bool:
        """删除指定会话中最后一条 Bot 回复
        
        供 recall_cancel 等插件调用，在撤回时同时清理 Bot 的回复记录。
        
        Args:
            unified_msg_origin: 会话标识
            
        Returns:
            是否成功删除
        """
        result = self._sessions.remove_last_bot_message(unified_msg_origin)
        if result:
            logger.debug("[ContextAware] 已删除最后一条 Bot 回复记录")
        return result

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
