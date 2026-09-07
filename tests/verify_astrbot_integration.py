"""Run explicitly with an AstrBot installation on PYTHONPATH; no API calls."""

import asyncio
import base64
import io
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from astrbot.core.agent.hooks import BaseAgentRunHooks  # noqa: E402
from astrbot.core.agent.message import Message, dump_messages_with_checkpoints  # noqa: E402
from astrbot.core.agent.run_context import ContextWrapper  # noqa: E402
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner  # noqa: E402
from astrbot.core.agent.tool import ToolSet  # noqa: E402
from astrbot.core.agent.tool_image_cache import tool_image_cache  # noqa: E402
from astrbot.core.provider.entities import LLMResponse, ProviderRequest  # noqa: E402
from astrbot.core.star.register.star_handler import llm_tools  # noqa: E402
from astrbot.api.message_components import Image as ImageComponent  # noqa: E402
from astrbot.core.platform.astr_message_event import AstrMessageEvent  # noqa: E402
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember  # noqa: E402
from astrbot.core.platform.message_type import MessageType  # noqa: E402
from astrbot.core.platform.platform_metadata import PlatformMetadata  # noqa: E402
from astrbot.core.pipeline.preprocess_stage.stage import PreProcessStage  # noqa: E402

from image_context import SEEN_KEY, SNAPSHOT_KEY, TOOL_NAME, strip_tool_images  # noqa: E402
from main import Main, MessageRecord, TRIGGER_AT  # noqa: E402


class Event:
    unified_msg_origin = "integration:GroupMessage:isolated"

    def __init__(self):
        self.extra = {}

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value

    def is_private_chat(self):
        return False


async def verify():
    with tempfile.TemporaryDirectory() as temporary:
        provider = SimpleNamespace(
            provider_config={"id": "test", "modalities": ["text", "image", "tool_use"]}
        )
        context = SimpleNamespace(
            get_using_provider=lambda **kw: provider,
            get_config=lambda **kw: {"wake_prefix": []},
        )
        plugin = Main(context, {"image_cache_dir": temporary})
        try:
            # Exercise Core preprocessing and the real event-finally cleanup.
            # The old background-only capture returned zero images here.
            for compressed in (False, True):
                raw = AstrBotMessage()
                raw.type = MessageType.GROUP_MESSAGE
                raw.self_id = "bot"
                raw.sender = MessageMember("a", "Alice")
                raw.message_id = f"lifecycle_{compressed}"
                raw.group_id = f"lifecycle_{compressed}"
                buffer = io.BytesIO()
                Image.new("RGB", (1024, 768), "blue").save(buffer, format="PNG")
                raw.message = [
                    ImageComponent(
                        file="base64://" + base64.b64encode(buffer.getvalue()).decode()
                    )
                ]
                raw.message_str = ""
                raw.raw_message = {}
                ev = AstrMessageEvent(
                    "", raw, PlatformMetadata("aiocqhttp", "test", "test"), raw.group_id
                )
                stage = PreProcessStage()
                stage.config = {}
                stage.platform_settings = {}
                stage.stt_settings = {}
                await stage.process(ev)
                owned = list(ev._temporary_local_files)
                assert owned, "Core must own the normalized image files"
                from image_compression import ImageCompressionOptions

                plugin._image_compress_options = ImageCompressionOptions.from_mapping(
                    {
                        "enable": compressed,
                        "max_edge": 512,
                    }
                )
                before_compressions = plugin._image_compress_count
                await plugin.on_message(ev)
                assert (
                    plugin._image_compress_count > before_compressions
                ) == compressed
                owned = list(ev._temporary_local_files)
                ids = tuple(plugin._image_index.sessions[ev.unified_msg_origin])
                ev.cleanup_temporary_local_files()
                assert all(not Path(p).exists() for p in owned)
                ev.set_extra(SNAPSHOT_KEY, ids)
                result = await plugin.context_aware_view_images(ev, list(ids))
                assert sum(p.type == "image" for p in result.content) == 1
                print(
                    f"PASS Core preprocess + immediate event cleanup + recall (compression={compressed})"
                )
            plugin._image_compress_options = ImageCompressionOptions.from_mapping({})
            event = Event()
            source_event = Event()
            plugin._stamp_image_event(source_event)
            buffer = io.BytesIO()
            Image.new("RGB", (24, 24), "red").save(buffer, format="PNG")
            source = MessageRecord(
                msg_id="source",
                sender_id="a",
                sender_name="Alice",
                content="[图片]",
                timestamp=time.time(),
                has_image=True,
                image_count=1,
                image_urls=[
                    "data:image/png;base64,"
                    + base64.b64encode(buffer.getvalue()).decode()
                ],
            )
            await plugin._record_recall_images(source_event, source)
            question = MessageRecord(
                msg_id="q",
                sender_id="a",
                sender_name="Alice",
                content="图里是什么？",
                timestamp=time.time(),
            )
            tool = next((t for t in llm_tools.func_list if t.name == TOOL_NAME), None)
            assert tool is not None
            assert (
                tool.parameters["properties"]["image_ids"]["items"]["type"] == "string"
            )
            assert tool.parameters["properties"]["detail"]["type"] == "string"
            request = ProviderRequest(
                prompt=question.content, func_tool=ToolSet([tool])
            )
            await plugin._prepare_image_recall(event, request, question, TRIGGER_AT)
            assembled = await request.assemble_context()
            assert sum(p.get("type") == "image_url" for p in assembled["content"]) == 1
            saved = dump_messages_with_checkpoints([Message.model_validate(assembled)])
            assert "base64" not in str(saved), saved
            assert question.content in str(saved)
            print(
                "PASS actual schema + automatic image assembly + temporary serialization"
            )

            tool_event = Event()
            ids = event.get_extra(SNAPSHOT_KEY)
            tool_event.set_extra(SNAPSHOT_KEY, ids)
            tool_event.set_extra(SEEN_KEY, {})
            calls = []

            async def text_chat(**kwargs):
                calls.append(kwargs["contexts"])
                if len(calls) == 1:
                    return LLMResponse(
                        role="assistant",
                        tools_call_name=[TOOL_NAME],
                        tools_call_args=[{"image_ids": list(ids), "detail": "auto"}],
                        tools_call_ids=["verify_call"],
                    )
                images = [
                    p
                    for m in kwargs["contexts"]
                    if isinstance(m.get("content"), list)
                    for p in m["content"]
                    if p.get("type") == "image_url"
                ]
                assert images and images[-1]["image_url"]["url"].startswith(
                    "data:image/"
                )
                return LLMResponse(
                    role="assistant", completion_text="red image verified"
                )

            provider.text_chat = text_chat

            class Executor:
                @classmethod
                async def execute(cls, tool, run_context, **kwargs):
                    yield await plugin.context_aware_view_images(tool_event, **kwargs)

            runner = ToolLoopAgentRunner()
            request = ProviderRequest(
                prompt="查看刚才的图片", func_tool=ToolSet([tool])
            )
            await runner.reset(
                provider=provider,
                request=request,
                run_context=ContextWrapper(context=None),
                tool_executor=Executor,
                agent_hooks=BaseAgentRunHooks(),
            )
            async for _ in runner.step_until_done(3):
                pass
            assert len(calls) == 2
            saved = dump_messages_with_checkpoints(runner.run_context.messages)
            assert "base64" in str(saved), (
                "Test must observe Core saving the tool image"
            )
            clean, removed = strip_tool_images(saved)
            assert removed == 1 and "base64" not in str(clean)
            assert any(m.get("tool_call_id") == "verify_call" for m in clean)
            print(
                "PASS actual tool loop delivers image + next-turn filter preserves protocol and removes pixels"
            )

            # Clean only files made by this integration run.
            for message in saved:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        image_id = part.get("image_url", {}).get("id")
                        if image_id and Path(image_id).parent == Path(
                            tool_image_cache._cache_dir
                        ):
                            Path(image_id).unlink(missing_ok=True)
        finally:
            await plugin.terminate()


if __name__ == "__main__":
    asyncio.run(verify())
