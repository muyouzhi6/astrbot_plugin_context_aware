"""Real Core and plugin objects; no network calls or platform sends.

Usage: python verify_two_plugin_bridge.py CONTEXT_PLUGIN_DIR GITEE_PLUGIN_DIR
AstrBot must be installed or available on PYTHONPATH.
"""

# ruff: noqa: E402
import asyncio
import base64
import importlib.util
import io
import logging
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock
from PIL import Image as PILImage


def load_package(name, root):
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".main", root / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name + ".main"] = mod
    spec.loader.exec_module(mod)
    return mod


ca = load_package("bridge_context", Path(sys.argv[1]))
gi = load_package("bridge_gitee", Path(sys.argv[2]))
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.pipeline.preprocess_stage.stage import PreProcessStage
from astrbot.api.message_components import Image, Plain, At
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.agent.tool import ToolSet
from astrbot.core.star.register.star_handler import llm_tools


def event(mid, sender="alice", text="", components=None):
    m = AstrBotMessage()
    m.type = MessageType.GROUP_MESSAGE
    m.self_id = "bot"
    m.sender = MessageMember(sender, sender)
    m.message_id = mid
    m.group_id = "group"
    m.session_id = "group"
    m.message = components or [At(qq="bot"), Plain(text)]
    m.message_str = text
    m.raw_message = {}
    e = AstrMessageEvent(
        text, m, PlatformMetadata("aiocqhttp", "isolated test", "bridge"), "group"
    )
    e.is_at_or_wake_command = bool(text)
    e.set_extra(
        "provider_request",
        types.SimpleNamespace(conversation=types.SimpleNamespace(cid="conversation")),
    )
    return e


def png(color):
    b = io.BytesIO()
    PILImage.new("RGB", (40, 30), color).save(b, format="PNG")
    return b.getvalue()


async def run():
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory() as tmp:
        provider = types.SimpleNamespace(
            provider_config={"modalities": ["text", "tool_use"]}
        )
        context = types.SimpleNamespace(
            get_config=lambda **kw: {"wake_prefix": []},
            get_using_provider=lambda **kw: provider,
        )
        cp = ca.Main(context, {"image_cache_dir": tmp})
        context.get_registered_star = lambda name: (
            types.SimpleNamespace(star_cls=cp)
            if name == "astrbot_plugin_context_aware"
            else None
        )
        gp = object.__new__(gi.GiteeAIImagePlugin)
        gp.context = context
        gp._last_image_by_user = {}
        gp._last_image_task_meta_cache = {}
        gp.config = {
            "features": {"selfie": {"enabled": True, "llm_tool_enabled": True}}
        }
        manager = gi.BackgroundImageTaskManager(
            Path(tmp) / "jobs", heartbeat_seconds=60
        )
        await manager.start()
        try:
            source = event(
                "cat",
                components=[
                    Image(file="base64://" + base64.b64encode(png("orange")).decode())
                ],
            )
            stage = PreProcessStage()
            stage.config = {}
            stage.platform_settings = {}
            stage.stt_settings = {}
            await stage.process(source)
            await cp.on_message(source)
            owned = list(source._temporary_local_files)
            source.cleanup_temporary_local_files()
            assert all(not Path(p).exists() for p in owned)
            q = event("edit", "bob", "把Alice刚才的图背景改成雪地")
            await cp.on_message(q)
            tool = llm_tools.get_func("aiimg_generate")
            assert (
                tool.parameters["properties"]["reference_image_ids"]["items"]["type"]
                == "string"
            )
            req = ProviderRequest(prompt=q.message_str, func_tool=ToolSet([tool]))
            await cp.on_llm_request(q, req)
            catalog = cp.get_image_catalog(q)
            assert len(catalog) == 1
            cid = catalog[0]["id"]
            assert "reference_image_ids" in str(
                [p.text for p in req.extra_user_content_parts]
            )
            selection = await gp._prepare_context_reference_request(
                q, [cid], ["subject"]
            )
            raw = selection.images[0]
            assert raw[:2] == b"\xff\xd8"  # Core normalized JPEG; not preview.
            print(
                "PASS actual Core cleanup -> text-only model catalog -> cross-user same-group byte handoff"
            )
            output = Path(tmp) / "generated.png"
            output.write_bytes(png("blue"))
            captured = []

            async def edit(**kwargs):
                captured.append(kwargs)
                return output

            gp.edit = types.SimpleNamespace(edit=edit)
            paths, manifest = await manager.spool_inputs(
                "bridge-task", list(selection.images)
            )
            job = gi.PreparedImageJob(
                mode="edit",
                user_prompt="snow",
                effective_prompt="snow",
                backend=None,
                output={"aspect_ratio": "4:3", "resolution": "4K"},
                input_paths=paths,
                task_meta={
                    "mode": "edit",
                    "reference_sources": list(selection.sources),
                },
                options={"input_manifest": manifest},
            )
            # The worker remains independent of source availability.
            resource = cp._image_index.resources[cid]
            resource.data = b""
            result, meta = await gp._execute_prepared_image_job(manager, job)
            assert captured[-1]["images"] == [raw]
            resource.data = raw
            print(
                "PASS actual Gitee task spool -> worker preserves selected bytes and output intent"
            )
            gp._send_image_with_fallback = AsyncMock(
                return_value=gi.SendImageResult(True)
            )
            gp._save_last_image_task_meta = AsyncMock()
            gi.mark_success = AsyncMock()
            answer = await gp._finalize_llm_tool_image(q, result, task_meta=meta)
            rid = meta["result_image_id"]
            assert rid in answer.content[0].text
            next_q = event("continue", "bob", "把刚生成的背景换成海边，其他不动")
            await cp.on_message(next_q)
            next_rows = cp.get_image_catalog(next_q)
            row = next(x for x in next_rows if x["id"] == rid)
            assert row["is_requester_latest_result"] and row["parent_ids"] == [cid]
            next_selection = await gp._prepare_context_reference_request(
                next_q, [rid], ["subject"]
            )
            assert next_selection.images[0] == png("blue")
            print(
                "PASS delivered result registration -> requester follow-up edits exact generated image"
            )
            selfie = event("selfie", "bob", "抱着Alice那只猫拍照")
            await cp.on_message(selfie)
            await gp._prepare_context_reference_request(selfie, [cid], ["object"])
            gp._get_selfie_reference_paths = AsyncMock(
                return_value=([Path("face")], "webui")
            )
            gp._read_paths_bytes = AsyncMock(return_value=[png("red")])
            gp._get_life_context_without_llm = AsyncMock(return_value={})
            gp._get_selfie_default_output = lambda: ""
            images, prompt, options, meta = await gp._prepare_background_selfie(
                selfie, "抱着猫拍照", None
            )
            assert images == [png("red"), raw]
            assert "参考图 2" in prompt and "动物或物体" in prompt
            assert (
                options["reference_count"] == 1
                and options["extra_reference_count"] == 1
            )
            print(
                "PASS actual selfie preparation keeps identity + referenced cat with explicit roles"
            )
            token = selfie.get_extra(gi.RECEIPT_KEY)
            cp._image_index.clear(selfie.unified_msg_origin)
            late = {}
            await gp._publish_context_result(
                selfie, output, late, receipt=token, task_id="late"
            )
            assert "result_image_id" not in late
            print("PASS reset revokes old task result registration")
        finally:
            await manager.close()
            await cp.terminate()


asyncio.run(run())
