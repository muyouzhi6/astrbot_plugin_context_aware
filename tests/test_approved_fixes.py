import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from test_v320_changes import load_plugin_module
from video_input import is_media_rejection


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['', ' ', '\n\t  ', '\u3000\n'])
async def test_blank_summary_preserves_history_and_previous_summary(text):
    mod = load_plugin_module()
    class Provider(mod.Provider):
        text_chat = AsyncMock(return_value=SimpleNamespace(completion_text=text))
    plugin = mod.Main(SimpleNamespace(get_using_provider=lambda **kw: Provider()), {
        'history_compress_strategy': 'llm_summary',
        'history_compress_trigger_count': 10, 'history_compress_keep_recent': 5,
    })
    try:
        for i in range(12):
            await plugin._sessions.add_message_async('room', mod.MessageRecord(
                msg_id=str(i), sender_id='u', sender_name='User', content='history', timestamp=i,
            ))
        state = plugin._sessions.get('room')
        state.summary = 'previous summary'
        state.summary_message_count = 7
        before = await plugin._sessions.get_snapshot_async('room')
        result = await plugin._maybe_compress_history('room', before)
        after = await plugin._sessions.get_snapshot_async('room')
        assert after.messages == before.messages == result.messages
        assert after.summary == result.summary == 'previous summary'
        assert after.summary_message_count == 7
        assert not state.compressing
    finally:
        await plugin.terminate()


@pytest.mark.parametrize('new_count', [2, 20])
def test_summary_preserves_unsummarized_tail_and_arrivals(new_count):
    mod = load_plugin_module()

    async def run():
        manager = mod.SessionManager(max_messages=100)
        records = [SimpleNamespace(msg_id='', is_bot=True) for _ in range(25)]
        for record in records:
            await manager.add_message_async('room', record)
        token, snapshot = await manager.mark_compressing_async('room')
        arrivals = [SimpleNamespace(msg_id='', is_bot=True) for _ in range(new_count)]
        for record in arrivals:
            await manager.add_message_async('room', record)
        result = await manager.set_summary_and_trim_async(
            'room', token=token, summarized_messages=snapshot.messages[:-5],
            summary='summary', keep_recent=5, summarized_count=20, updated_at=1,
        )
        assert [id(m) for m in result.messages] == [id(m) for m in records[-5:] + arrivals]
        await manager.clear_compressing_async('room', token)
        assert not token.compressing

    asyncio.run(run())

@pytest.mark.parametrize('evict', [False, True])
def test_invalidated_summary_cannot_recreate_or_clear_new_job(evict):
    mod = load_plugin_module()

    async def run():
        manager = mod.SessionManager(max_sessions=10)
        manager.get('room')
        token, snapshot = await manager.mark_compressing_async('room')
        if evict:
            for i in range(10):
                manager.get(str(i))
        else:
            await manager.remove_session_async('room')
        result = await manager.set_summary_and_trim_async(
            'room', token=token, summarized_messages=snapshot.messages,
            summary='stale', keep_recent=5, summarized_count=20, updated_at=1,
        )
        assert result is None
        assert not manager.has_session('room')
        current = manager.get('room')
        await manager.mark_compressing_async('room')
        await manager.clear_compressing_async('room', token)
        assert current.compressing
        assert not current.summary

    asyncio.run(run())

def test_event_analyzers_are_independent():
    mod = load_plugin_module()
    plugin = mod.Main(SimpleNamespace(get_config=lambda umo: {'wake_prefix': [umo]}), {})
    first = plugin._ensure_initialized(SimpleNamespace(get_self_id=lambda: 'a', unified_msg_origin='one'))
    second = plugin._ensure_initialized(SimpleNamespace(get_self_id=lambda: 'b', unified_msg_origin='two'))
    assert first.bot_id == 'a'
    assert second.bot_id == 'b'
    assert first is not second

@pytest.mark.parametrize('status', [401, 402, 403, 429, 500, 503])
def test_service_errors_do_not_retry_media(status):
    exc = RuntimeError('unsupported video')
    exc.status_code = status
    assert not is_media_rejection(exc)

def test_only_explicit_media_rejection_retries():
    assert is_media_rejection(ValueError('unsupported video format'))
    assert not is_media_rejection(ValueError('bad request'))
    assert not is_media_rejection(TimeoutError())

@pytest.mark.parametrize('status', [401, 402, 403, 429, 500, 503])
def test_gemini_sdk_service_errors_do_not_retry_media(status):
    from google.genai.errors import ClientError, ServerError

    error_type = ServerError if status >= 500 else ClientError
    exc = error_type(status, {'error': {'code': status, 'message': 'unsupported video'}})
    assert not is_media_rejection(exc)

@pytest.mark.asyncio
@pytest.mark.parametrize('trigger,keep,invalid', [(60, 5, True), (20, 50, True), (20, 16, False)])
async def test_summary_configuration_reachability_before_message_threshold(trigger, keep, invalid):
    mod = load_plugin_module()
    plugin = mod.Main(SimpleNamespace(), {
        'max_history': 50, 'history_compress_strategy': 'llm_summary',
        'history_compress_trigger_count': trigger, 'history_compress_keep_recent': keep,
    })
    snapshot = await plugin._sessions.get_snapshot_async('room')
    with patch.object(mod.logger, 'warning') as warning, patch.object(plugin, '_remember_status') as status:
        assert await plugin._maybe_compress_history('room', snapshot) is snapshot
        assert warning.called is invalid
        if invalid:
            status.assert_called_once_with('room', 'summary_unreachable')
        else:
            status.assert_not_called()
    await plugin.terminate()

@pytest.mark.parametrize("outcome", ["empty", "error", "timeout", "cancel"])
def test_summary_failure_after_reset_does_not_return_old_snapshot(outcome):
    mod = load_plugin_module()

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class Provider(mod.Provider):
            async def text_chat(self, **kwargs):
                entered.set()
                await release.wait()
                if outcome == "error":
                    raise ValueError("upstream")
                if outcome == "timeout":
                    raise TimeoutError()
                return SimpleNamespace(completion_text="")
        plugin = mod.Main(SimpleNamespace(get_using_provider=lambda **kw: Provider()), {
            "history_compress_strategy": "llm_summary",
            "history_compress_trigger_count": 10, "history_compress_keep_recent": 5,
        })
        for i in range(12):
            await plugin._sessions.add_message_async("room", mod.MessageRecord(
                msg_id=str(i), sender_id="u", sender_name="User", content="old", timestamp=i,
            ))
        snapshot = await plugin._sessions.get_snapshot_async("room")
        task = asyncio.create_task(plugin._maybe_compress_history("room", snapshot))
        await entered.wait()
        await plugin._sessions.remove_session_async("room")
        current = plugin._sessions.get("room")
        await plugin._sessions.mark_compressing_async("room")
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            result = await task
            assert not result.messages and not result.summary
        assert current.compressing
        assert not current.summary
        await plugin.terminate()

    asyncio.run(run())

@pytest.mark.asyncio
async def test_caption_provider_lookup_failure_is_safe():
    mod = load_plugin_module()
    plugin = mod.Main(SimpleNamespace(get_using_provider=AsyncMock()), {"image_caption": {"enable": True}})
    plugin._image_caption_enabled = True
    plugin._using_provider = AsyncMock(side_effect=ValueError("sensitive"))
    assert await plugin._get_image_caption("unused") is None
    await plugin.terminate()

@pytest.mark.asyncio
async def test_two_bots_interleave_during_caption(tmp_path):
    mod = load_plugin_module()
    entered, release = asyncio.Event(), asyncio.Event()
    plugin = mod.Main(SimpleNamespace(get_config=lambda umo: {"wake_prefix": [umo]}), {"image_caption": True, "image_caption_lazy": False, "image_cache_dir": str(tmp_path)})
    async def caption(url):
        entered.set()
        await release.wait()
        return "image"
    plugin._get_image_caption = caption
    def event(bot):
        return SimpleNamespace(
            get_self_id=lambda: bot, get_sender_id=lambda: bot, get_sender_name=lambda: bot,
            unified_msg_origin=bot, message_obj=SimpleNamespace(message_id=bot),
            get_messages=lambda: [mod.Image(url="https://example.invalid/p.png"), mod.At(qq=bot)],
            get_extra=lambda key, default=None: default, get_message_outline=lambda: "image",
        )
    first = asyncio.create_task(plugin._extract_message_with_caption(event("a")))
    await entered.wait()
    second = asyncio.create_task(plugin._extract_message_with_caption(event("b")))
    await asyncio.sleep(0)
    release.set()
    a, b = await asyncio.gather(first, second)
    assert a.is_bot and b.is_bot and a.at_bot and b.at_bot
    await plugin.terminate()

@pytest.mark.asyncio
async def test_status_denies_non_admin_without_model_lookup():
    mod = load_plugin_module()
    plugin = mod.Main(SimpleNamespace(), {})
    event = SimpleNamespace(stop_event=lambda: None, is_admin=lambda: False, plain_result=lambda x: x)
    plugin._media_capability = AsyncMock(side_effect=AssertionError("must not query"))
    assert [part async for part in plugin.show_context_status(event)] == ["此命令需要管理员权限。"]
    await plugin.terminate()

@pytest.mark.asyncio
async def test_caption_cache_separates_models_and_does_not_log_text(tmp_path):
    mod = load_plugin_module()
    class Provider(mod.Provider):
        provider_config = {"id": "caption", "model": "one"}
        text_chat = AsyncMock(return_value=SimpleNamespace(completion_text="sensitive-caption"))
    provider = Provider()
    plugin = mod.Main(SimpleNamespace(get_using_provider=lambda: provider), {"image_caption": True, "image_cache_dir": str(tmp_path)})
    plugin._materialize_image_for_compression = AsyncMock(return_value=None)
    with patch.object(mod.logger, "info") as info:
        assert await plugin._get_image_caption("image") == "sensitive-caption"
        assert await plugin._get_image_caption("image") == "sensitive-caption"
        assert provider.text_chat.await_count == 1
        provider.provider_config = {"id": "caption", "model": "two"}
        await plugin._get_image_caption("image")
        assert provider.text_chat.await_count == 2
        assert "sensitive-caption" not in str(info.call_args_list)
    await plugin.terminate()
