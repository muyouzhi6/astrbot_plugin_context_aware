"""Versioned inter-plugin image handoff; never exposed as a model file API."""

from __future__ import annotations

import asyncio
import hashlib
import time
from types import SimpleNamespace

try:
    from .image_context import EPOCH_KEY, SNAPSHOT_KEY, validate_image
except ImportError:
    from image_context import EPOCH_KEY, SNAPSHOT_KEY, validate_image


class ImageReferenceAPI:
    image_reference_api_version = 1

    def capture_image_request(self, event, conversation_id=""):
        """Capture an opaque task receipt before async work starts.

        A receipt is valid only in this plugin instance and session epoch.
        User-visible image IDs do not carry authority across groups/resets.
        """
        if not self._recall_enabled or not self._should_process(event):
            return None
        sequence = self._stamp_image_event(event)
        session = event.unified_msg_origin
        epoch = event.get_extra(EPOCH_KEY)
        if not self._image_index.is_current(session, epoch):
            return None
        return {
            "owner": self,
            "epoch": epoch,
            "session": session,
            "request_sequence": sequence,
            "sender_id": str(event.get_sender_id()),
            "sender_name": str(event.get_sender_name() or event.get_sender_id()),
            "message_id": str(getattr(event.message_obj, "message_id", "")),
            "conversation_id": str(conversation_id),
            "allowed_parent_ids": tuple(
                row["id"] for row in self.get_image_catalog(event)
            ),
        }

    def get_image_catalog(self, event):
        if not self._recall_enabled or not self._should_process(event):
            return []
        sequence = self._stamp_image_event(event)
        session = event.unified_msg_origin
        if not self._image_index.is_current(session, event.get_extra(EPOCH_KEY)):
            return []
        ids = event.get_extra(SNAPSHOT_KEY)
        if ids is None:
            # Include current attachments for mixed "this clothing + old cat"
            # requests. Automatic vision still preserves native attachments.
            ids = self._image_index.snapshot(session, sequence + 1)
            event.set_extra(SNAPSHOT_KEY, ids)
        resources = [self._image_index.get(session, i) for i in ids]
        entries = [r.entry for r in resources if r is not None]
        req = event.get_extra("provider_request")
        current_cid = str(getattr(getattr(req, "conversation", None), "cid", "") or "")
        latest = {}
        for e in entries:
            if e.kind == "generated":
                key = (e.sender_id, e.conversation_id)
                if (
                    key not in latest
                    or e.request_sequence > latest[key].request_sequence
                ):
                    latest[key] = e
        return [
            {
                "id": e.image_id,
                "sender": e.sender_name[:60],
                "sender_id": e.sender_id,
                "message_id": e.message_id,
                "position": f"{e.ordinal}/{e.count}",
                "seconds_ago": max(0, int(time.time() - e.timestamp)),
                "kind": e.kind,
                "task_id": e.task_id,
                "conversation_id": e.conversation_id,
                "parent_ids": list(e.parent_ids),
                "is_current_attachment": e.sequence == sequence and e.kind == "message",
                "is_requester_latest_result": e.kind == "generated"
                and e.sender_id == str(event.get_sender_id())
                and (not current_cid or e.conversation_id == current_cid)
                and latest.get((e.sender_id, e.conversation_id)) is e,
                "quality": "retained_input; may already be normalized by Core",
            }
            for e in entries
        ]

    async def resolve_reference_images(self, event, image_ids):
        """Export an all-or-nothing set of current-scope images for editing."""
        if not isinstance(image_ids, list) or not 1 <= len(image_ids) <= 8:
            raise ValueError("Select 1 to 8 image IDs from this request's catalog")
        catalog = {row["id"]: row for row in self.get_image_catalog(event)}
        if any(not isinstance(i, str) or i not in catalog for i in image_ids):
            raise ValueError(
                "Reference image is not in this request's session snapshot"
            )
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("Duplicate reference IDs are not allowed")
        session, epoch = event.unified_msg_origin, event.get_extra(EPOCH_KEY)
        result, total = [], 0
        for image_id in image_ids:
            data = await self._image_index.read_bytes(session, image_id)
            if data is None or len(data) > 20 * 1024 * 1024:
                raise ValueError(
                    "A selected reference is unavailable or exceeds 20 MiB; do not omit it"
                )
            total += len(data)
            if total > 64 * 1024 * 1024:
                raise ValueError("Selected references exceed 64 MiB")
            result.append(
                {
                    **catalog[image_id],
                    "data": data,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        if not self._image_index.is_current(session, epoch) or any(
            self._image_index.get(session, i) is None for i in image_ids
        ):
            raise ValueError(
                "Image context was cleared or expired during reference handoff"
            )
        return result

    async def register_generated_image(self, receipt, *, task_id, data, parent_ids=()):
        """Register one successfully delivered image using a captured receipt."""
        if not isinstance(receipt, dict) or receipt.get("owner") is not self:
            return None
        session, epoch = receipt["session"], receipt["epoch"]
        if not self._recall_enabled or not self._image_index.is_current(session, epoch):
            return None
        if (
            not isinstance(data, bytes)
            or not data
            or len(data) > self._image_index.max_download
        ):
            return None
        if not isinstance(task_id, str) or not task_id or len(task_id) > 200:
            return None
        if (
            not isinstance(parent_ids, (list, tuple))
            or len(parent_ids) > 8
            or any(
                not isinstance(i, str) or i not in receipt.get("allowed_parent_ids", ())
                for i in parent_ids
            )
        ):
            return None
        try:
            await asyncio.wait_for(asyncio.to_thread(validate_image, data), timeout=3)
        except (asyncio.TimeoutError, ValueError, OSError):
            return None
        if not self._image_index.is_current(session, epoch):
            return None
        self._recall_sequence += 1
        message = SimpleNamespace(
            msg_id=f"gitee-result:{task_id}",
            sender_id=receipt["sender_id"],
            sender_name=receipt["sender_name"],
            timestamp=time.time(),
            is_bot=False,
            has_gif=False,
            image_urls=["generated"],
            image_count=1,
        )
        return self._image_index.add_generated(
            message,
            session,
            self._recall_sequence,
            data,
            task_id=str(task_id),
            request_sequence=receipt["request_sequence"],
            conversation_id=receipt["conversation_id"],
            parent_ids=parent_ids,
        )
