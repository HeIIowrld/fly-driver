"""Privileged native AI-controller flag for collection/evaluation labels, never actor input."""

import base64
import hashlib
import struct
import time

from .ac_image_buffer import ExistingNamedMapping

# Reviewed upstream SPageFilePhysics: isAIControlled at byte416, pack4.
AI_STATUS_PREFIX = struct.Struct('<i412xi')


class NativeTeacherStatusReader:
    def __init__(self):
        self.mapping = ExistingNamedMapping('Local\\acpmf_physics', AI_STATUS_PREFIX.size)

    def snapshot(self):
        for _ in range(8):
            first, second = self.mapping.read(), self.mapping.read()
            if first != second:
                continue
            packet, active = AI_STATUS_PREFIX.unpack(second)
            if packet < 0 or active not in (0, 1):
                raise ValueError('Teacher status ABI/reset differs')
            return dict(packet_id=packet, builtin_ai_controlled=bool(active),
                received_monotonic_seconds=time.monotonic(),
                prefix_base64=base64.b64encode(second).decode(),
                prefix_sha256=hashlib.sha256(second).hexdigest(), actor_input=False)
        raise RuntimeError('Teacher status changed during repeated-copy attempts')

    def close(self):
        self.mapping.close()
