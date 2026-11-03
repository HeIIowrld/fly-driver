import base64
import hashlib

import pytest

from scripts.audit_causal_copy_body_context import causal_packet_index, first_observed_packets
from tests.test_ac_teacher_physics import packet


def row(raw, start, end):
    return {'authority': {'prefix_base64': base64.b64encode(raw).decode(),
                          'prefix_sha256': hashlib.sha256(raw).hexdigest(),
                          'packet_id': int.from_bytes(raw[:4], 'little'),
                          'received_monotonic_seconds': end},
            'copy': {'first_copy_start': start, 'second_copy_end': end}}


def test_duplicate_packet_preserves_first_receipt_and_strict_render_order():
    rows = [row(packet(7), 1., 1.01), row(packet(7), 2., 2.01),
            row(packet(8), 3., 3.01)]
    packets = first_observed_packets(rows)
    assert [x['first_end'] for x in packets] == [1.01, 3.01]
    assert causal_packet_index([1.01, 3.01], 3.01) == 0
    assert causal_packet_index([1.01, 3.01], 3.02) == 1


def test_duplicate_packet_changed_bytes_is_not_redated_or_accepted():
    changed = bytearray(packet(7))
    changed[4] ^= 1
    with pytest.raises(ValueError, match='changed bytes'):
        first_observed_packets([row(packet(7), 1., 1.01), row(bytes(changed), 2., 2.01)])
