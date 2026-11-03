import numpy as np
import pytest

from flybrain_drive.ac_expert_handoff_labels import handoff_applied_targets


def records():
    return [dict(index=i, decision_monotonic_seconds=1+i*.04,
                 native_packet_id=i+1, expert=i >= 2, bridge=i == 1) for i in range(5)]


def native():
    return [dict(packet_id=i+1, received_monotonic_seconds=1.005+i*.04,
                 fields=dict(steerAngle=.12, gas=.2, brake=0)) for i in range(8)]


def test_prefix_and_waiting_inside_ai_window_still_have_no_expert_labels():
    targets = handoff_applied_targets(records(), native(), native_gain=1, ai_window=(.9, 2))
    assert not targets['action_valid'][:2].any()
    assert targets['action_valid'][2:].all()
    assert np.isnan(targets['raw_canonical'][:2]).all()
    assert (targets['action'][:2] == 0).all()
    assert (targets['label_packet_id'][:2] == -1).all()
    assert (targets['label_packet_id'][2:] > [3, 4, 5]).all()


def test_conservative_ai_end_masks_future_labels_without_cutting_history():
    targets = handoff_applied_targets(records(), native(), native_gain=1, ai_window=(1.07, 1.17))
    assert targets['action_valid'].shape == (5, 3)
    assert targets['action_valid'][2:4].all()
    assert not targets['action_valid'][4].any()


def test_gaps_cannot_be_hidden_by_resetting_at_handoff():
    rows = records()
    for row in rows[2:]:
        row['decision_monotonic_seconds'] += .1
    with pytest.raises(ValueError, match='one episode'):
        handoff_applied_targets(rows, native(), native_gain=1, ai_window=(.9, 2))


def test_missing_expert_receipts_are_not_replaced_with_student_predictions():
    with pytest.raises(ValueError, match='No actual future'):
        handoff_applied_targets(records(), native(), native_gain=1, ai_window=(1.3, 2))


def test_interleaved_roles_cannot_reset_the_actor_history():
    rows = records()
    rows[3]['expert'] = False
    with pytest.raises(ValueError, match='one original'.capitalize()):
        handoff_applied_targets(rows, native(), native_gain=1, ai_window=(.9, 2))
