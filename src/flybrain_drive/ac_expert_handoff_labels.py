"""Apply expert receipts only to acknowledged AI rows, retaining the complete actor history."""

import numpy as np

from .ac_engine_ai_labels import match_future_applied_controls, observation_segments


def handoff_applied_targets(records, native, *, native_gain, ai_window):
    """Student and waiting rows initialize recurrent state but never supply targets."""
    if [row['index'] for row in records] != list(range(len(records))):
        raise ValueError('Contiguous original handoff observation indexes required')
    times = [row['decision_monotonic_seconds'] for row in records]
    if observation_segments(times) != [(0, len(records))]:
        raise ValueError('Handoff history must remain one episode with no gaps over80ms')
    expert_indexes = [i for i, row in enumerate(records) if row['expert']]
    if (not expert_indexes or expert_indexes[0] == 0
            or expert_indexes != list(range(expert_indexes[0], len(records)))
            or any(row['bridge'] for row in records if row['expert'])):
        raise ValueError('One original student/waiting prefix '
                         'and acknowledged expert suffix required')
    events = [dict(decision_monotonic_seconds=records[i]['decision_monotonic_seconds'],
                   native=dict(packet_id=records[i]['native_packet_id'])) for i in expert_indexes]
    selected = match_future_applied_controls(events, native, native_gain=native_gain,
                                             ai_window=ai_window)
    result = {}
    for key, value in selected.items():
        fill = False if value.dtype == np.bool_ else (
            -1 if np.issubdtype(value.dtype, np.integer) else (
                0 if key == 'action' else np.nan))
        full = np.full((len(records), *value.shape[1:]), fill, dtype=value.dtype)
        full[expert_indexes] = value
        result[key] = full
    if not result['action_valid'].any():
        raise ValueError('No actual future expert receipts survived the timing and value masks')
    return result
