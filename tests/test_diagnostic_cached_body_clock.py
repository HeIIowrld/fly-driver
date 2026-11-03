import copy
import pytest
from flybrain_drive.diagnostic_cached_body_clock import summarize_cached_body_around_render


def frame():
    def pair(first):
        return {'source_clock_before': first, 'source_clock_after': first + .001,
                'sim_time_ms': 100, 'car_timestamp_ms': 99,
                'body_diagnostic': {'speed_kmh': 12., 'local_angular_velocity': [0., 1., 2.],
                                    'acceleration_g': [.1, .2, .3]}}
    return {'training_admissible': False, 'actor_stream_created': False,
            'views': [{'view_id': name, 'diagnostic_clock_pair_before': pair(1.),
                       'diagnostic_clock_pair_after': pair(1.03), 'render_started_clock': 1.01,
                       'render_call_finished_clock': 1.02} for name in ('front', 'left', 'right', 'rear')]}


def test_cached_values_require_original_pairs_and_remain_diagnostic():
    data = frame()
    data['views'][0]['diagnostic_clock_pair_after']['body_diagnostic']['speed_kmh'] = 12.5
    result = summarize_cached_body_around_render(data)
    assert result['views'][0]['speed_difference_kmh'] == .5
    assert not result['native_packet_identity_verified']
    assert not result['physical_exposure_sync_verified']
    assert not result['training_admissible']
    assert data['views'][0]['diagnostic_clock_pair_before']['body_diagnostic']['speed_kmh'] == 12.


def test_invalid_samples_and_bad_render_order_are_rejected():
    for mutation in ('missing', 'nan', 'order', 'admission'):
        data = copy.deepcopy(frame())
        if mutation == 'missing': del data['views'][0]['diagnostic_clock_pair_before']['body_diagnostic']
        if mutation == 'nan': data['views'][0]['diagnostic_clock_pair_after']['body_diagnostic']['acceleration_g'][0] = float('nan')
        if mutation == 'order': data['views'][0]['render_started_clock'] = .5
        if mutation == 'admission': data['training_admissible'] = True
        with pytest.raises(ValueError): summarize_cached_body_around_render(data)
