import json

import numpy as np
import pytest

from flybrain_drive.flight_fast_integrity import verify_trial_evidence
from flybrain_drive.graph import sha256


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def evidence(tmp_path):
    config_file = tmp_path/'config.json'
    write(config_file, {})
    original_root = tmp_path/'original'
    original_root.mkdir()
    sources, trials = [], []
    for index in (0, 3, 5, 6, 8, 11):
        source = tmp_path/f'source-{index}'
        source.mkdir()
        clock = index*100+np.arange(257)*.04
        np.savez(source/'fast-hypotheses.npz', original_clock=clock)
        write(source/'result.json', dict(fast_original_replay=dict(
            archive_sha256=sha256(source/'fast-hypotheses.npz'))))
        source_sha = sha256(source/'result.json')
        sources.append(dict(index=index, path=str(source), result_sha256=source_sha))
        for mode in ('full', 'zero_yaw'):
            root = tmp_path/f'trial-{index}-{mode}'
            root.mkdir()
            np.savez(root/'outputs.npz', intents=np.zeros((257, 2)), original_clock=clock)
            write(root/'result.json', dict(status='completed_fast_stream_trial_replay',
                original_observations=257, mode=mode, trial_index=index,
                configured_parameters_unchanged=True, fine_reference_persistent=True,
                fine_reference_hidden_resets=0, hidden_actual_resets=0,
                real_time_budget_passed=True,
                actuator_messages_sent=0, maximum_intent_error=1e-7,
                maximum_exact_neuron_error=1e-7,
                maximum_fine_neuron_error=1e-5, wall_seconds_percentiles=[.02, .03, .04],
                archive_sha256=sha256(root/'outputs.npz')))
            write(root/'manifest.json', dict(config=dict(fast_config_sha256=sha256(config_file),
                                                        source_audit_result_sha256=source_sha)))
            write(root/'scratch-warmup.json', dict(actual_neural_state_unchanged=True,
                observations_created=0, actual_prefix_consumed=0, actual_counters=0))
            trials.append(dict(index=index, mode=mode, path=str(root),
                               result_sha256=sha256(root/'result.json')))
    write(original_root/'result.json', dict(trials=sources))
    config = dict(original_history_verification=str(original_root), fine_neuron_tolerance=.001)
    return config, config_file, dict(trials=trials)


def test_metadata_counts_cannot_replace_complete_unique_child_evidence(evidence):
    config, path, receipt = evidence
    verify_trial_evidence(config, path, receipt)
    for trials in ([], receipt['trials'][:-1], [receipt['trials'][0]]*12):
        with pytest.raises(ValueError, match='six unique'):
            verify_trial_evidence(config, path, dict(trials=trials))


@pytest.mark.parametrize('value', [float('nan'), -1., .01])
def test_invalid_neuron_error_rejected_even_when_receipt_hash_matches(evidence, value):
    from pathlib import Path

    config, path, receipt = evidence
    first = receipt['trials'][0]
    child = Path(first['path'])/'result.json'
    data = json.loads(child.read_text(encoding='utf8'))
    data['maximum_fine_neuron_error'] = value
    write(child, data)
    first['result_sha256'] = sha256(child)
    with pytest.raises(ValueError, match='numerical'):
        verify_trial_evidence(config, path, receipt)


def test_original_clock_change_and_warmup_consumption_are_not_admitted(evidence):
    from pathlib import Path

    config, path, receipt = evidence
    root = Path(receipt['trials'][0]['path'])
    write(root/'scratch-warmup.json', dict(actual_neural_state_unchanged=True,
        observations_created=1, actual_prefix_consumed=1, actual_counters=1))
    with pytest.raises(ValueError, match='scratch warmup'):
        verify_trial_evidence(config, path, receipt)
    write(root/'scratch-warmup.json', dict(actual_neural_state_unchanged=True,
        observations_created=0, actual_prefix_consumed=0, actual_counters=0))
    np.savez(root/'outputs.npz', intents=np.zeros((257, 2)),
             original_clock=np.arange(257)*.04+.001)
    child = json.loads((root/'result.json').read_text(encoding='utf8'))
    child['archive_sha256'] = sha256(root/'outputs.npz')
    write(root/'result.json', child)
    receipt['trials'][0]['result_sha256'] = sha256(root/'result.json')
    with pytest.raises(ValueError, match='identical original clocks'):
        verify_trial_evidence(config, path, receipt)
