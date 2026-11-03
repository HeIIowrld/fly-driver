"""Validate concrete complete fast-stream child evidence, never just aggregate success flags."""

import json
import math
from pathlib import Path

import numpy as np

from .episode_storage import load_array
from .graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_trial_evidence(config, config_file, receipt):
    sources = read(Path(config['original_history_verification'])/'result.json')['trials']
    by_index = {entry['index']: entry for entry in sources}
    expected = {(index, mode) for index in by_index for mode in ('full', 'zero_yaw')}
    trials = receipt.get('trials', [])
    pairs = [(entry['index'], entry['mode']) for entry in trials]
    if len(sources) != 6 or len(by_index) != 6 or len(pairs) != 12 or set(pairs) != expected:
        raise ValueError('All six unique original histories and both modes required')
    for entry in trials:
        root = Path(entry['path'])
        if sha256(root/'result.json') != entry['result_sha256']:
            raise ValueError('Fast-stream child receipt bytes changed')
        child, manifest = read(root/'result.json'), read(root/'manifest.json')
        if (child['status'] != 'completed_fast_stream_trial_replay'
                or child['original_observations'] != 257 or child['mode'] != entry['mode']
                or child['trial_index'] != entry['index']
                or not child['configured_parameters_unchanged']
                or not child['fine_reference_persistent']
                or child['fine_reference_hidden_resets'] != 0 or child['hidden_actual_resets'] != 0
                or not child['real_time_budget_passed']
                or child['actuator_messages_sent'] != 0):
            raise ValueError('Incomplete or invalid concrete fast-stream trial')
        for field, maximum in (('maximum_intent_error', 3e-6),
                               ('maximum_exact_neuron_error', 3e-6),
                               ('maximum_fine_neuron_error', config['fine_neuron_tolerance'])):
            if not math.isfinite(child[field]) or not 0 <= child[field] <= maximum:
                raise ValueError('Invalid numerical fast-stream trial bound')
        percentiles = child['wall_seconds_percentiles']
        if (len(percentiles) != 3 or not all(math.isfinite(v) and v >= 0 for v in percentiles)
                or percentiles != sorted(percentiles)
                or percentiles[1] > .04 or percentiles[2] > .08):
            raise ValueError('Fast-stream processing deadline exceeded')
        source = Path(by_index[entry['index']]['path'])
        if (manifest['config']['fast_config_sha256'] != sha256(config_file)
                or manifest['config']['source_audit_result_sha256']
                != by_index[entry['index']]['result_sha256']):
            raise ValueError('Fast-stream child configuration/source mismatch')
        warm = read(root/'scratch-warmup.json')
        if (warm['actual_neural_state_unchanged'] is not True or warm['observations_created'] != 0
                or warm['actual_prefix_consumed'] != 0 or warm['actual_counters'] != 0):
            raise ValueError('Actual episode was consumed or modified by scratch warmup')
        if sha256(root/'outputs.npz') != child['archive_sha256']:
            raise ValueError('Fast-stream output archive changed')
        source_result = read(source/'result.json')
        if (sha256(source/'result.json') != by_index[entry['index']]['result_sha256']
                or sha256(source/'fast-hypotheses.npz')
                != source_result['fast_original_replay']['archive_sha256']):
            raise ValueError('Source history evidence changed')
        with (load_array(root/'outputs.npz', allow_pickle=False) as output,
              load_array(source/'fast-hypotheses.npz', allow_pickle=False) as original):
            if (output['intents'].shape != (257, 2) or not np.isfinite(output['intents']).all()
                    or not np.array_equal(output['original_clock'], original['original_clock'])):
                raise ValueError('Complete finite outputs on identical original clocks required')
