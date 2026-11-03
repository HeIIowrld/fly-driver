"""Synthetic codec integration only: no AC frames, anatomy claim or driver access."""

import json
import sys

import numpy as np
import pytest
import torch
from test_multiview import maps, model, packer

from flybrain_drive.graph import sha256
from flybrain_drive.multiview_sensory import MultiViewPacker


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@pytest.mark.parametrize('validity_mode', ['learned_drive', 'gate_only'])
@pytest.mark.parametrize('gain_mode', ['softplus', 'log'])
@pytest.mark.parametrize('update_scope', ['chunk', 'epoch'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_codec_three_head_training_persists_constructor_and_keeps_recurrent_graph_fixed(
        tmp_path, monkeypatch, validity_mode, gain_mode, update_scope, device):
    from scripts.train_multiview import main
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA/ROCm device unavailable')

    runtime = packer()
    net = model(runtime)
    with torch.no_grad():
        net.adapter.visual_adapter.weight.fill_(.1)
        for layer in net.adapter.organ_adapters:
            layer.weight.fill_(.05)
    graph = dict(ids=list(map(str, range(10))), types=['synthetic'] * 10,
                 roles=['input'] * 5 + ['internal'] + ['output'] * 4,
                 pre=[0, 1, 2, 3, 4, 5, 5, 5, 5], post=[5] * 5 + [6, 7, 8, 9],
                 weights=[1.] * 5 + [2., 1., 1., 1.])
    constructor = dict(
        visual_ids=['0', '1', '2', '3'],
        organ_groups={'body': {'body_ids': ['4'], 'feature_indices': [0]}},
        directions=np.stack([view.directions_body for view in maps()]),
        available=np.ones((4, 4)), port_sources=[0, 1, 2, 3], signs=[1] * 10,
        incoming_totals=np.bincount(graph['post'], weights=graph['weights'], minlength=10),
        output_groups={'steering': ['6', '7'], 'forward': ['8'], 'halt': ['9']},
        steering_signs=[-1, 1], sensory_calibration=runtime.calibration, neighbors=2)
    source = tmp_path / 'synthetic-source.pt'
    torch.save(dict(graph=graph, constructor=plain(constructor), model=net.state_dict()), source)
    teacher = tmp_path / 'synthetic-teacher'
    teacher.mkdir()
    n = 8
    observation = dict(
        view_rgb=np.full((n, 4, 4, 3), .2, np.float32),
        view_contrast=np.zeros((n, 4, 4, 3), np.float32),
        view_valid=np.ones((n, 4, 4), np.float32),
        contrast_valid=np.zeros((n, 4, 4), np.float32),
        view_age=np.zeros((n, 4), np.float32),
        organs=np.full((n, 6), .2, np.float32), organ_valid=np.ones((n, 6), np.float32))
    episode = teacher / 'episode.npz'
    np.savez(episode, **{'obs_' + k: v for k, v in observation.items()},
             action=np.tile([.1, .4, .2], (n, 1)), action_valid=np.ones((n, 3), bool),
             observation_time=np.arange(n) * .02)
    manifest = dict(schema='flybrain.multiview.teacher.v1',
        source='synthetic integration fixture, not AC data',
        action_profile='normalized_ac_absolute', sensory_calibration=runtime.calibration,
        actor_fields=sorted(MultiViewPacker.fields), decisions=n, decision_dt_seconds=.02,
        brain_ticks_per_decision=2, episodes=[dict(episode_id='synthetic', file=episode.name,
                                                 sha256=sha256(episode), steps=n)])
    (teacher / 'dataset.json').write_text(json.dumps(manifest))
    config = dict(source_checkpoint=str(source), source_checkpoint_sha256=sha256(source),
        action_profile='normalized_ac_absolute', seed=851101, epochs=2, chunk_length=4,
        learning_rate=.001, brain_ticks_per_decision=2, control_dt_seconds=.02,
        trainable_groups=['visual', 'organs', 'readout'], pedal_mode='independent',
        organ_validity_mode=validity_mode,
        readout_gain_parameterization=gain_mode,
        pedal_conditioning=dict(target_rms=[.1, .1], max_gain=1e8),
        condition_steering_gain=False)
    config['optimizer_update_scope'] = update_scope
    config['device'] = device
    if gain_mode == 'log':
        config['readout_gain_learning_rate'] = .03
    surrogate = gain_mode == 'log' and validity_mode == 'gate_only'
    if surrogate:
        config.update(activation_gradient='leaky_surrogate', surrogate_negative_slope=.05)
    config_file = tmp_path / 'synthetic-config.json'
    config_file.write_text(json.dumps(config))
    output = tmp_path / 'synthetic-run'
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(config_file), '--teacher',
                                     str(teacher), '--output', str(output)])
    main()
    result = json.loads((output / 'result.json').read_text())
    assert result['status'] == 'completed'
    assert result['optimizer_updates'] == (4 if update_scope == 'chunk' else 2)
    assert result['optimizer_update_scope'] == update_scope
    assert result['training_device'] == device
    assert all(name.startswith(device) for name in result['optimizer_parameter_devices'])
    assert all(name.startswith(device) for name in result['gradient_devices'])
    assert result['after']['supervision_counts'] == [8, 8, 8]
    assert result['selection_score_after'] <= result['selection_score_before']
    assert not result['AC_ready'] and not result['goal_achieved']
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
    assert all(value.device.type == 'cpu' for value in checkpoint['model'].values())
    assert checkpoint['checkpoint_optimizer_updates'] == checkpoint['selected_epoch'] * (
        2 if update_scope == 'chunk' else 1)
    assert checkpoint['constructor']['pedal_mode'] == 'independent'
    assert checkpoint['constructor']['organ_validity_mode'] == validity_mode
    assert checkpoint['constructor']['readout_gain_parameterization'] == gain_mode
    assert checkpoint['constructor'].get('activation_gradient', 'exact') == (
        'leaky_surrogate' if surrogate else 'exact')
    assert checkpoint['trainable_groups'] == ['visual', 'organs', 'readout']
    assert all(not name.startswith('circuit.') for name in checkpoint['trainable_parameters'])
    assert checkpoint['graph'] == graph
    for name, original in net.state_dict().items():
        if name.startswith('circuit.'):
            updated = checkpoint['model'][name]
            if original.is_sparse:
                original, updated = original.to_dense(), updated.to_dense()
            torch.testing.assert_close(original, updated, rtol=0, atol=0)
    assert sha256(source) == config['source_checkpoint_sha256']
    if update_scope == 'epoch' and validity_mode == 'gate_only' and gain_mode == 'log':
        # Separate batches with distinct targets. Accumulated updates must
        # combine both losses before Adam, independent of which episode comes last.
        second = teacher / 'second.npz'
        np.savez(second, **{'obs_'+k: v for k, v in observation.items()},
                 action=np.tile([.3, .1, .6], (n, 1)), action_valid=np.ones((n, 3), bool),
                 observation_time=np.arange(n)*.02)
        manifest['episodes'].append(dict(episode_id='second', file=second.name,
                                         sha256=sha256(second), steps=n))
        manifest['decisions'] = 2*n
        config['epochs'] = 1
        config['episodes_per_batch'] = 1
        config_file.write_text(json.dumps(config))
        compared = []
        for order in ('forward', 'reverse'):
            (teacher / 'dataset.json').write_text(json.dumps(manifest))
            destination = tmp_path / order
            monkeypatch.setattr(sys, 'argv', ['train', '--config', str(config_file), '--teacher',
                                             str(teacher), '--output', str(destination)])
            main()
            compared.append(torch.load(destination / 'checkpoint.pt', weights_only=True))
            manifest['episodes'].reverse()
        assert compared[0]['optimizer_updates'] == compared[1]['optimizer_updates'] == 1
        for name, first in compared[0]['model'].items():
            second = compared[1]['model'][name]
            if first.is_sparse:
                first, second = first.to_dense(), second.to_dense()
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-7)
