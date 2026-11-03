"""Synthetic confidence calibration checks; not evidence of better driving."""

import copy
import json
import sys

import numpy as np
import pytest
import torch
from test_multiview import maps, model, observation, packer, sample
from test_native_action_training_pipeline import plain

from flybrain_drive.graph import sha256
from flybrain_drive.policy_anchor import prepare_policy_anchors, replay_anchor_mse
from flybrain_drive.visual_age_calibration import calibrate_visual_age_decay


def fixture():
    runtime = packer()
    net = model(runtime)
    packed = runtime.pack(observation(0, 0, [sample(v) for v in runtime.views]))
    values = {k: torch.tensor(v)[None] for k, v in packed.items()}
    values['view_age'] = torch.tensor([[.02, .04, .06, .08]])
    episode = dict(source={'sha256': 'a'*64}, observation=copy.deepcopy(values),
        action=torch.zeros(1, 3), action_valid=torch.ones(1, 3, dtype=torch.bool))
    return runtime, net, values, episode


def test_default_is_bit_exact_and_soft_confidence_never_changes_freshness():
    runtime, net, values, _ = fixture()
    hard_age = runtime.calibration['max_view_age']
    explicit = model(runtime, visual_age_scale=hard_age)
    explicit.load_state_dict(net.state_dict())
    assert set(explicit.state_dict()) == set(net.state_dict())
    assert torch.equal(explicit.prepare_drive(values), net.prepare_drive(values))
    changed = model(runtime, visual_age_scale=.6)
    changed.load_state_dict(net.state_dict())
    values['view_age'][0, 0] = hard_age+.001
    with pytest.raises(ValueError, match='Stale'):
        changed.prepare_drive(values)
    assert changed.sensory_calibration['max_view_age'] == hard_age


def test_mean_operating_point_preserved_and_constructor_roundtrip_keeps_it():
    runtime, net, values, episode = fixture()
    before = net.prepare_drive(values).detach().clone()
    original = copy.deepcopy(net.state_dict())
    inputs = copy.deepcopy(episode['observation'])
    report = calibrate_visual_age_decay(net, [episode], .6)
    torch.testing.assert_close(net.prepare_drive(values), before, rtol=2e-6, atol=1e-9)
    for key, value in original.items():
        if key != 'adapter.visual_adapter.weight':
            torch.testing.assert_close(net.state_dict()[key], value, rtol=0, atol=0)
    for key, value in inputs.items():
        assert torch.equal(episode['observation'][key], value)
    rebuilt = model(runtime, visual_age_scale=report['scale_seconds'])
    rebuilt.load_state_dict(net.state_dict())
    assert torch.equal(rebuilt.prepare_drive(values), net.prepare_drive(values))
    assert report['valid_observations_per_view'] == [1]*4
    assert not report['input_ages_or_masks_modified']


def test_missing_source_and_stale_valid_calibration_reject_without_mutation():
    _, net, _, episode = fixture()
    original = net.adapter.visual_adapter.weight.detach().clone()
    episode['observation']['view_valid'][:, 1] = 0
    with pytest.raises(ValueError, match='Each source'):
        calibrate_visual_age_decay(net, [episode], .6)
    episode['observation']['view_valid'][:, 1] = 1
    episode['observation']['view_age'][:, 1] = .21
    with pytest.raises(ValueError, match='stale'):
        calibrate_visual_age_decay(net, [episode], .6)
    assert torch.equal(original, net.adapter.visual_adapter.weight)
    assert net.adapter.visual_adapter.age_scale == .2


@pytest.mark.parametrize('scale', [True, float('nan'), .1, 1.01])
def test_invalid_confidence_scales_reject(scale):
    runtime, net, _, episode = fixture()
    with pytest.raises(ValueError, match='scale'):
        calibrate_visual_age_decay(net, [episode], scale)
    with pytest.raises(ValueError, match='scale'):
        model(runtime, visual_age_scale=scale)


def test_original_policy_targets_remain_frozen_before_confidence_change():
    _, net, _, episode = fixture()
    episode['observation'] = {k: v.repeat(3, *([1]*(v.ndim-1)))
                              for k, v in episode['observation'].items()}
    episode['observation']['view_age'][0] *= .5
    episode['observation']['view_age'][2] *= 1.5
    episode.update(action=torch.zeros(3, 3), action_valid=torch.ones(3, 3, dtype=torch.bool))
    targets = prepare_policy_anchors(net, [episode], ['a'*64], 2, 4)
    original = targets['a'*64].clone()
    calibrate_visual_age_decay(net, [episode], .6)
    assert torch.equal(targets['a'*64], original)
    assert replay_anchor_mse(net, [episode], targets, 2, 4) > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_training_freezes_original_reference_before_age_change_and_saves_constructor(
        tmp_path, monkeypatch, device):
    from scripts.train_multiview import main
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    runtime, _, _, episode = fixture()
    net = model(runtime, pedal_mode='independent', organ_validity_mode='gate_only',
                readout_gain_parameterization='log')
    episode['observation'] = {k: v.repeat(3, *([1]*(v.ndim-1)))
                              for k, v in episode['observation'].items()}
    episode['observation']['view_age'][0] *= .5
    episode['observation']['view_age'][2] *= 1.5
    episode.update(action=torch.zeros(3, 3), action_valid=torch.ones(3, 3, dtype=torch.bool))
    graph = dict(ids=list(map(str, range(10))), types=['synthetic']*10,
        roles=['input']*5+['internal']+['output']*4, pre=[0, 1, 2, 3, 4, 5, 5, 5, 5],
        post=[5]*5+[6, 7, 8, 9], weights=[1.]*5+[2., 1., 1., 1.])
    constructor = dict(visual_ids=['0', '1', '2', '3'],
        organ_groups={'body': {'body_ids': ['4'], 'feature_indices': [0]}},
        directions=np.stack([v.directions_body for v in maps()]), available=np.ones((4, 4)),
        port_sources=[0, 1, 2, 3], signs=[1]*10,
        incoming_totals=np.bincount(graph['post'], weights=graph['weights'], minlength=10),
        output_groups={'steering': ['6', '7'], 'forward': ['8'], 'halt': ['9']},
        steering_signs=[-1, 1], sensory_calibration=runtime.calibration, neighbors=2,
        pedal_mode='independent', organ_validity_mode='gate_only',
        readout_gain_parameterization='log')
    source = tmp_path/'source.pt'
    torch.save(dict(graph=graph, constructor=plain(constructor), model=net.state_dict()), source)
    teacher = tmp_path/'teacher'
    teacher.mkdir()
    data = teacher/'episode.npz'
    np.savez(data, **{'obs_'+k: v.numpy() for k, v in episode['observation'].items()},
        action=episode['action'].numpy(), action_valid=episode['action_valid'].numpy(),
        observation_time=np.arange(3)*.04)
    episode['source']['sha256'] = sha256(data)
    original_targets = prepare_policy_anchors(net, [episode], [sha256(data)], 2, 4)
    manifest = dict(schema='flybrain.multiview.teacher.v1', action_profile='normalized_ac_absolute',
        sensory_calibration=runtime.calibration, actor_fields=sorted(episode['observation']),
        decisions=3, decision_dt_seconds=.04, brain_ticks_per_decision=4,
        episodes=[dict(episode_id='synthetic', file=data.name, sha256=sha256(data), steps=3)])
    (teacher/'dataset.json').write_text(json.dumps(manifest))
    config = dict(source_checkpoint=str(source), source_checkpoint_sha256=sha256(source),
        seed=862953, epochs=1, chunk_length=2, brain_ticks_per_decision=4,
        control_dt_seconds=.04, action_profile='normalized_ac_absolute',
        learning_rate=.0001, trainable_groups=['visual'], device=device,
        pedal_mode='independent', organ_validity_mode='gate_only',
        readout_gain_parameterization='log', condition_steering_gain=False,
        visual_age_confidence={'scale_seconds': .6},
        policy_anchor=dict(checkpoint_sha256=sha256(source), episode_sha256=[sha256(data)],
                           coefficient=3.))
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path), '--teacher', str(teacher),
                                     '--output', str(output)])
    main()
    report = json.loads((output/'visual-age-calibration.json').read_text())
    assert report['old_scale_seconds'] == .2 and report['hard_freshness_seconds'] == .2
    saved = np.load(output/'policy-anchor-targets.npz')
    np.testing.assert_allclose(saved[sha256(data)], original_targets[sha256(data)].numpy(),
                               atol=1e-7, rtol=1e-5)
    checkpoint = torch.load(output/'checkpoint.pt', weights_only=True)
    assert checkpoint['constructor']['visual_age_scale'] == .6
    assert checkpoint['constructor']['sensory_calibration']['max_view_age'] == .2
    assert checkpoint['graph'] == graph and sha256(data) == manifest['episodes'][0]['sha256']
    assert json.loads((output/'policy-anchor.json').read_text())['source_initial_anchor_mse'] == 0
    if device == 'cuda':
        torch.cuda.synchronize()
