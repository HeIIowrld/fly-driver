"""Reconstruct the Miata public teacher on its real stored 75x50 windows.

This is offline candidate qualification, not a live lap or a student policy.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='assets/external/vectorq-miata-teacher-20261102')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root)
    provenance = json.loads((root/'provenance.json').read_text())
    for x in provenance:
        if sha256(x['path']) != x['sha256']:
            raise ValueError('Downloaded candidate hash differs')
    run = begin_run(a.output, dict(arguments=vars(a), seed=780102,
        network_source_sha256=sha256(root/'network.py'),
        network_source_url='https://github.com/chandankeelara/transformer-based-autonomous-racing-agent/blob/45d33fa3b33252031d91384b0f47fc449a276672/src/gym/transformer_sac_vectorq_v2_final_fineTune/network.py',
        downloaded_provenance=provenance, teacher_only=True, actuator_messages_sent=0))
    torch.set_num_threads(1)
    checkpoint = torch.load(root/'inference_ckpt.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if (config['token_dim'], config['window_size'], config['action_dim']) != (50, 75, 3):
        raise ValueError('Reviewed Vector-Q input/output contract differs')
    spec = importlib.util.spec_from_file_location('reviewed_vectorq_network', root/'network.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.TransformerEncoder(token_dim=50, d_model=config['d_model'],
        n_heads=config['n_heads'], n_layers=config['n_layers'], ffn_dim=config['ffn_dim'], window_size=75)
    head = module.PolicyHead(d_model=config['d_model'], action_dim=3, hidden_units=config['policy_hidden'])
    encoder.load_state_dict(checkpoint['policy_encoder'], strict=True)
    head.load_state_dict(checkpoint['policy_head'], strict=True)
    encoder.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    with np.load(root/'sample_data.npz', allow_pickle=False) as data:
        observations, stored = data['obs'], data['action']
    if observations.shape != (216, 75, 50) or not np.isfinite(observations).all():
        raise ValueError('Finite public recorded window archive required')
    inputs = torch.tensor(observations, dtype=torch.float32).clamp(-3, 3)
    with torch.no_grad():
        means = head.net(encoder(inputs)).chunk(2, dim=-1)[0]
        actions = means.tanh().numpy()
        singles, durations = [], []
        for i in range(12):
            started = time.perf_counter()
            predicted = head.net(encoder(inputs[i:i+1])).chunk(2, dim=-1)[0].tanh().numpy()
            durations.append(time.perf_counter()-started)
            singles.append(predicted[0])
    difference = np.max(np.abs(actions[:12]-np.array(singles)))
    if not np.isfinite(actions).all() or difference > 1e-5:
        raise ValueError('Batched/single-window reconstruction differs')
    np.savez_compressed(run/'predictions.npz', action=actions, stored_action=stored,
        single_window_seconds=np.array(durations))
    write_json(run/'result.json', dict(status='completed_offline_candidate', vehicle='ks_mazda_miata',
        windows=len(observations), window_shape=[75, 50], output_shape=list(actions.shape),
        action_mean=actions.mean(0).tolist(), action_std=actions.std(0).tolist(),
        mean_absolute_difference_from_stored_collection_action=np.abs(actions-stored).mean(0).tolist(),
        single_batch_max_difference=float(difference),
        cpu_single_window_median_ms=float(np.median(durations)*1000),
        cpu_single_window_max_ms=float(max(durations)*1000),
        live_lap_verified=False, cross_car_transfer_verified=False, student_optimizer_updates=0,
        predictions_sha256=sha256(run/'predictions.npz')))


if __name__ == '__main__':
    main()
