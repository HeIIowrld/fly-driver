"""Describe saved four-view pixel changes against diagnostic yaw, with no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--pixel', required=True)
    p.add_argument('--body', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    recording, pixel, body = map(Path, (a.recording, a.pixel, a.body))
    source = recording/'result.json'
    receipt, pr, br = read(source), read(pixel/'result.json'), read(body/'result.json')
    if (receipt['status'] != 'completed'
            or pr['status'] != 'completed_diagnostic_owned_ai_image_archive_and_projection'
            or br['status'] != 'completed_causal_body_window_diagnostic'
            or pr['training_admissible'] is not False
            or sha256(pixel/'retinal-inputs.npz') != pr['retinal_sha256']
            or sha256(body/'predictions.npz') != br['predictions_sha256']):
        raise ValueError('Exact diagnostic-only pixel and body histories required')
    pcfg = read(pixel/'manifest.json')['config']['source_hashes']
    bcfg = read(body/'manifest.json')['config']['source_hashes']
    for cfg in (pcfg, bcfg):
        required = {str(Path(k)).lower(): v for k, v in cfg.items()}
        key = str(source).lower()
        if key not in required or required[key] != sha256(source):
            raise ValueError('Both evidence sources must pin the same recording')
    with np.load(pixel/'retinal-inputs.npz', allow_pickle=False) as x:
        retina, clocks = x['retinal'].copy(), x['render_source_clocks'].copy()
    with np.load(body/'predictions.npz', allow_pickle=False) as x:
        indices = x['frame_index'].copy()
        target, native = x['target'].copy(), x['causal_window'].copy()
    if (retina.shape != (receipt['frames'], 4, 721)
            or clocks.shape != (receipt['frames'], 4)
            or indices.ndim != 1 or len(indices) != len(target) or len(target) != len(native)
            or not all(np.isfinite(x).all() for x in (retina, clocks, target, native))
            or np.any(indices < 1) or np.any(indices >= len(retina)-1)):
        raise ValueError('Finite aligned original arrays and adjacent frames required')
    dt = clocks[indices]-clocks[indices-1]
    if np.any(dt <= 0) or np.any(dt > .15):
        raise ValueError('Per-view past frame spacing must be positive and at most 150ms')
    # At i use only the i-1 and i camera frames. The following interval's heading
    # and native window ending near i+1 are diagnostic comparisons, never inputs.
    difference = (retina[indices]-retina[indices-1])/dt[:, :, None]
    signed = difference.mean(axis=2)
    energy = np.abs(difference).mean(axis=2)
    split = len(indices)//2
    if np.any(np.diff(indices) <= 0):
        raise ValueError('Original ordered indices required')
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_result_sha256': sha256(source), 'pixel_sha256': pr['retinal_sha256'],
        'body_prediction_sha256': br['predictions_sha256'],
        'feature': 'per-view signed and absolute mean of backward retinal intensity differences',
        'target': 'following-interval diagnostic heading change; host receipt yaw diagnostic',
        'scope': 'correlation only; no fit; neither FlyVis activity nor optic flow',
        'actor_deployed': False, 'optimizer_updates': 0})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'observations.npz', frame_index=indices, per_view_dt=dt,
                        signed_rate=signed, abs_rate=energy, diagnostic_future_yaw=target,
                        diagnostic_native_yaw=native)
    reports = {}
    for k, name in enumerate(('front', 'left', 'right', 'rear')):
        reports[name] = {'signed_rate_median': float(np.median(signed[:, k])),
            'absolute_rate_median': float(np.median(energy[:, k])),
            'signed_vs_future_yaw_correlation': corr(signed[:, k], target),
            'abs_vs_future_abs_yaw_correlation': corr(energy[:, k], np.abs(target)),
            'first_half_signed_vs_future_yaw': corr(signed[:split, k], target[:split]),
            'second_half_signed_vs_future_yaw': corr(signed[split:, k], target[split:])}
    result = {'status': 'completed_retinal_change_yaw_diagnostic', 'frames': len(indices),
        'views': reports, 'positive_yaw_count': int((target > .02).sum()),
        'negative_yaw_count': int((target < -.02).sum()),
        'half_turn_counts': [[int((part > .02).sum()), int((part < -.02).sum())]
                             for part in (target[:split], target[split:])],
        'observations_sha256': sha256(run/'observations.npz'),
        'actor_deployed': False, 'optimizer_updates': 0, 'physical_sync_verified': False,
        'training_admissible': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
