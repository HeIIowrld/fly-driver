"""Describe retained angular comparison residuals without changing validity gates."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.review)
    result_path = root / 'result.json'
    prior = json.loads(result_path.read_text(encoding='utf8'))
    if prior['status'] != 'completed' or len(prior['recordings']) != 4:
        raise ValueError('The four completed original geometry reviews are required')
    arrays = [root / f'geometry-{i:02}.npz' for i in range(1, 5)]
    run = begin_run(args.output, {
        'source_result_sha256': sha256(result_path),
        'source_arrays_sha256': [sha256(p) for p in arrays],
        'top_fraction': 0.05,
        'diagnostic_only': True,
        'actor_deployed': False,
    })
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    episodes = []
    for row, path in zip(prior['recordings'], arrays, strict=True):
        with np.load(path, allow_pickle=False) as archive:
            native = archive['angular_native_body'].copy()
            renderer = archive['angular_renderer_body'].copy()
        if (native.ndim != 2 or native.shape[1] != 3 or renderer.shape != native.shape
                or not np.isfinite(native).all() or not np.isfinite(renderer).all()):
            raise ValueError('Finite matched native and renderer comparisons required')
        axes = {}
        for i, name in enumerate(('roll', 'pitch', 'yaw')):
            if len(native) != row['angular'][name]['samples']:
                raise ValueError('Sample count differs from retained review')
            residual = native[:, i] - renderer[:, i]
            squared = residual ** 2
            rmse = float(np.sqrt(squared.mean()))
            if not np.isclose(rmse, row['angular'][name]['uncorrected_rmse'], atol=1e-12):
                raise ValueError('Original review RMSE differs from retained array')
            n = int(np.ceil(len(residual) * 0.05))
            total = float(squared.sum())
            axes[name] = {
                'rmse': rmse,
                'median_absolute_error': float(np.median(np.abs(residual))),
                'p95_absolute_error': float(np.quantile(np.abs(residual), 0.95)),
                'largest_5_percent_squared_error_fraction': (
                    float(np.sort(squared)[-n:].sum() / total) if total else 0.0
                ),
                'original_gate_passed': row['angular'][name]['passed'],
            }
        episodes.append({'recording': row['recording'], 'samples': len(native), 'axes': axes})
    write_json(run / 'result.json', {
        'status': 'completed_residual_description',
        'episodes': episodes,
        'source_clock_verified': False,
        'no_validity_gate_changed': True,
        'no_root_cause_established': True,
        'actor_deployed': False,
        'goal_achieved': False,
    })
    print(json.dumps(episodes), flush=True)


if __name__ == '__main__':
    main()
