"""Inspect unmodified archived stationary camera frames for luminance transients."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from scripts.audit_stationary_camera_mount_pair import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mount', choices=('dashboard', 'exterior'), required=True)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config_path = Path('configs/ac-csp-yaw-retina.json' if args.mount == 'dashboard'
                       else 'configs/ac-csp-exterior-front-v3-retina.json')
    source, frames = load(args.run, config_path, 2 if args.mount == 'dashboard' else 3, count=args.frames)
    config = json.loads(config_path.read_text(encoding='utf8'))
    uv = np.array([(q, r) for q in range(-15, 16)
                   for r in range(max(-15, -15-q), min(15, 15-q)+1)])
    projector = FlyvisCameraProjector(config, uv[:, 0], uv[:, 1])
    views = {}
    for view in ('front', 'left', 'right', 'rear'):
        retina = np.stack([projector.sample(row[view][1]) for _, row in frames])
        views[view] = dict(means=retina.mean(axis=1).tolist(),
                           mean_absolute_change_from_first=np.abs(retina-retina[:1]).mean(axis=1).tolist())
    result = dict(status='completed_stationary_retina_brightness_diagnostic', mount=args.mount,
                  parent_run_status=source['status'], input_bundle_sha256=source['archive']['bundle_sha256'],
                  original_program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  camera_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  producer_frame_clocks=[frame['script_clock'] for frame, _ in frames], views=views,
                  interpretation='Unmodified sequential LDR frames and receptor samples; brightness changes do not '
                  'identify their rendering cause, physical scene motion, biological response or driving performance',
                  training_performed=False, vehicle_control_sent=False, actor_deployed=False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
