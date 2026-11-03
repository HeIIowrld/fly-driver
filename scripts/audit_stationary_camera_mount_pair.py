"""Compare archived paired camera poses and RGB, retaining failed run status."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_camera_mount import validate_frame_mount
from flybrain_drive.flyvis_camera import FlyvisCameraProjector


def load(root, config_file, expected_revision, count=5):
    result = json.loads((root / 'result.json').read_text(encoding='utf8'))
    config = json.loads(config_file.read_text(encoding='utf8'))
    if (type(count) is not int or count < 1 or result.get('vision', {}).get('frames') != count
            or result.get('archive', {}).get('all_hashes_verified') is not True):
        raise ValueError('Original declared-frame archived lease required')
    frames = []
    with CameraArchive(root / 'vision') as archive:
        if archive.bundle_sha256 != result['archive']['bundle_sha256']:
            raise ValueError('Original archive changed')
        for index in range(count):
            frame = archive.json(f'frame-{index:04d}.json')
            if frame['index'] != index or frame['camera_layout_revision'] != expected_revision:
                raise ValueError('Original capture index and mount revision required')
            validate_frame_mount(frame, config)
            views = {}
            for view in frame['views']:
                raw = archive.read(Path(view['file']).name)
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = np.asarray(image.convert('RGB'))
                if rgb.shape != (180, 320, 3):
                    raise ValueError('Original RGB dimensions changed')
                views[view['view_id']] = (view, rgb)
            frames.append((frame, views))
    return result, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dashboard', 'exterior', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    old, dashboard = load(args.dashboard, Path('configs/ac-csp-yaw-retina.json'), 2)
    new, exterior = load(args.exterior, Path('configs/ac-csp-exterior-front-v3-retina.json'), 3)
    a, da = dashboard[2]
    b, ex = exterior[2]
    if a['generation'] != b['generation'] or a['car_id'] != b['car_id'] or a['track_id'] != b['track_id']:
        raise ValueError('Original paired session, car and track required')
    uv = np.array([(q, r) for q in range(-15, 16)
                   for r in range(max(-15, -15-q), min(15, 15-q)+1)])
    dc = json.loads(Path('configs/ac-csp-yaw-retina.json').read_text(encoding='utf8'))
    ec = json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
    dp, ep = [FlyvisCameraProjector(c, uv[:, 0], uv[:, 1]) for c in (dc, ec)]
    if not np.array_equal(dp.directions_body, ep.directions_body):
        raise ValueError('Registered camera ray directions differ')
    summary = {}
    for name in ('front', 'left', 'right', 'rear'):
        v1, img1 = da[name]
        v2, img2 = ex[name]
        p1, p2 = [np.array(v['renderer_only_world_position']) for v in (v1, v2)]
        look1, look2 = [np.array(v['renderer_only_world_look']) for v in (v1, v2)]
        delta = p2 - p1
        diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
        retina1, retina2 = dp.sample(img1), ep.sample(img2)
        summary[name] = dict(world_center_displacement_m=float(np.linalg.norm(delta)),
                             view_direction_dot=float(look1 @ look2),
                             rgb_mean_absolute_difference=float(diff.mean()),
                             changed_pixel_fraction=float(np.any(diff > 0, axis=2).mean()),
                             retina_mean_absolute_difference=float(np.abs(retina1-retina2).mean()),
                             dashboard_retina_mean=float(retina1.mean()), exterior_retina_mean=float(retina2.mean()))
    front = da['front'][0]
    vfront = np.array(front['renderer_only_world_look'])
    vup = np.array(front['renderer_only_world_up'])
    vleft = np.array(da['left'][0]['renderer_only_world_look'])
    d = np.array(ex['front'][0]['renderer_only_world_position']) - np.array(front['renderer_only_world_position'])
    result = dict(status='completed_archived_frame_comparison', input_run_statuses=[old['status'], new['status']],
                  input_failures=[old.get('reason'), new.get('reason')], same_generation=a['generation'],
                  input_bundle_sha256=[old['archive']['bundle_sha256'], new['archive']['bundle_sha256']],
                  pair_frame_index=2, forward_left_up_offset_m=[float(d @ v) for v in (vfront, vleft, vup)],
                  views=summary, actor_deployed=False, vehicle_control_sent=False, training_performed=False,
                  interpretation='Sequential original frames; parent run statuses must be considered individually; '
                  'frame metadata and RGB only; no physical timing, scene identity or driving guarantee')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
