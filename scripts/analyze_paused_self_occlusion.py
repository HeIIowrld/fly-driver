"""Compare original paused camera sources, never admit diagnostic images as actor input."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_geometry import camera_geometry, matrix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--front-mount-comparison', action='store_true')
    args = parser.parse_args()
    source = Path(args.source)
    proof = read(source/'result.json')
    if (proof['status'] != 'completed_paused_camera_rotation_collection'
            or proof['original_frames'] != 2 or proof['actor_admissible'] is not False):
        raise ValueError('Completed paired diagnostic-only static source required')
    run = begin_run(args.output, dict(source_result_sha256=sha256(source/'result.json'),
        arguments=vars(args), program_sha256=sha256(__file__), actor_admissible=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    fig, axes = plt.subplots(2, 4, figsize=(16, 5.4))
    images, metadata, states = [], [], []
    names = (['native_dashboard_v2', 'bmw_z4_gt3_bumper_v1'] if args.front_mount_comparison
             else ['full_scene', 'track_only'])
    for i, name in enumerate(names):
        scene = 'full_scene' if args.front_mount_comparison else name
        condition = proof['conditions'][i]
        if (condition['condition']['scene'] != scene
                or condition['condition']['yaw_left_rad_s'] != 0):
            raise ValueError('Same static pose source comparison required')
        if args.front_mount_comparison and condition['condition'].get(
                'front_mount', 'native_dashboard_v2') != name:
            raise ValueError('Explicit full-scene front mount pair required')
        root = source/f'condition-{i:02d}'
        row = read(root/'rows.json')[0]
        before, after = read(root/'before.json'), read(root/'after.json')
        for state in (before, after):
            if not state['session']['paused'] or state['session']['ai_controlled']:
                raise ValueError('Paused AI-off original source required')
            states.append(state)
        with CameraArchive(root/'vision') as archive:
            frame = archive.read(row['file'])
            if hashlib.sha256(frame).hexdigest() != row['file_sha256']:
                raise ValueError('Original camera metadata changed')
            meta = json.loads(frame)
            camera_geometry(meta['views'])
            metadata.append(meta)
            entries = {}
            for j, name in enumerate(['front', 'left', 'right', 'rear']):
                v = next(v for v in row['views'] if v['view_id'] == name)
                raw = archive.read(v['file'])
                if hashlib.sha256(raw).hexdigest() != v['file_sha256']:
                    raise ValueError('Original camera file changed')
                rgb = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
                if hashlib.sha256(rgb.tobytes()).hexdigest() != v['rgb_sha256']:
                    raise ValueError('Original RGB changed')
                entries[name] = rgb
                Image.fromarray(rgb).save(run/(names[i]+'-'+name+'.png'))
                axes[i, j].imshow(rgb)
                axes[i, j].axis('off')
                axes[i, j].set_title(names[i]+' / '+name, fontsize=9)
            images.append(entries)
    packets = [s['packet_id'] for s in states]
    native_equal = all(s['native'] == states[0]['native'] for s in states)
    if len(set(packets)) != 1 or not native_equal:
        raise ValueError('Native paused packet/values changed across static comparison')
    records = []
    for name in ['front', 'left', 'right', 'rear']:
        views = [next(v for v in m['views'] if v['view_id'] == name) for m in metadata]
        np.testing.assert_allclose(matrix(views[0]['projection_matrix_diagnostic']),
                                   matrix(views[1]['projection_matrix_diagnostic']),
                                   atol=1e-8, rtol=0)
        rotation_only = args.front_mount_comparison and name == 'front'
        view0, view1 = (matrix(v['view_matrix_diagnostic']) for v in views)
        np.testing.assert_allclose(view0[:3, :3] if rotation_only else view0,
                                   view1[:3, :3] if rotation_only else view1, atol=1e-8, rtol=0)
        delta = np.abs(images[0][name].astype(float)-images[1][name])
        records.append(dict(view=name, mean_absolute_rgb_delta=float(delta.mean()),
            fraction_pixels_any_channel_changed=float((delta.max(2) > 0).mean()),
            same_pose_projection=not rotation_only))
    mount_error = None
    if args.front_mount_comparison:
        views = {v['view_id']: v for v in metadata[1]['views']}
        expected = (np.array(views['left']['renderer_only_world_position'])
                    + views['right']['renderer_only_world_position'])/2
        expected += 2.5*np.array(views['front']['renderer_only_world_look'])
        actual = np.array(views['front']['renderer_only_world_position'])
        mount_error = float(np.max(np.abs(actual-expected)))
        # Existing camera_geometry uses2mm for returned native-float pose consistency.
        # Never infer or fit a translation to make this comparison pass.
        np.testing.assert_allclose(actual, expected, atol=.002, rtol=0)
    fig.tight_layout()
    fig.savefig(run/'comparison.png', dpi=130)
    plt.close(fig)
    write_json(run/'result.json', dict(status='completed_static_self_occlusion_source_comparison',
        views=records, native_packets_identical=True, native_values_identical=True,
        front_mount_comparison=args.front_mount_comparison,
        maximum_front_mount_formula_error_m=mount_error,
        existing_camera_pose_tolerance_m=.002,
        actor_admissible=False, ui_actions_sent=0, actuator_messages_sent=0,
        trained_samples=0, goal_achieved=False))


if __name__ == '__main__':
    main()
