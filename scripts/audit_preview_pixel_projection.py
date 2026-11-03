"""All-history label reconstruction and selected original-pixel projection; offline diagnostics."""

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
from flybrain_drive.diagnostic_path_projection import FutureTrace, project_reference_points
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout-batch', required=True)
    parser.add_argument('--preview-data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    batch_root, preview_root = Path(args.readout_batch), Path(args.preview_data)
    source = read(batch_root/'result.json')
    labels = {e['id']: e for e in read(preview_root/'dataset.json')['episodes']}
    if source['status'] != 'completed_intermediate_visual_readout_batch':
        raise ValueError('Completed original histories required')
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        projection_module_sha256=sha256('src/flybrain_drive/diagnostic_path_projection.py'),
        source_result_sha256=sha256(batch_root/'result.json'),
        original_dataset_sha256=sha256(preview_root/'dataset.json'),
        sample_rule='indices0,N//2,3*N//4 in every original whole history',
        actor_input=False, optimizer_updates=0, actuator_messages_sent=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    summaries = []
    for entry in source['episodes']:
        label, root = labels[entry['id']], Path(entry['recording'])
        if label['role'] not in ('train', 'development') or label['role'] != entry['role']:
            raise ValueError('No evaluation access or role changes')
        original = read(Path(entry['output'])/'manifest.json')['config']
        if (sha256(root/'events.jsonl') != original['events_sha256']
                or sha256(label['targets_file']) != label['targets_sha256']):
            raise ValueError('Original clocks/targets/events changed')
        events = [json.loads(line) for line in (root/'events.jsonl').read_text('utf8').splitlines()]
        telemetry_path = root/'control/api-receipt.json'
        trace = FutureTrace(read(telemetry_path)['telemetry'])
        with load_array(label['targets_file'], allow_pickle=False) as a:
            targets, masks = a['preview'].copy(), a['preview_valid'].copy()
            clocks = a['observation_time'].copy()
        if len(events) != len(targets):
            raise ValueError('Every original event required')
        selected = [0, len(events)//2, 3*len(events)//4]
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        points_all, uv_all, projected_all, eligible_all = [], [], [], []
        maximum_error = 0.
        samples = []
        with CameraArchive(root/'vision') as archive:
            for i, event in enumerate(events):
                np.testing.assert_equal(event['decision_monotonic_seconds'], clocks[i])
                camera = event['camera']
                raw = archive.read(Path(camera['file']).name)
                if hashlib.sha256(raw).hexdigest() != camera['file_sha256']:
                    raise ValueError('Original camera metadata hash changed')
                frame = json.loads(raw)
                camera_geometry(frame['views'])
                front = next(v for v in frame['views'] if v['view_id'] == 'front')
                start, points, available = trace.at(front['render_started_clock'])
                forward, right = (np.asarray(x)[[0, 2]] for x in (
                    front['renderer_only_world_look'],
                    matrix(front['view_matrix_diagnostic'])[:3, 0]))
                forward, right = forward/np.linalg.norm(forward), right/np.linalg.norm(right)
                displacement = (points-start)[:, [0, 2]]
                f, r = displacement@forward, displacement@right
                chord2 = np.sum(displacement**2, axis=1)
                eligible = available & (f > 0) & (chord2 > .01)
                np.testing.assert_array_equal(eligible, masks[i])
                reconstructed = np.zeros((3, 2))
                reconstructed[eligible, 0] = np.arctan2(r[eligible], f[eligible])
                reconstructed[eligible, 1] = 2*r[eligible]/chord2[eligible]
                error = float(np.max(np.abs(reconstructed-targets[i])))
                if error > 1e-9:
                    raise ValueError('Independent whole-history target reconstruction differs')
                maximum_error = max(maximum_error, error)
                uv, in_front, inside = project_reference_points(points, front)
                points_all.append(points)
                uv_all.append(uv)
                projected_all.append(inside & eligible)
                eligible_all.append(eligible)
                if i in selected:
                    pixel = next(v for v in camera['views'] if v['view_id'] == 'front')
                    image_bytes = archive.read(Path(pixel['file']).name)
                    if hashlib.sha256(image_bytes).hexdigest() != pixel['file_sha256']:
                        raise ValueError('Original image bytes differ')
                    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
                    if hashlib.sha256(image.tobytes()).hexdigest() != pixel['rgb_sha256']:
                        raise ValueError('Original RGB differs')
                    axis = axes[selected.index(i)]
                    axis.imshow(image, extent=(0, 1, 1, 0))
                    axis.set_aspect(image.shape[0]/image.shape[1])
                    for h, color in enumerate(('cyan', 'yellow', 'magenta')):
                        if eligible[h] and in_front[h]:
                            axis.plot(uv[h, 0], uv[h, 1], '+', color=color, markersize=12)
                            axis.annotate(f'{(5, 10, 20)[h]}m', uv[h], color=color, fontsize=9)
                    axis.set(xlim=(0, 1), ylim=(1, 0), title=f'{entry["role"]} row{i}')
                    axis.axis('off')
                    samples.append(dict(index=i, capture_id=camera['capture_id'],
                        rgb_sha256=pixel['rgb_sha256'], pixel_file_sha256=pixel['file_sha256'],
                        normalized_projected_points=uv.tolist(), valid=eligible.tolist(),
                        in_image=(inside & eligible).tolist()))
        fig.suptitle(entry['id']+' — future car reference points; diagnostic only')
        fig.tight_layout()
        fig.savefig(run/(entry['id']+'.png'), dpi=130)
        plt.close(fig)
        visible, eligible_all = np.array(projected_all), np.array(eligible_all)
        np.savez_compressed(run/(entry['id']+'.npz'), world_reference_points=points_all,
            projected_uv=uv_all, in_image_and_valid=visible, valid=eligible_all,
            observation_time=clocks)
        bearing = targets[:, :, 0]
        summaries.append(dict(id=entry['id'], role=entry['role'], observations=len(events),
            maximum_target_reconstruction_error=maximum_error,
            eligible_counts=eligible_all.sum(0).tolist(), visible_counts=visible.sum(0).tolist(),
            bearing_min_max_rad=[[float(bearing[eligible_all[:, h], h].min()),
                                  float(bearing[eligible_all[:, h], h].max())] for h in range(3)],
            original_events_sha256=sha256(root/'events.jsonl'),
            telemetry_sha256=sha256(telemetry_path), samples=samples))
        write_json(run/'progress.json', dict(complete=len(summaries), total=len(labels)))
    write_json(run/'result.json', dict(status='completed_preview_pixel_projection_audit',
        episodes=summaries, source_rows_removed=0, target_changes=0, actor_inputs_added=False,
        pixel_occlusion_or_road_membership_verified=False,
        reference_points_are_original_car_positions_not_road_surface=True,
        optimizer_updates=0, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
