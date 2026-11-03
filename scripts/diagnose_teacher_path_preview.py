"""Offline visual information assay with privileged future-path labels only."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from flybrain_drive.ac_csp_geometry import camera_geometry, matrix
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.teacher_path_preview import future_path_preview


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    parent = Path(args.data)
    manifest = read(parent/'dataset.json')
    assert all(e['role'] in ('train', 'development') for e in manifest['episodes'])
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        label_module_sha256=sha256('src/flybrain_drive/teacher_path_preview.py'),
        parent_sha256=sha256(parent/'dataset.json'),
        distances_m=[5, 10, 20], ridge_penalty=1.,
        fixed_probe='Standardized scene only, ridge mean-square loss + coefficient squared norm',
        actor_input='Existing scene576 only; no body, clock, pose, trajectory or controls',
        scope='Engineering information probe, not fly circuit learning or driving qualification',
        evaluation_access=False, ui_actions_sent=0, actuator_messages_sent=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'executed-label-module.py').write_bytes(
        Path('src/flybrain_drive/teacher_path_preview.py').read_bytes())
    episodes, provenance = [], []
    for entry in manifest['episodes']:
        path = parent/entry['file']
        assert sha256(path) == entry['sha256']
        with load_array(path, allow_pickle=False) as a:
            scene = a['scene_features'].copy()
            observation_time = a['observation_time'].copy()
        source = Path(entry['original']['recording'])
        assert sha256(source/'events.jsonl') == entry['source_events_sha256']
        events = [json.loads(line) for line in
                  (source/'events.jsonl').read_text(encoding='utf8').splitlines()]
        telemetry = read(source/'control/api-receipt.json')['telemetry']
        labels, masks, reasons = [], [], Counter()
        assert len(events) == len(scene)
        with zipfile.ZipFile(source/'vision/camera.zip') as archive:
            for i, event in enumerate(events):
                assert event['decision_monotonic_seconds'] == observation_time[i]
                cam = event['camera']
                raw = archive.read(Path(cam['file']).name)
                assert hashlib.sha256(raw).hexdigest() == cam['file_sha256']
                frame = json.loads(raw)
                assert frame['generation'] == cam['generation']
                try:
                    camera_geometry(frame['views'])
                except ValueError as error:
                    labels.append(np.zeros((3, 2)))
                    masks.append(np.zeros(3, bool))
                    reasons[str(error)] += 1
                    continue
                front = next(v for v in frame['views'] if v['view_id'] == 'front')
                y, mask = future_path_preview(telemetry, front['render_started_clock'],
                    front['renderer_only_world_look'],
                    matrix(front['view_matrix_diagnostic'])[:3, 0])
                labels.append(y)
                masks.append(mask)
        y, valid = np.array(labels), np.array(masks)
        out = run/entry['file']
        np.savez_compressed(out, preview=y, preview_valid=valid, observation_time=observation_time)
        episodes.append(dict(role=entry['role'], file=entry['file'], x=scene, y=y, valid=valid))
        provenance.append(dict(file=entry['file'], role=entry['role'], observations=len(scene),
            auxiliary_sha256=sha256(out), original_episode_sha256=entry['sha256'],
            events_sha256=entry['source_events_sha256'],
            telemetry_receipt_sha256=sha256(source/'control/api-receipt.json'),
            valid_per_distance=valid.sum(0).tolist(), geometry_rejections=dict(reasons)))
        write_json(run/'progress.json', dict(completed_episodes=len(episodes)))
    train = [e for e in episodes if e['role'] == 'train']
    x = np.concatenate([e['x'] for e in train]).astype(float)
    y = np.concatenate([e['y'] for e in train])
    mask = np.concatenate([e['valid'] for e in train])
    mean, scale = x.mean(0), x.std(0).clip(1e-5)
    z = ((x-mean)/scale).clip(-5, 5)
    weights, biases = [], []
    for horizon in range(3):
        keep = mask[:, horizon]
        assert keep.sum() > 576
        xx, yy = z[keep], y[keep, horizon]
        mx, my = xx.mean(0), yy.mean(0)
        centered = xx-mx
        weight = np.linalg.solve(centered.T@centered/len(xx)+np.eye(576),
                                 centered.T@(yy-my)/len(xx))
        weights.append(weight)
        biases.append(my-mx@weight)
    weights, biases = np.array(weights), np.array(biases)
    np.savez_compressed(run/'probe.npz', weights=weights, biases=biases, mean=mean, scale=scale)
    scores = []
    for episode in episodes:
        z = ((episode['x']-mean)/scale).clip(-5, 5)
        variants = dict(intact=z, all_visual_train_mean=np.zeros_like(z))
        for name, channels in [('motion_train_mean', slice(1, 9)),
                               ('intensity_train_mean', slice(0, 1))]:
            altered = z.reshape(-1, 4, 9, 16).copy()
            altered[:, :, channels] = 0
            variants[name] = altered.reshape(-1, 576)
        for condition, zz in variants.items():
            prediction = np.einsum('nf,hfc->nhc', zz, weights)+biases
            metrics = []
            for h in range(3):
                keep = episode['valid'][:, h]
                target = episode['y'][keep, h]
                metrics.append(dict(count=int(keep.sum()),
                    rmse=np.sqrt(np.mean((prediction[keep, h]-target)**2, axis=0)).tolist(),
                    train_mean_rmse=np.sqrt(np.mean(
                        (target-y[mask[:, h], h].mean(0))**2, axis=0)).tolist(),
                    zero_rmse=np.sqrt(np.mean(target**2, axis=0)).tolist()))
            scores.append(dict(file=episode['file'], role=episode['role'], condition=condition,
                               metrics_bearing_rad_and_chord_curvature_inv_m=metrics))
    write_json(run/'result.json', dict(status='completed_visual_preview_information_probe',
        episodes=provenance, scores=scores, normalization_and_fit_train_only=True,
        original_arrays_and_labels_unchanged=True, endpoint_missing_only_auxiliary_invalid=True,
        physiological_claim=False, closed_loop_claim=False, goal_achieved=False))


if __name__ == '__main__':
    main()
