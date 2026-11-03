"""Replay original four-camera pixels through both retinal timing implementations."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_retina_producer_time import CSPProducerTimeRetina
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    manifest, bundle = read(root/'manifest.json'), read(root/'vision/bundle.json')
    if (sha256(root/'vision/camera.zip') != bundle['sha256']
            or sha256(root/'vision/archive.json') != bundle['archive_manifest_sha256']):
        raise ValueError('Original camera archive evidence mismatch')
    source_files = [root/'manifest.json', root/'events.jsonl', root/'observations.npz',
                    root/'vision/bundle.json', root/'vision/archive.json',
                    root/'vision/camera.zip']
    hashes = {str(path): sha256(path) for path in source_files}
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': hashes,
        'original_role_unchanged': True, 'optimizer_updates': 0, 'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    config = manifest['config']['sensory_configuration']
    legacy, candidate = CSPCameraRetina(config), CSPProducerTimeRetina(config)
    events = [json.loads(line) for line in (root/'events.jsonl').open(encoding='utf8')]
    with np.load(root/'observations.npz', allow_pickle=False) as data:
        keys = ('view_rgb', 'view_contrast', 'view_valid', 'contrast_valid', 'view_age')
        saved = {k: data['obs_'+k].copy() for k in keys}
        times = data['observation_time'].copy()
    if len(events) != len(times):
        raise ValueError('Complete original events and observations required')
    count, max_delta, sum_delta, mismatch, offsets = 0, 0., 0., 0, []
    cache_id, cache = None, None
    with CameraArchive(root/'vision') as archive:
        for i, event in enumerate(events):
            row = event['camera']
            if event['decision_monotonic_seconds'] != times[i]:
                raise ValueError('Original decision clock changed')
            if row['capture_id'] != cache_id:
                raw = archive.read(Path(row['file']).name)
                if hashlib.sha256(raw).hexdigest() != row['file_sha256']:
                    raise ValueError('Original metadata hash mismatch')
                metadata = json.loads(raw)
                if (metadata['index'] != row['capture_id']
                        or metadata['generation'] != row['generation']):
                    raise ValueError('Original source identity mismatch')
                view_metadata = {v['view_id']: v for v in metadata['views']}
                cache = {}
                for view in row['views']:
                    raw = archive.read(Path(view['file']).name)
                    if hashlib.sha256(raw).hexdigest() != view['file_sha256']:
                        raise ValueError('Original image byte hash mismatch')
                    pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
                    if hashlib.sha256(pixels.tobytes()).hexdigest() != view['rgb_sha256']:
                        raise ValueError('Original decoded image hash mismatch')
                    source_time = view_metadata[view['view_id']]['render_started_clock']
                    cache[view['view_id']] = (pixels, source_time)
                cache_id = row['capture_id']
            old_samples, new_samples = {}, {}
            for view in row['views']:
                name = view['view_id']
                pixels, source_time = cache[name]
                old_samples[name] = dict(rgb=pixels, capture_id=row['capture_id'],
                    generation=row['generation'], received=row['received'],
                    render_started_lower=view['render_started_lower'])
                new_samples[name] = dict(old_samples[name], producer_render_started=source_time)
                offsets.append(view['render_started_lower']-source_time)
            old = legacy.pack(old_samples, decision_time=times[i])
            new = candidate.pack(new_samples, decision_time=times[i])
            for key in keys:
                if not np.array_equal(old[key], saved[key][i]):
                    raise ValueError('Legacy sensory replay mismatch at '+str(i)+' '+key)
            for key in ('view_rgb', 'view_valid', 'contrast_valid', 'view_age'):
                mismatch += int(not np.array_equal(old[key], new[key]))
            delta = np.abs(old['view_contrast']-new['view_contrast'])
            max_delta = max(max_delta, float(delta.max()))
            sum_delta += float(delta.sum(dtype=np.float64))
            count += delta.size
            if (i+1) % 100 == 0:
                print(json.dumps({'observations_replayed': i+1}), flush=True)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Original source evidence changed')
    result = {'status': 'completed_original_producer_time_retina_replay',
        'observations': len(events), 'legacy_all_visual_arrays_exact': True,
        'noncontrast_array_mismatch_observations': mismatch,
        'contrast_mean_abs_difference': sum_delta/count, 'contrast_max_abs_difference': max_delta,
        'render_lower_minus_producer_clock_min': min(offsets),
        'render_lower_minus_producer_clock_max': max(offsets),
        'physical_exposure_or_body_registration_verified': False,
        'new_capture_or_actor_integration': False, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
