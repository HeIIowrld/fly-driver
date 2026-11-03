"""Project retained diagnostic views to the registered fly retinal lattice; no model or control."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER, FlyvisCameraProjector
from flybrain_drive.graph import sha256


def check_view(view, name, index, config):
    """Reject changed dimensions, identity or projection; no optics calibration claim."""
    if (view.get('view_id') != name or view.get('file') != f'{index:04d}-{name}.png'
            or view.get('width') != 320 or view.get('height') != 180
            or view.get('requested_fov_degrees') != (56 if name == 'front' else 90)):
        raise ValueError('Original view identity, resolution and requested FOV required')
    matrix = view.get('projection_matrix_diagnostic')
    if not isinstance(matrix, str):
        raise ValueError('Original view projection matrix required')
    values = [float(x) for x in re.findall(r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',
                                            matrix)]
    if (len(values) != 16 or not np.isfinite(values).all()
            or not np.isclose(values[0], config['projection_x'], atol=1e-5)
            or not np.isclose(values[5], config['projection_y'], atol=1e-5)):
        raise ValueError('Original projection differs from retinal mapping')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--body-audit', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, body = Path(args.recording), Path(args.body_audit)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    body_result = json.loads((body/'result.json').read_text(encoding='utf8'))
    manifest = json.loads((body/'manifest.json').read_text(encoding='utf8'))['config']
    if (result.get('status') != 'completed' or result.get('training_admissible') is not False
            or body_result.get('status') != 'completed_causal_copy_body_diagnostic'
            or body_result.get('training_admissible') is not False
            or sha256(root/'result.json') != manifest['recording_result_sha256']
            or sha256(root/'diagnostic-frames.json') != result['metadata_sha256']
            or sha256(body/'body-context.npz') != body_result['body_context_sha256']
            or sha256(body/'view-receipts.json') != body_result['view_receipts_sha256']):
        raise ValueError('Unchanged original diagnostic and causal-body audit required')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    for key in ('camera_config', 'retina_registration'):
        if sha256(config[key]) != config[key+'_sha256']:
            raise ValueError('Pinned original camera and retina registration required')
    camera = json.loads(Path(config['camera_config']).read_text(encoding='utf8'))
    with load_array(config['retina_registration'], allow_pickle=False) as registration:
        projector = FlyvisCameraProjector(camera, registration['u'], registration['v'])
        np.testing.assert_array_equal(projector.uv, registration['image_uv'])
        np.testing.assert_array_equal(projector.directions_body, registration['body_directions'])
    rows = json.loads((root/'diagnostic-frames.json').read_text(encoding='utf8'))
    view_receipts = json.loads((body/'view-receipts.json').read_text(encoding='utf8'))
    with np.load(body/'body-context.npz', allow_pickle=False) as data:
        body_values, body_valid = data['values'].copy(), data['valid'].copy()
    if len(view_receipts) != len(rows)*4 or body_values.shape != (len(rows)*4, 21):
        raise ValueError('Complete original per-view body diagnostics required')
    retinal, clocks, image_records, colours = [], [], [], []
    source_dir = Path(rows[0]['source_file']).parent
    for index, row in enumerate(rows):
        source = Path(row['source_file'])
        raw = source.read_bytes()
        if source.parent != source_dir or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Original same-directory metadata differs')
        frame = json.loads(raw)
        if (row['index'] != index or frame.get('index') != index
                or frame.get('request_id') != result['request_id']
                or frame.get('camera_layout_revision') != 2
                or frame.get('aid_probe_only') is not True
                or frame.get('airflow_diagnostic') != row['diagnostic']):
            raise ValueError('Exact original diagnostic frame and camera layout required')
        view_map = {v['view_id']: v for v in frame['views']}
        if set(view_map) != set(VIEW_ORDER) or len(frame['views']) != 4:
            raise ValueError('Exactly four distinct original views required')
        projected, render_times = [], []
        for j, name in enumerate(VIEW_ORDER):
            view, body_row = view_map[name], view_receipts[index*4+j]
            if body_row['index'] != index or body_row['view'] != name:
                raise ValueError('Body mask and original view order differ')
            check_view(view, name, index, camera['views'][name])
            image_path = source_dir/view['file']
            image_bytes = image_path.read_bytes()
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = np.array(image.convert('RGB'))
            if rgb.shape != (180, 320, 3) or rgb.dtype != np.uint8:
                raise ValueError('Exact original RGB shape required')
            colours.append(len(np.unique(rgb.reshape(-1, 3), axis=0)))
            projected.append(projector.sample(rgb))
            render_times.append(view['render_started_clock'])
            image_records.append({'path': str(image_path),
                                  'sha256': hashlib.sha256(image_bytes).hexdigest()})
        retinal.append(np.stack(projected))
        clocks.append(render_times)
    if not rows or min(colours) <= 16 or body_valid.shape != body_values.shape:
        raise ValueError('Nonempty, nonconstant source images and explicit body masks required')
    if any(sha256(x['path']) != x['sha256'] for x in image_records):
        raise ValueError('Original diagnostic images changed during projection')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'recording_result_sha256': sha256(root/'result.json'),
        'body_audit_sha256': sha256(body/'result.json'), 'config_sha256': sha256(args.config),
        'images_hashed_after_capture': True, 'scope': __doc__})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'retina-body-diagnostic.npz', retinal=np.stack(retinal),
                        render_source_clock=np.asarray(clocks),
                        body_values=body_values.reshape(-1, 4, 21),
                        body_valid=body_valid.reshape(-1, 4, 21))
    write_json(run/'source-images.json', image_records)
    out = {'status': 'completed_retina_body_diagnostic_projection', 'groups': len(rows),
        'views': len(image_records), 'retinal_shape': list(np.stack(retinal).shape),
        'minimum_colours': min(colours), 'source_images_sha256': sha256(run/'source-images.json'),
        'arrays_sha256': sha256(run/'retina-body-diagnostic.npz'),
        'image_hashes_observed_at_capture': False, 'flyvis_network_executed': False,
        'exposure_or_physical_sample_time_verified': False, 'training_admissible': False,
        'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
