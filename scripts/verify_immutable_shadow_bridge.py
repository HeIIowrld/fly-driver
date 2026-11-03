"""Offline original-clock replay of immutable inputs through the real shadow producer."""

import argparse
import hashlib
import io
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import flybrain_drive.multirate_live_shadow as live
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('recording', 'config', 'training', 'opponent', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    root = Path(a.recording)
    result, manifest = read(root/'result.json'), read(root/'manifest.json')['config']
    bundle = read(root/'vision/bundle.json')
    archive = root/'vision'/bundle['file']
    if result['status'] != 'completed' or sha256(archive) != bundle['sha256']:
        raise ValueError('Completed unchanged original camera archive required')
    rows = [json.loads(x) for x in (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    if not rows:
        raise ValueError('Recorded chronological decisions required')
    run = begin_run(a.output, {'program_sha256': sha256(__file__), 'recording': str(root),
        'recording_sha256': sha256(root/'result.json'), 'bundle_sha256': bundle['sha256'],
        'events_sha256': sha256(root/'events.jsonl'), 'config_sha256': sha256(a.config),
        'shadow_source_sha256': sha256('src/flybrain_drive/multirate_live_shadow.py'),
        'adapter_sha256': sha256('src/flybrain_drive/immutable_camera_group.py'),
        'clock': 'original recorded decision time substituted for host clock; offline only'})
    output = {'status': 'failed', 'training_admissible': False, 'vehicle_control_sent': False}
    original_time_module = live.time
    try:
        shadow = live.MultirateLiveShadow(config_path=a.config, training=a.training,
                                          opponent=a.opponent)
        shadow.configure_immutable(manifest.get('pixel_configuration') or
                                   manifest['sensory_configuration'], a.config)
        reference, held, previous_clock, previous_capture = shadow.vis_state, None, None, None
        records, ticks = [], 0
        with zipfile.ZipFile(archive) as z, torch.no_grad(), torch.device('cuda'):
            for index, source in enumerate(rows):
                camera, clock = source['camera'], source['decision_monotonic_seconds']
                capture, samples, reference_images = camera['capture_id'], {}, []
                if camera.get('metadata_transport') != 'mmf_slot_sha256_v1':
                    raise ValueError('Original MMF metadata required')
                for view in camera['views']:
                    raw = z.read(view['file'].removeprefix('vision/'))
                    if hashlib.sha256(raw).hexdigest() != view['file_sha256']:
                        raise ValueError('Original image hash differs')
                    with Image.open(io.BytesIO(raw)) as image:
                        rgb = np.asarray(image.convert('RGB')).copy()
                    reference_images.append(shadow.projector.sample(rgb))
                    samples[view['view_id']] = dict(rgb=rgb, capture_id=capture,
                        generation=camera['generation'], received=camera['received'],
                        render_started_lower=view['render_started_lower'])
                if previous_capture is not None and capture != previous_capture:
                    for dt, count in held_input_batches(clock-previous_clock):
                        while count:
                            n = min(count, 16)
                            reference = shadow.network.simulate(
                                held[:, None, None, :].expand(-1, n, -1, -1),
                                dt=dt, initial_state=reference, as_states=True)[-1]
                            count -= n
                            ticks += n
                live.time = types.SimpleNamespace(perf_counter=lambda value=clock: value)

                def no_body(event, value=clock):
                    event['decision'] = value
                    return None

                event, observed = shadow.process_immutable(camera, samples, no_body)
                duplicate = observed['status'] == 'duplicate_not_reprocessed'
                if duplicate != (previous_capture == capture):
                    raise ValueError('Duplicate handling differs from original capture sequence')
                if not duplicate:
                    held = torch.as_tensor(np.stack(reference_images), dtype=torch.float32,
                                           device='cuda')
                    previous_clock, previous_capture = clock, capture
                error = float(torch.max(torch.abs(reference.nodes.activity -
                                                  shadow.vis_state.nodes.activity)).item())
                if error > 2e-6:
                    raise ValueError('Real shadow producer state differs from original RGB replay')
                records.append({'index': index, 'capture': capture, 'status': observed['status'],
                                'missing': event['missing_intermediate_captures'],
                                'state_error': error})
        write_json(run/'events.json', records)
        output.update(status='completed_immutable_shadow_bridge_replay', decisions=len(records),
            unique_captures=sum(r['status'] != 'duplicate_not_reprocessed' for r in records),
            duplicates=sum(r['status'] == 'duplicate_not_reprocessed' for r in records),
            absent_intermediate_captures=sum(r['missing'] for r in records), internal_ticks=ticks,
            maximum_state_error=max(r['state_error'] for r in records),
            events_sha256=sha256(run/'events.json'), initialization=shadow.provenance,
            synthetic_clock_substitution=True, body_context_available=False,
            live_mmf_consumer_verified=False, realtime_latency_verified=False,
            physical_sample_or_exposure_time_verified=False, optimizer_updates=0)
    except Exception as error:
        output.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        live.time = original_time_module
        write_json(run/'result.json', output)
        print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
