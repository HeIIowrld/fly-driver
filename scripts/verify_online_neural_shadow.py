"""Replay sealed diagnostic inputs through the actual neural-shadow implementation."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.online_neural_shadow import OnlineNeuralShadow
from flybrain_drive.per_view_body_history import PerViewBodyHistory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('recording', 'flight', 'config', 'training', 'opponent', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--late-body', action='store_true')
    args = p.parse_args()
    root, flight = Path(args.recording), Path(args.flight)
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    sr, fr = read(root/'result.json'), read(flight/'result.json')
    files = ((root/'online-body-shadow.json', sr['shadow_sha256']),
             (root/'native-copy-windows.json', sr['native_copy_sha256']),
             (root/'diagnostic-frames.json', sr['metadata_sha256']),
             (flight/'outputs.npz', fr['arrays_sha256']))
    fm = read(flight/'manifest.json')['config']
    if (sr['status'] != 'completed' or fr['status'] != 'completed_offline_online_context_flight_replay'
            or fm['recording_sha256'] != sha256(root/'result.json')
            or any(sha256(path) != expected for path, expected in files)):
        raise ValueError('Exact earlier online diagnostic and flight replay required')
    events, native, frames = (read(path) for path, _ in files[:3])
    with np.load(files[3][0], allow_pickle=False) as data:
        expected = data['intent'][:, 0].copy()
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'shadow_source_sha256': sha256('src/flybrain_drive/online_neural_shadow.py'),
        'late_body_source_sha256': sha256('src/flybrain_drive/online_neural_late_body.py')
        if args.late_body else None,
        'recording_sha256': sha256(root/'result.json'), 'flight_sha256': fr['arrays_sha256'],
        'scope': 'offline original inputs; current wall clock is not original receipt clock',
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'training_admissible': False, 'actor_deployed': False}
    try:
        if args.late_body:
            from flybrain_drive.online_neural_late_body import LateBodyNeuralShadow
            shadow = LateBodyNeuralShadow(config_path=args.config, training=args.training,
                                         opponent=args.opponent)
        else:
            shadow = OnlineNeuralShadow(config_path=args.config, training=args.training,
                                        opponent=args.opponent)
        identity = read(Path('configs/ac-virtual-body-v1.json'))['native_identity']
        history = PerViewBodyHistory(episode_id=sr['request_id'], native_identity=identity)
        pos, intents, records = 0, [], []
        for i, event in enumerate(events):
            file = Path(frames[i]['source_file'])
            if sha256(file) != frames[i]['sha256'] or event['metadata_sha256'] != frames[i]['sha256']:
                raise ValueError('Exact recorded camera metadata required')
            frame = read(file)
            while pos < event['native_rows_consumed']:
                row = native[pos]
                raw = base64.b64decode(row['authority']['prefix_base64'], validate=True)
                if hashlib.sha256(raw).hexdigest() != row['authority']['prefix_sha256']:
                    raise ValueError('Original native bytes required')
                history.ingest_native_physics(decode_physics_prefix(raw[:PHYSICS_PREFIX.size],
                                                                    row['copy']['second_copy_end']))
                pos += 1
            context = history.build(render_start_lowers=tuple(event['render_start_lower']),
                camera_receipt_time=event['camera_receipt'], decision_time=event['decision'])
            if args.late_body:
                # Numeric equivalence only: both synthetic phase clocks equal the
                # original decision. This is not a live latency measurement.
                event['visual_snapshot'] = event['decision']
                shadow.clock = lambda: event['decision']
                out = shadow.process_late(frame, event, lambda: context, file.parent)
            else:
                out = shadow.process(frame, event, context, file.parent)
            records.append({'index': i, 'status': out['status'], 'intent': out['intent'],
                            'processing_seconds': out['processing_seconds']})
            if out['intent'] is not None:
                intents.append(out['intent'])
        intents = np.asarray(intents)
        if intents.shape != expected.shape:
            raise ValueError('Original valid flight segment length changed')
        maximum = float(np.max(np.abs(intents-expected)))
        if maximum > 2e-6:
            raise ValueError('Neural shadow differs from frozen offline reference')
        write_json(run/'outputs.json', records)
        result.update(status='completed_neural_shadow_offline_original_replay', groups=len(events),
            intents=len(intents), maximum_intent_error=maximum, error_tolerance=2e-6,
            outputs_sha256=sha256(run/'outputs.json'), initialization=shadow.provenance,
            late_body_equal_clock_equivalence=args.late_body,
            original_image_hashes_at_capture=False, actual_live_execution=False,
            physical_sample_or_exposure_time_verified=False, optimizer_updates=0)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
