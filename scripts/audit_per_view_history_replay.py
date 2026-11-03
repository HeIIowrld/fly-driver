"""Compare online-style receipt history with the independently audited diagnostic."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.per_view_body_history import PerViewBodyHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'body', 'availability', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, body, availability = map(Path, (args.recording, args.body, args.availability))
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    sr, br, ar = (read(x/'result.json') for x in (root, body, availability))
    am = read(availability/'manifest.json')['config']
    paths = ((root/'native-copy-windows.json', sr['native_copy_sha256']),
             (body/'view-receipts.json', br['view_receipts_sha256']),
             (availability/'receipts.json', ar['receipts_sha256']),
             (root/'render-copy-brackets.json', sr['render_copy_brackets_sha256']))
    if (any(sha256(path) != expected for path, expected in paths)
            or am['recording_result_sha256'] != sha256(root/'result.json')
            or ar['status'] != 'completed_availability_body_diagnostic'
            or ar['training_admissible'] is not False):
        raise ValueError('Exact original diagnostic and audit lineage required')
    native, views, groups, brackets = (read(path) for path, _ in paths)
    if len(views) != len(groups)*4:
        raise ValueError('Four original view receipts per group required')
    identity = read(Path('configs/ac-virtual-body-v1.json'))['native_identity']
    history = PerViewBodyHistory(episode_id='diagnostic', native_identity=identity)
    pos, comparisons, missing, masked = 0, 0, 0, 0
    for i, group in enumerate(groups):
        lo, hi = group['availability_host_bounds']
        while pos < len(native) and native[pos]['copy']['second_copy_end'] < lo:
            row = native[pos]
            raw = base64.b64decode(row['authority']['prefix_base64'], validate=True)
            timestamp = row['copy']['second_copy_end']
            if (hashlib.sha256(raw).hexdigest() != row['authority']['prefix_sha256']
                    or timestamp != row['authority']['received_monotonic_seconds']):
                raise ValueError('Original copy hash and host time required')
            sample = decode_physics_prefix(raw[:PHYSICS_PREFIX.size], timestamp)
            if sample.packet_id != row['authority']['packet_id']:
                raise ValueError('Original packet identity required')
            history.ingest_native_physics(sample)
            pos += 1
        source_views = views[4*i:4*i+4]
        bounds = tuple(b['render_host_bounds'][0] for b in brackets[4*i:4*i+4])
        ctx = history.build(render_start_lowers=bounds, camera_receipt_time=hi, decision_time=hi)
        if group['status'] == 'no_observed_prior_transition':
            if ctx is not None:
                raise ValueError('Initial unknown packet must not become current body')
            missing += 1
            continue
        if ctx is None or ctx.decision_body.packet_id != group['packet_id']:
            raise ValueError('Current availability packet differs from independent audit')
        for j, observation in enumerate(ctx.visual_contexts):
            old = source_views[j]
            expected = old.get('packet_id')
            if (None if observation is None else observation.packet_id) != expected:
                raise ValueError('Per-view prior packet differs from independent audit')
            if observation is not None and not observation.valid[4]:
                masked += 1
            comparisons += 1
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'history_source_sha256': sha256('src/flybrain_drive/per_view_body_history.py'),
        'recording_result_sha256': sha256(root/'result.json'),
        'body_receipts_sha256': br['view_receipts_sha256'],
        'availability_receipts_sha256': ar['receipts_sha256'],
        'input_cutoff': 'original group host availability lower bound',
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'completed_online_history_diagnostic_replay', 'groups': len(groups),
        'missing_current_body_groups': missing, 'valid_current_body_groups': len(groups)-missing,
        'view_packet_matches': comparisons, 'view_yaw_masked_at_camera_receipt': masked,
        'original_native_copies_ingested': pos, 'physical_or_exposure_time_verified': False,
        'actual_live_actor_executed': False, 'training_admissible': False, 'optimizer_updates': 0}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
