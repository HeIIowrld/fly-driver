"""Independently audit retained MMF pixels and original live body-ticker receipts."""

import argparse
import hashlib
import io
import json
import zipfile
from bisect import bisect_right
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.online_neural_late_body import output_host_freshness
from scripts.audit_causal_copy_body_context import first_observed_packets


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.recording)
    result = read(root/'result.json')
    paths = [('native-copy-windows.json', 'native_sha256'), ('bridge-events.json', 'bridge_sha256'),
             ('body-ticker.json', 'ticker_sha256')]
    if any(sha256(root/name) != result.get(key) for name, key in paths):
        raise ValueError('Exact original live shadow evidence required')
    bundle = read(root/'vision/bundle.json')
    archive = root/'vision'/bundle['file']
    if (sha256(archive) != bundle['sha256']
            or bundle['sha256'] != result['archive']['bundle_sha256']):
        raise ValueError('Exact retained camera archive required')
    native, bridge, ticks = (read(root/name) for name, _ in paths)
    packets = first_observed_packets(native)
    receipts = [p['first_end'] for p in packets]
    published = {}
    with zipfile.ZipFile(archive) as z:
        for row in bridge:
            n = row.get('neural', {})
            if n.get('status') != 'visual_published':
                continue
            frame = row['capture_id']
            if frame in published or n.get('source_images_hash_kind') != 'immutable_rgb':
                raise ValueError('One immutable publication per capture required')
            for i, name in enumerate(VIEW_ORDER):
                raw = z.read(f'{frame:04d}-{name}.bmp')
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = np.asarray(image.convert('RGB'))
                if hashlib.sha256(rgb.tobytes()).hexdigest() != n['source_images_sha256'][i]:
                    raise ValueError('Archived MMF RGB differs from actual neural input')
            published[frame] = row
    last_time, previous_receipt, verified = -1., None, []
    for e in ticks:
        decision, finished, count = e['decision'], e['finished'], e['native_rows_consumed']
        if not last_time < decision <= e['started'] <= finished or not 1 <= count <= len(native):
            raise ValueError('Ticker time/order/native count differs')
        last_copy = native[count-1]['copy']['second_copy_end']
        packet_idx = bisect_right(receipts, last_copy)-1
        if packet_idx <= 0 or last_copy > decision:
            raise ValueError('Native source is missing or from the future')
        packet = packets[packet_idx]
        scene = published[e['visual_frame_index']]
        if (e['packet'] != packet['packet'] or e['body_receipt'] != packet['first_end']
                or e['held_body_receipt'] != previous_receipt
                or e['visual_published'] != scene['neural']['published']
                or e['visual_published'] > decision or e['vehicle_control'] is not False):
            raise ValueError('Original selected/held packet or visual publication differs')
        selected = finished-packet['first_end']
        held = None if previous_receipt is None else finished-previous_receipt
        views = [finished-x for x in scene['visual']['render_start_lower']]
        fresh = output_host_freshness(has_intent=True, selected_body_age=selected,
                                      held_body_age=held, render_ages=views)
        if (abs(selected-e['body_receipt_age_at_output_seconds']) > 1e-9
                or (held is None) != (e['held_body_receipt_age_at_output_seconds'] is None)
                or held is not None
                and abs(held-e['held_body_receipt_age_at_output_seconds']) > 1e-9
                or not np.allclose(views, e['view_render_age_upper_at_output_seconds'],
                                   atol=1e-9, rtol=0)
                or fresh != e['output_host_freshness_passed']):
            raise ValueError('Original output-time freshness differs')
        verified.append({'fresh': fresh, 'selected_age_ms': selected*1000,
                         'held_age_ms': None if held is None else held*1000,
                         'view_age_upper_ms': max(views)*1000,
                         'decision_to_output_ms': (finished-decision)*1000})
        last_time, previous_receipt = decision, packet['first_end']
    if not verified:
        raise ValueError('No actual body output to audit')
    run = begin_run(a.output, {'recording': str(root), 'result_sha256': sha256(root/'result.json'),
        'program_sha256': sha256(__file__), 'archive_sha256': bundle['sha256'], 'seed': None})
    write_json(run/'ticks.json', verified)
    held = [x['held_age_ms'] for x in verified if x['held_age_ms'] is not None]
    summary = {'status': 'completed_mmf_immutable_shadow_audit', 'ticks': len(verified),
        'host_freshness_passed': sum(x['fresh'] for x in verified),
        'held_over_50ms': sum(x > 50 for x in held),
        'view_over_150ms': sum(x['view_age_upper_ms'] > 150 for x in verified),
        'published_visual_groups': len(published),
        'held_ms_p50_p95_max': np.percentile(held, [50, 95, 100]).tolist(),
        'decision_ms_p50_p95_max': np.percentile(
            [x['decision_to_output_ms'] for x in verified], [50, 95, 100]).tolist(),
        'ticks_sha256': sha256(run/'ticks.json'), 'physical_source_time_verified': False,
        'training_admissible': False, 'vehicle_control_sent': False}
    write_json(run/'result.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
