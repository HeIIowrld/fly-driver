"""Verify live body-ticker sources, visual mailbox ordering and output-time host ages."""

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.online_neural_late_body import output_host_freshness
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from scripts.audit_causal_copy_body_context import first_observed_packets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--body-audit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, body = Path(a.recording), Path(a.body_audit)
    read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
    sr, br = read(root/'result.json'), read(body/'result.json')
    source_manifest = read(root/'manifest.json')['config']
    timing_enabled = source_manifest.get('diagnostic_cuda_timing') is True
    bm = read(body/'manifest.json')['config']
    files = ((root/'body-ticker.json', sr['ticker_sha256']),
             (root/'native-copy-windows.json', sr['native_copy_sha256']),
             (root/'online-body-shadow.json', sr['shadow_sha256']),
             (root/'diagnostic-frames.json', sr['metadata_sha256']))
    if (sr['status'] != 'completed' or sr['training_admissible'] is not False
            or br['status'] != 'completed_online_body_shadow_audit'
            or bm['recording_sha256'] != sha256(root/'result.json')
            or any(sha256(path) != expected for path, expected in files)):
        raise ValueError('Exact completed recording and body audit required')
    ticks, native, visual, frames = (read(path) for path, _ in files)
    if len(ticks) != sr['ticker_events'] or len(visual) != sr['frames'] or not ticks:
        raise ValueError('All original live ticker and visual events required')
    packets = first_observed_packets(native)
    receipts = [x['first_end'] for x in packets]
    names = ('front', 'left', 'right', 'rear')
    for i, (event, frame_row) in enumerate(zip(visual, frames, strict=True)):
        source = Path(frame_row['source_file'])
        if (event['index'] != i or frame_row['index'] != i
                or sha256(source) != frame_row['sha256']
                or event['metadata_sha256'] != frame_row['sha256']):
            raise ValueError('Exact original frame metadata required')
        frame = read(source)
        if timing_enabled and (not isinstance(event['neural'].get('visual_cuda_event_span_ms'), (int, float))
                               or not np.isfinite(event['neural']['visual_cuda_event_span_ms'])
                               or event['neural']['visual_cuda_event_span_ms'] < 0):
            raise ValueError('Original finite vision CUDA event span required')
        camera_views = {view['view_id']: view for view in frame['views']}
        if set(camera_views) != set(names):
            raise ValueError('All four original views required')
        for j, name in enumerate(names):
            if sha256(source.parent/camera_views[name]['file']) != event['neural']['source_images_sha256'][j]:
                raise ValueError('Image differs from original neural-read hash')
    identity = read('configs/ac-virtual-body-v1.json')['native_identity']
    encoder = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json', native_identity=identity)
    previous_time, previous_receipt, previous_count, previous_frame = -1., None, 0, -1
    selections, held, times, intervals, views, fresh, replacements = [], [], [], [], [], 0, []
    for e in ticks:
        decision, started, finished, count = e['decision'], e['started'], e['finished'], e['native_rows_consumed']
        frame_id = e['visual_frame_index']
        if (not previous_time < decision <= started <= finished
                or not previous_count <= count <= len(native) or count < 1
                or not 0 <= frame_id < len(visual) or frame_id < previous_frame
                or native[count-1]['copy']['second_copy_end'] > decision):
            raise ValueError('Actual ticker chronology, native consumption and frame order required')
        if timing_enabled and (not decision <= e['body_encode_started'] <= e['body_encode_finished'] <= started
                or not isinstance(e['body_cuda_event_span_ms'], (int, float))
                or not np.isfinite(e['body_cuda_event_span_ms']) or e['body_cuda_event_span_ms'] < 0):
            raise ValueError('Exact finite body encoder and CUDA stream event timing required')
        last_consumed = native[count-1]['copy']['second_copy_end']
        packet_index = bisect_right(receipts, last_consumed)-1
        if packet_index <= 0:
            raise ValueError('First packet age cannot be assumed')
        packet = packets[packet_index]
        if (e['packet'] != packet['packet'] or e['body_receipt'] != packet['first_end']
                or e['held_body_receipt'] != previous_receipt):
            raise ValueError('Fresh original selected and previous held packet required')
        obs = encoder.encode(packet['sample'], episode_id=sr['request_id'], decision_monotonic_seconds=decision)
        if not obs.valid[[0, 1, 4, 16, 17, 18, 19, 20]].all():
            raise ValueError('Ticker accepted an invalid body')
        scene = visual[frame_id]
        n = scene['neural']
        if (n['status'] != 'visual_published' or e['visual_published'] != n['published']
                or not scene['context_ready'] <= n['published'] <= decision
                or e['vehicle_control'] is not False or not np.isfinite(e['intent']).all()):
            raise ValueError('Visual scene must be published before ticker decision; no control')
        if e['visual_replaced']:
            if frame_id <= previous_frame:
                raise ValueError('Visual scene cannot be silently re-applied')
            replacements.append(frame_id)
        elif frame_id != previous_frame:
            raise ValueError('Scene change must be explicitly applied')
        selected_age = finished-packet['first_end']
        held_age = None if previous_receipt is None else finished-previous_receipt
        source_ages = [finished-v for v in scene['render_start_lower']]
        np.testing.assert_allclose(source_ages, e['view_render_age_upper_at_output_seconds'], atol=1e-9, rtol=0)
        if (abs(selected_age-e['body_receipt_age_at_output_seconds']) > 1e-9
                or (held_age is None) != (e['held_body_receipt_age_at_output_seconds'] is None)
                or held_age is not None and abs(held_age-e['held_body_receipt_age_at_output_seconds']) > 1e-9
                or e['output_host_freshness_passed'] != output_host_freshness(has_intent=True,
                    selected_body_age=selected_age, held_body_age=held_age, render_ages=source_ages)):
            raise ValueError('Actual output-time source age or freshness differs')
        selections.append(selected_age)
        if held_age is not None:
            held.append(held_age)
        views.append(max(source_ages))
        times.append(finished-decision)
        fresh += e['output_host_freshness_passed']
        if previous_time >= 0:
            intervals.append(decision-previous_time)
        previous_time, previous_receipt, previous_count, previous_frame = decision, packet['first_end'], count, frame_id
    published = [i for i, e in enumerate(visual) if e['neural']['status'] == 'visual_published']
    if replacements != published[:len(replacements)]:
        raise ValueError('Published scenes were skipped or reordered by ticker')
    ms = lambda x: np.percentile(np.asarray(x)*1000, [50, 95, 100]).tolist()
    visual_windows = [(event['neural']['started'], event['neural']['vision_finished'])
                      for event in visual]
    overlap = [any(tick['decision'] < end and tick['finished'] > start
                   for start, end in visual_windows) for tick in ticks]
    overlap_times = [duration for duration, flag in zip(times, overlap, strict=True) if flag]
    separate_times = [duration for duration, flag in zip(times, overlap, strict=True) if not flag]
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_sha256': sha256(root/'result.json'), 'body_audit_sha256': sha256(body/'result.json'),
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    out = {'status': 'completed_multirate_live_shadow_audit', 'visual_groups': len(visual),
        'visual_scenes_published': len(published), 'visual_scenes_consumed': len(replacements),
        'last_published_not_consumed': len(published)-len(replacements),
        'ticker_events': len(ticks), 'ticker_error': sr['ticker_error'],
        'ticker_stop_reason': sr['ticker_stop_reason'], 'output_host_freshness_passed': fresh,
        'selected_body_age_ms_p50_p95_max': ms(selections), 'held_body_age_ms_p50_p95_max': ms(held),
        'selected_body_age_over_50ms': sum(x > .05 for x in selections),
        'held_body_age_over_50ms': sum(x > .05 for x in held),
        'view_age_upper_ms_p50_p95_max': ms(views), 'view_age_upper_over_150ms': sum(x > .15 for x in views),
        'decision_to_output_ms_p50_p95_max': ms(times), 'ticker_interval_ms_p50_p95_max': ms(intervals),
        'ticker_host_windows_overlapping_visual': sum(overlap),
        'overlap_decision_to_output_ms_p50_p95_max': ms(overlap_times) if overlap_times else None,
        'nonoverlap_decision_to_output_ms_p50_p95_max': ms(separate_times) if separate_times else None,
        'overlap_is_not_causal_gpu_attribution': True,
        'first_output_missing_prior_body': ticks[0]['held_body_receipt'] is None,
        'gpu_contention_included': True, 'physical_source_or_exposure_time_verified': False,
        'training_admissible': False, 'vehicle_control_sent': False}
    if timing_enabled:
        body_span = [tick['body_cuda_event_span_ms'] for tick in ticks]
        vision_span = [event['neural']['visual_cuda_event_span_ms'] for event in visual]
        out.update(cuda_event_span_is_not_kernel_occupancy=True,
            vision_cuda_event_span_ms_p50_p95_max=np.percentile(vision_span, [50, 95, 100]).tolist(),
            body_cuda_event_span_ms_p50_p95_max=np.percentile(body_span, [50, 95, 100]).tolist(),
            body_encoder_host_ms_p50_p95_max=ms([tick['body_encode_finished']-tick['body_encode_started']
                                                 for tick in ticks]),
            body_cuda_event_span_overlap_p50_p95_max=np.percentile(
                [value for value, flag in zip(body_span, overlap, strict=True) if flag],
                [50, 95, 100]).tolist() if any(overlap) else None,
            body_cuda_event_span_nonoverlap_p50_p95_max=np.percentile(
                [value for value, flag in zip(body_span, overlap, strict=True) if not flag],
                [50, 95, 100]).tolist() if not all(overlap) else None)
    write_json(run/'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
