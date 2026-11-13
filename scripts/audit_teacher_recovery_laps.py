"""Verify complete teacher laps from raw SDK/native/graphics and latched SDK validity.

The first finish crossing is partial. Native iLastTime, continuous spline travel,
SDK lap counters and LapInvalidated are checked separately. No teacher fitting.
"""
import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_clock import decode_graphics_timing
from flybrain_drive.ac_native_physics import NativePhysicsReader, PHYSICS_LAYOUT, PHYSICS_PREFIX
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.source_native_snapshot import WindowsQpc
from scripts.record_ac_wgc import physics_record


def decode_payload(row, data_key, hash_key):
    payload = base64.b64decode(row[data_key], validate=True)
    if hashlib.sha256(payload).hexdigest() != row[hash_key]:
        raise ValueError('Raw payload hash mismatch')
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--offline-only', action='store_true',
                   help='Audit recorded lap evidence without reading live AC or vJoy')
    a = p.parse_args()
    source = Path(a.recording)
    result = json.loads((source/'result.json').read_text(encoding='utf8'))
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
    run = begin_run(a.output, dict(arguments=vars(a), seed=780102,
        source_manifest_sha256=sha256(source/'manifest.json'),
        source_result_sha256=sha256(source/'result.json'), actuator_messages_sent=0))
    if (sha256(source/'events.jsonl') != result['events_sha256'] or
            sha256(source/'tcp-source-packets.jsonl') != result['tcp_source_archive_sha256']):
        raise ValueError('Source event/TCP archive hash differs')
    events = [json.loads(x) for x in (source/'events.jsonl').read_text(encoding='utf8').splitlines()]
    tcp = [json.loads(x) for x in (source/'tcp-source-packets.jsonl').read_text(encoding='utf8').splitlines()]
    sdk_rows = []
    for row in tcp:
        packet = json.loads(decode_payload(row, 'wire_base64', 'wire_sha256'))
        if packet['steps'] != row['sequence']:
            raise ValueError('Wire sequence differs')
        sdk_rows.append((row, packet))
    previous_lap = previous_source = previous_current_time = None
    previous_progress = None
    first_crossing = None
    crossings, laps = [], []
    travel = 0.
    tyres = 0
    max_tyres = 0
    source_native_mode = manifest['config']['arguments'].get('source_native', False)
    qpc_bridge, source_action_ages = None, []
    if source_native_mode:
        qpc = WindowsQpc()
        first = qpc.ticks()
        host = time.perf_counter()
        last = qpc.ticks()
        qpc_bridge = dict(first_ticks=first, last_ticks=last,
            frequency_hz=qpc.frequency, host_perf_counter=host)
        if not first/qpc.frequency-.00001 <= host <= last/qpc.frequency+.00001:
            raise ValueError('Audit host perf/QPC clock identity not within 10us')
    latest_source_snapshot = None
    latest_source_file_fence = False
    latest_file_io_counts = None
    teacher_file_io_counts = None
    source_intervals = []
    prior_source = None
    initial_native_fields = None
    maximum_speed_kmh = 0.
    for event in events:
        if source_native_mode and event['kind'] == 'teacher_request':
            if manifest['config']['arguments'].get('require_quiescent_file_bridge'):
                if not latest_source_file_fence:raise ValueError('Source file compatibility I/O fence missing at actual teacher request')
                if teacher_file_io_counts is None:teacher_file_io_counts=latest_file_io_counts
                if latest_file_io_counts!=teacher_file_io_counts:raise ValueError('Embedded compatibility file I/O advanced while teacher owned controls')
            if latest_source_snapshot is None:
                raise ValueError('Source-native teacher request without a preceding original pair')
            started = latest_source_snapshot['qpc_started_ticks']/qpc.frequency
            age = event['request']['driver_call_started_perf_counter_seconds']-started
            if not 0 <= age <= .05:
                raise ValueError('Original source-to-driver-call age exceeded 50ms')
            source_action_ages.append(age*1000)
        if event['kind'] == 'teacher_history_reset':
            prior_source = None
        if event['kind'] not in ('raw_pair', 'early_source_pair'):
            continue
        sdk = json.loads(decode_payload(event['sdk'], 'wire_base64', 'wire_sha256'))
        native_bytes = decode_payload(event['native'], 'prefix_base64', 'prefix_sha256')
        if source_native_mode:
            snapshot = sdk.get('source_native_snapshot', {})
            if (snapshot.get('protocol') != 'flybrain.ac.source_native.v1'
                    or snapshot.get('qpc_frequency_hz') != qpc.frequency
                    or not snapshot.get('repeated_copy_equal')
                    or snapshot.get('physical_tick_time_verified') is not False
                    or native_bytes != base64.b64decode(snapshot['prefix_base64'], validate=True)
                    or not 0 <= (snapshot['qpc_finished_ticks']-snapshot['qpc_started_ticks'])/qpc.frequency <= .005):
                raise ValueError('Source-native packet bytes/clock witness differ')
            latest_source_snapshot = snapshot
            latest_file_io_counts = {k:v['count'] for k,v in sdk.get('publisher_compatibility_completed_io_timing',{}).get('stages',{}).items()}
            latest_source_file_fence = (sdk.get('publisher_file_compatibility_suspended') is True
                and sdk.get('publisher_file_io_quiescent') is True)
        values = iter(PHYSICS_PREFIX.unpack(native_bytes))
        raw = {name:tuple(next(values) for _ in range(n)) for name, _, n in PHYSICS_LAYOUT}
        if initial_native_fields is None:
            initial_native_fields = event['native']['fields']
        maximum_speed_kmh = max(maximum_speed_kmh, abs(raw['speedKmh'][0]))
        tyres = max(tyres, raw['numberOfTyresOut'][0])
        max_tyres = max(max_tyres, raw['numberOfTyresOut'][0])
        if event['kind'] == 'early_source_pair':
            continue
        source_time = sdk['timestamp_ac']
        if prior_source is not None:
            source_intervals.append(source_time-prior_source)
        prior_source = source_time
        timing_bytes = decode_payload(event['graphics'], 'prefix_base64', 'prefix_sha256')
        timing = decode_graphics_timing(timing_bytes, event['graphics']['received_monotonic_seconds'])
        progress = sdk['privileged_teacher']['raw']['NormalizedSplinePosition']
        if previous_progress is not None:
            delta = progress-previous_progress
            if delta < -.5:
                delta += 1
            if not -.002 <= delta <= .02:
                raise ValueError('Spline discontinuity or backwards travel')
            travel += max(0, delta)
        initial_timed_start = (first_crossing is None and previous_current_time is not None
            and timing.completed_laps == previous_lap
            and timing.current_time_raw < previous_current_time-1000
            and (progress < .03 or previous_progress > .97))
        if initial_timed_start:
            # AC hotlap starts timing at the first finish line without increasing
            # completedLaps. Preserve this raw timer reset as the first anchor.
            first_crossing = dict(kind='initial_timed_lap_start',
                native_completed_laps=timing.completed_laps, sdk_lap=sdk['LapCount'],
                source_time=source_time, host_time=event['host_time'],
                last_lap_ms=timing.last_time_raw, current_lap_ms=timing.current_time_raw,
                progress=progress)
            crossings.append(first_crossing)
            travel, tyres = 0., raw['numberOfTyresOut'][0]
        elif previous_lap is not None and timing.completed_laps != previous_lap:
            if timing.completed_laps != previous_lap+1 or not (progress < .03 or previous_progress > .97):
                raise ValueError('Lap reset/skipped/away from finish')
            crossing = dict(kind='completed_lap', native_completed_laps=timing.completed_laps,
                sdk_lap=sdk['LapCount'], source_time=source_time,
                host_time=event['host_time'], last_lap_ms=timing.last_time_raw,
                current_lap_ms=timing.current_time_raw, progress=progress)
            crossings.append(crossing)
            if first_crossing is not None:
                lap_rows = [(receipt, packet) for receipt, packet in sdk_rows
                    if first_crossing['source_time'] <= packet['timestamp_ac'] < source_time]
                invalid_values = [packet['privileged_teacher']['raw'].get('LapInvalidated')
                    for _, packet in lap_rows]
                sdk_lap_values = sorted({packet['LapCount'] for _, packet in lap_rows})
                sdk_counter_agreement = (bool(lap_rows) and
                    sum(packet['LapCount'] == timing.completed_laps-1 for _, packet in lap_rows)
                    >= .95*len(lap_rows) and set(sdk_lap_values).issubset(
                        {timing.completed_laps-1, timing.completed_laps}))
                elapsed = source_time-first_crossing['source_time']
                evidence_ok = (.97 <= travel <= 1.03 and timing.last_time_raw > 0
                    and abs(elapsed-timing.last_time_raw/1000) <= .15 and sdk_counter_agreement)
                official_valid = bool(invalid_values) and all(v == 0 for v in invalid_values)
                laps.append(dict(native_completed_laps=timing.completed_laps,
                    lap_time_ms=timing.last_time_raw, source_finish_interval_seconds=elapsed,
                    integrated_spline_laps=travel, maximum_tyres_out=tyres,
                    sdk_source_packets=len(lap_rows), sdk_lap_values=sdk_lap_values,
                    sdk_counter_agreement=sdk_counter_agreement,
                    sdk_lap_invalidated_values=sorted(set(invalid_values), key=str),
                    full_lap_evidence_verified=evidence_ok,
                    sdk_official_validity_observed=official_valid,
                    zero_tyres_out=tyres == 0,
                    verified_valid_lap=bool(evidence_ok and official_valid and tyres <= 2)))
            first_crossing = crossing
            travel, tyres = 0., raw['numberOfTyresOut'][0]
        previous_lap, previous_progress, previous_source = timing.completed_laps, progress, source_time
        previous_current_time = timing.current_time_raw
    requests = [x for x in events if x['kind']=='teacher_request']
    closes = [x for x in events if x['kind']=='close']
    if requests and (not closes or closes[-1]['request']['requested_control'] !=
            dict(steering=0., throttle=0., brake=1.)):
        raise ValueError('Final neutral/full-brake close is absent')
    samples, ai, hb, stopped = [], None, None, None
    if not a.offline_only:
        axis = manifest['config']['axis_configuration']
        vjoy = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        samples = []
        with NativePhysicsReader(timeout=.15) as reader:
            deadline = time.perf_counter()+15
            while time.perf_counter() < deadline:
                sample = reader.receive()
                samples.append(physics_record(sample))
                if len(samples) >= 3 and all(abs(x['fields']['speedKmh']) < .1 and
                        x['fields']['gas'] < .02 and x['fields']['brake'] >= .98 for x in samples[-3:]):
                    break
                time.sleep(.05)
        native_ai = NativeTeacherStatusReader()
        try:
            ai = native_ai.snapshot()
        finally:
            native_ai.close()
        hb, modified = read_published_json(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
        stopped = (len(samples) >= 3 and len({x['packet_id'] for x in samples[-3:]}) == 3
            and all(abs(x['fields']['speedKmh']) < .1 and x['fields']['gas'] < .02 and
                    x['fields']['brake'] >= .98 for x in samples[-3:])
            and vjoy.GetVJDStatus(axis['device_id']) == 1 and not ai['builtin_ai_controlled']
            and not hb['ai_controlled'] and not hb['camera_active'] and not hb['expert_probe_active']
            and 0 <= time.time()-modified <= 2)
    raw_sources = np.array([packet['timestamp_ac'] for _, packet in sdk_rows])
    raw_receipts = np.array([receipt['received_monotonic_seconds'] for receipt, _ in sdk_rows])
    write_json(run/'result.json', dict(
        status='completed_offline_lap_audit' if a.offline_only else 'completed',
        verified_valid_laps=sum(x['verified_valid_lap'] for x in laps),
        laps=laps, crossings=crossings, maximum_tyres_out_all_raw=max_tyres,
        initial_native_fields=initial_native_fields, maximum_speed_kmh=maximum_speed_kmh,
        compatibility_file_io_counts_during_teacher=teacher_file_io_counts,
        source_native_pairs_verified=source_native_mode, qpc_bridge=qpc_bridge,
        source_to_driver_age_p50_p95_max_ms=np.percentile(source_action_ages,[50,95,100]).tolist()
            if source_action_ages else None,
        selected_source_interval_min_ms=min(source_intervals)*1000 if source_intervals else None,
        selected_source_interval_max_ms=max(source_intervals)*1000 if source_intervals else None,
        tcp_source_interval_max_ms=float(np.diff(raw_sources).max()*1000),
        tcp_receipt_interval_max_ms=float(np.diff(raw_receipts).max()*1000),
        teacher_requests=len(requests), request_interval_mean_ms=float(np.diff(
            [x['host_time'] for x in requests]).mean()*1000) if len(requests)>1 else None,
        final_neutral_close_verified=bool(closes), post_stop_verified=stopped,
        post_stop_native=None if a.offline_only else samples[-3:],
        post_stop_ai=ai, post_stop_session=hb,
        raw_payload_hashes_verified=True, actuator_messages_sent=0, ui_actions_sent=0,
        student_optimizer_updates=0))
    print(json.dumps(dict(laps=laps, post_stop_verified=stopped)))


if __name__ == '__main__':
    main()
