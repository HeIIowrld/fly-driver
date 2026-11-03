"""Audit sealed partial camera and shadow evidence; no simulator or control access."""

import argparse
import hashlib
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def analyze(recording, camera):
    recording, camera = Path(recording), Path(camera)
    result = json.loads((recording / 'result.json').read_text(encoding='utf8'))
    terminal = json.loads((camera / 'result.json').read_text(encoding='utf8'))
    if result['request_id'] != terminal['request_id'] or terminal['status'] != 'stopped':
        raise ValueError('Original stopped camera and recording identity required')
    frames, hashes = [], []
    for i in range(terminal['frames']):
        path = camera / f'frame-{i:04d}.json'
        raw = path.read_bytes()
        frame = json.loads(raw)
        if frame['request_id'] != result['request_id'] or frame['index'] != i:
            raise ValueError('Original ordered frame identity differs')
        hashes.append(hashlib.sha256(raw).hexdigest())
        frames.append(frame)
    if len(frames) < 2:
        raise ValueError('At least two recorded frames required')
    gaps = sorted(({'from': a['index'], 'to': b['index'],
                    'seconds': b['script_clock'] - a['script_clock']}
                   for a, b in zip(frames, frames[1:])),
                  key=lambda row: row['seconds'], reverse=True)
    saves = []
    for frame in frames:
        for view in frame['views']:
            duration = view['save_finished_clock'] - view['render_call_finished_clock']
            if duration < 0:
                raise ValueError('Nonmonotonic original render and save clocks')
            saves.append({'frame': frame['index'], 'view': view['view_id'], 'seconds': duration})
    saves.sort(key=lambda row: row['seconds'], reverse=True)
    durations = sorted(row['seconds'] for row in saves)
    shadow_path, ticker_path = recording / 'online-body-shadow.json', recording / 'body-ticker.json'
    if sha256(shadow_path) != result['shadow_sha256'] or sha256(ticker_path) != result['ticker_sha256']:
        raise ValueError('Sealed original shadow or ticker hash differs')
    shadow = json.loads(shadow_path.read_text(encoding='utf8'))
    ticker = json.loads(ticker_path.read_text(encoding='utf8'))
    receipt = json.loads((recording / 'control/api-receipt.json').read_text(encoding='utf8'))
    if not shadow or not ticker:
        raise ValueError('Partial original shadow and ticker events required')
    return {'status': 'completed_partial_capture_audit', 'request_id': result['request_id'],
            'camera_frames': len(frames), 'frame_metadata_sha256': hashes,
            'camera_frame_hashes_sealed_at_capture': False,
            'camera_terminal_sha256': sha256(camera / 'result.json'),
            'recording_result_sha256': sha256(recording / 'result.json'),
            'shadow_sha256': result['shadow_sha256'], 'ticker_sha256': result['ticker_sha256'],
            'receipt_sha256': sha256(recording / 'control/api-receipt.json'),
            'largest_camera_script_clock_gaps': gaps[:10],
            'largest_render_end_to_save_end': saves[:10],
            'save_duration_median_seconds': durations[len(durations)//2],
            'last_ticker': ticker[-1], 'last_shadow': shadow[-1],
            'control_status': receipt['status'], 'control_stop_reason': receipt.get('stop_reason'),
            'api_cleanup_clock': next((x['script_clock'] for x in receipt.get('api_calls', [])
                                       if x['name'] == 'disable_ai'), None),
            'causality_verified': False, 'physical_sample_or_exposure_time_verified': False,
            'training_admissible': False, 'vehicle_control_sent': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--camera', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.recording, args.camera)
    run = begin_run(args.output, {'recording': args.recording, 'camera': args.camera,
                                  'program_sha256': sha256(__file__)})
    write_json(run / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'camera_frames', 'control_status',
          'control_stop_reason', 'save_duration_median_seconds',
          'largest_camera_script_clock_gaps', 'largest_render_end_to_save_end')}, indent=2))


if __name__ == '__main__':
    main()
