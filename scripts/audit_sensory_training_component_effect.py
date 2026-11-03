"""Attribute an offline candidate change using geometry-zero graph replays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'original', 'candidate', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    old_result = json.loads((args.original / 'result.json').read_text(encoding='utf8'))
    new_result = json.loads((args.candidate / 'result.json').read_text(encoding='utf8'))
    if any(r['status'] != 'completed_low_speed_unified_graph_diagnostic'
           or r.get('neutral_sensory_command') is not True for r in (old_result, new_result)):
        raise ValueError('Original completed neutral-sensory replay receipts required')
    event_file = args.probe / 'events.jsonl'
    event_hash = digest(event_file)
    configs = []
    for root in (args.original, args.candidate):
        config = json.loads((root / 'manifest.json').read_text(encoding='utf8'))['config']
        if config['source_events_sha256'] != event_hash:
            raise ValueError('Replays must use exactly the same original events')
        configs.append(config)
    for key in ('visual_sha256', 'body_audit_sha256', 'host_audit_sha256',
                'identity_source_sha256', 'start_index', 'command', 'geometry_hold'):
        if configs[0][key] != configs[1][key]:
            raise ValueError('Replay source or condition differs: ' + key)
    programs = (Path('scripts/replay_low_speed_unified_sensory_graph.py'),
                Path('scripts/replay_low_speed_sensory_synthetic_candidate.py'))
    if any(digest(program) != config['program_sha256'] for program, config in zip(programs, configs, strict=True)):
        raise ValueError('Replay program differs from the recorded source')
    old_path, new_path = args.original / 'intents.npz', args.candidate / 'intents.npz'
    if digest(old_path) != old_result['intents_sha256'] or digest(new_path) != new_result['intents_sha256']:
        raise ValueError('Original replay array hash mismatch')
    events = [r for r in map(json.loads, event_file.read_text(encoding='utf8').splitlines()) if r['kind'] == 'native']
    start = old_result['original_start_index']
    if start != new_result['original_start_index']:
        raise ValueError('Same starting decision required')
    with np.load(old_path, allow_pickle=False) as original, np.load(new_path, allow_pickle=False) as candidate:
        if (not np.array_equal(original['clocks'], candidate['clocks'])
                or not np.array_equal(original['ready'], candidate['ready'])
                or len(events) - start != len(original['clocks'])):
            raise ValueError('Same decision clocks, readiness and complete history required')
        native = events[start:]
        steer = np.array([r['native']['fields']['steerAngle'] for r in native])
        speed = np.array([r['native']['fields']['speedKmh'] for r in native])
        old, new = original['intents'], candidate['intents']
        groups = {'negative_pulse': (steer < -.08) & (speed > 3),
                  'positive_pulse': (steer > .08) & (speed > 3),
                  'stationary': speed < .1}
        result_groups = {}
        for name, mask in groups.items():
            if not mask.any():
                raise ValueError('All requested observed groups must be present')
            result_groups[name] = dict(
                count=int(mask.sum()),
                original_intact_turn=float(old[0, mask, 0].mean()),
                original_geometry_zero_turn=float(old[1, mask, 0].mean()),
                candidate_intact_turn=float(new[0, mask, 0].mean()),
                candidate_geometry_zero_turn=float(new[1, mask, 0].mean()),
                geometry_zero_candidate_minus_original=float((new[1, mask, 0] - old[1, mask, 0]).mean()),
                candidate_geometry_intact_minus_zero=float((new[0, mask, 0] - new[1, mask, 0]).mean()))
    result = dict(status='completed_offline_component_comparison', events_sha256=event_hash,
                  original_sha256=digest(old_path), candidate_sha256=digest(new_path), groups=result_groups,
                  interpretation='Geometry-zero conditions differ by the trained visual bias under the pinned replay setup; '
                  'intact-minus-zero also includes recurrent effects. Steering-pulse groups are diagnostic, not optimal labels.',
                  physical_time_verified=False, physiology_verified=False, training_performed=False,
                  vehicle_control_sent=False, actor_deployed=False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
