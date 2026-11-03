"""Audit all prospectively assigned spatial histories; retain roles and reject whole gaps."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.spatial_teacher_plan import validate_lease_route


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--partial-diagnostic', action='store_true',
                        help='Audit completed originals of a failed batch; never approve fitting')
    args = parser.parse_args()
    source = Path(args.collection)
    collection = read(source/'result.json')
    complete = (collection['status'] == 'completed_spatial_teacher_collection'
                and collection['completed_counts'] == dict(train=3, development=1))
    if not complete and not (args.partial_diagnostic and collection['status'] == 'failed'):
        raise ValueError('Complete declared three-train/one-development geographic batch required')
    candidates = [e for e in collection['episodes'] if e['lease']['role'] != 'transit']
    counts = {role: sum(e['lease']['role'] == role for e in candidates)
              for role in ('train', 'development')}
    if (not candidates or counts != collection['completed_counts']
            or complete and len(candidates) != 4):
        raise ValueError('Every candidate and original role required; no favourable subset')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        program_sha256=sha256(__file__), collection_result_sha256=sha256(source/'result.json'),
        collection_manifest_sha256=sha256(source/'manifest.json'),
        evaluation_access=False, no_fitting=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', episodes=[], fit_allowed=False, student_deployed=False,
                  labels_created=False, goal_achieved=False)
    try:
        for entry in candidates:
            root = Path(entry['path'])
            original = read(root/'result.json')
            assignment = read(source/(root.name+'-assignment.json'))
            if (sha256(root/'result.json') != entry['result_sha256']
                    or assignment['lease'] != entry['lease']
                    or original['status'] != 'completed' or original['camera_wait_misses'] != 0):
                raise ValueError('Unchanged original and prospective assignment required')
            receipt = read(root/'control/api-receipt.json')
            if validate_lease_route(assignment['lease'], receipt) != entry['admission']:
                raise ValueError('Original whole-history geographic admission differs')
            output = run/root.name
            command = [sys.executable, '-m', 'scripts.audit_ac_engine_ai_probe',
                       '--recording', str(root), '--require-aid-off', '--output', str(output)]
            with (run/(root.name+'.log')).open('x', encoding='utf8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(run/'progress.json', dict(episode=root.name, pid=child.pid, active=True))
                print(json.dumps(dict(episode=root.name, audit_pid=child.pid)), flush=True)
                code = child.wait()
            audited = read(output/'result.json')
            if (code or audited['status'] != 'completed'
                    or any(audited['actor_replay_max_errors'].values())
                    or audited['decisions'] != original['decisions']
                    or audited['intervals_above_80ms'] != 0
                    or audited['raw_mmf_slots_and_original_metadata_verified'] is not True
                    or audited['camera_aid_state_verified'] is not True):
                raise ValueError('Whole raw replay, MMF slots and gap qualification required')
            driving = [r for r in receipt['telemetry'] if r['phase'] == 'driving'
                       and r['ai_controlled'] and not r['paused'] and r['speed_kmh'] > 1]
            late = [r for r in driving if r['script_clock'] > driving[0]['script_clock']+3]
            result['episodes'].append(dict(path=str(root), role=entry['lease']['role'],
                raw_audit=str(output), raw_audit_manifest_sha256=sha256(output/'manifest.json'),
                raw_audit_result_sha256=sha256(output/'result.json'),
                original_result_sha256=entry['result_sha256'],
                admission=entry['admission'], observations=audited['decisions'],
                maximum_observation_interval_seconds=audited['maximum_decision_interval_seconds'],
                csp_reported_steer_min=min(r['steer_degrees'] for r in driving),
                csp_reported_steer_max=max(r['steer_degrees'] for r in driving),
                late_driving_positive_brake_samples=sum(r['brake'] > .01 for r in late),
                late_driving_samples=len(late),
                steering_unit_or_action_codec_not_qualified_by_this_audit=True))
            write_json(run/'partial-result.json', result)
        result.update(status='completed_whole_spatial_history_raw_audit' if complete else
                      'completed_partial_spatial_history_raw_audit',
                      original_collection_task_passed=complete,
                      observations=sum(e['observations'] for e in result['episodes']),
                      no_temporal_slicing=True, roles_unchanged=True,
                      visual_auxiliary_targets_not_created=True)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    finally:
        write_json(run/'progress.json', dict(active=False, status=result['status']))
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
