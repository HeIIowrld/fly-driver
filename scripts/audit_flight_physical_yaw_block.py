"""Full original replay of all four trials followed by predeclared matched-pair metrics."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.owned_worker_wait import wait_owned_worker


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block', required=True)
    p.add_argument('--site-packages', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source = Path(a.block)
    block = read(source / 'result.json')
    plan = read(source / 'manifest.json')['config']['plan']
    assert block['status'] == 'completed_physical_yaw_block' and len(block['trials']) == 4
    run = begin_run(a.output, dict(arguments=vars(a), plan=plan,
        source_result_sha256=sha256(source / 'result.json'),
        source_manifest_sha256=sha256(source / 'manifest.json'), actuator_messages_sent=0,
        additional_pulse_width_match_seconds=.01,
        timing_match_declared_before_first_completed_zero_yaw_trial=True,
        timing_amendment_basis='first full pulse324ms; no completed zero-yaw pulse yet'))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
                  full_study_complete=False, audits=[], pairs=[])
    try:
        bootstrap = ('import sys,site,runpy; p=sys.argv.pop(1); site.addsitedir(p); '
                     'sys.path.insert(0,p); '
                     'runpy.run_module("scripts.audit_flight_physical_yaw_trial",'
                     'run_name="__main__")')
        audited = []
        for entry in block['trials']:
            assert sha256(Path(entry['path']) / 'result.json') == entry['result_sha256']
            index = entry['index']
            output = run / f'trial-{index:02d}'
            with (run / f'trial-{index:02d}.log').open('x', encoding='utf8') as log:
                child = subprocess.Popen([sys.executable, '-c', bootstrap, a.site_packages,
                    '--recording', entry['path'], '--output', str(output)],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(run / 'progress.json', dict(index=index, pid=child.pid,
                                                      completed_audits=result['audits']))
                print(json.dumps(dict(index=index, pid=child.pid)), flush=True)
                wait_owned_worker(child, module='scripts.audit_flight_physical_yaw_trial',
                    result_file=output/'result.json',
                    receipt_file=run/f'trial-{index:02d}-process.json')
            receipt = read(output / 'result.json')
            assert (child.returncode == 0
                    and receipt['status'] == 'completed_physical_yaw_trial_audit')
            assert receipt['trial'] == entry['condition']
            audited.append(receipt)
            result['audits'].append(dict(index=index, path=str(output),
                                        result_sha256=sha256(output / 'result.json')))
        spec = plan['evaluation']
        for sign in (1, -1):
            full = next(r for r in audited if r['trial']['mode'] == 'full'
                        and r['trial']['sign'] == sign)
            zero = next(r for r in audited if r['trial']['mode'] == 'zero_yaw'
                        and r['trial']['sign'] == sign)
            position = float(np.linalg.norm(np.array(full['initial_position'])
                                            - zero['initial_position']))
            speed = abs(full['baseline_speed_kmh']-zero['baseline_speed_kmh'])
            yaw = abs(full['physical_metrics']['baseline_yaw_left_rad_s']
                      - zero['physical_metrics']['baseline_yaw_left_rad_s'])
            width_difference = abs((full['pulse_off_host_clock']-full['pulse_on_host_clock'])
                                   - (zero['pulse_off_host_clock']-zero['pulse_on_host_clock']))
            checks = dict(start_position=position <= spec['pair_start_position_tolerance_m'],
                baseline_speed=speed <= spec['pair_baseline_speed_tolerance_kmh'],
                baseline_yaw=yaw <= spec['pair_baseline_yaw_tolerance_rad_s'],
                pulse_width_match=width_difference <= .01,
                individual_timing=(full['pulse_timing_passed'] and zero['pulse_timing_passed']
                                   if plan['schema'].endswith('.v2') else True),
                unclipped_pulse=not full['pulse_clipped'] and not zero['pulse_clipped'],
                full_history=full['intervals_above_80ms'] == zero['intervals_above_80ms'] == 0,
                nonzero_reference_response=zero['physical_metrics']['iae_rad'] > 1e-12)
            improvement = (1-full['physical_metrics']['iae_rad']/zero['physical_metrics']['iae_rad']
                           if checks['nonzero_reference_response'] else None)
            result['pairs'].append(dict(sign=sign, block=block['block'],
                checks=checks, pair_valid=all(checks.values()),
                start_position_difference_m=position, baseline_speed_difference_kmh=speed,
                baseline_yaw_difference_rad_s=yaw, iae_improvement_fraction=improvement,
                pulse_width_difference_seconds=width_difference,
                full_metrics=full['physical_metrics'], zero_yaw_metrics=zero['physical_metrics']))
        result.update(status='completed_physical_yaw_block_audit', block=block['block'],
                      all_pairs_valid=all(p['pair_valid'] for p in result['pairs']))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
