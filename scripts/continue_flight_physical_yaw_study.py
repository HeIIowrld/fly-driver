"""Finish all three predeclared physical blocks and full-history audits before evaluating."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verified_completed_prefix(previous, plan, block0):
    root = Path(previous)
    result, manifest = read(root/'result.json'), read(root/'manifest.json')
    if (result['status'] != 'failed' or result['full_study_complete']
            or manifest['config']['plan'] != plan):
        raise ValueError('Terminal failed study with the exact unchanged plan required')
    blocks = result['blocks']
    if not 1 <= len(blocks) < 3 or [b['index'] for b in blocks] != list(range(len(blocks))):
        raise ValueError('Complete contiguous audited block prefix required')
    if Path(blocks[0]['original']) != Path(block0):
        raise ValueError('Same original first block required')
    pairs = []
    for entry in blocks:
        index = entry['index']
        source, audit = Path(entry['original']), Path(entry['audit'])
        original, checked = read(source/'result.json'), read(audit/'result.json')
        if (sha256(audit/'result.json') != entry['audit_sha256']
                or checked['status'] != 'completed_physical_yaw_block_audit'
                or checked['block'] != index or len(checked['audits']) != 4
                or original['status'] != 'completed_physical_yaw_block'
                or original['block'] != index or len(original['trials']) != 4
                or read(source/'manifest.json')['config']['plan'] != plan
                or read(audit/'manifest.json')['config']['source_result_sha256']
                != sha256(source/'result.json')):
            raise ValueError('Preserved complete block/audit identity changed')
        for trial in original['trials']:
            if sha256(Path(trial['path'])/'result.json') != trial['result_sha256']:
                raise ValueError('Preserved actual trial changed')
        for trial in checked['audits']:
            if sha256(Path(trial['path'])/'result.json') != trial['result_sha256']:
                raise ValueError('Preserved independent trial audit changed')
        pairs.extend(checked['pairs'])
    if result['pairs'] != pairs or len(pairs) != 2*len(blocks):
        raise ValueError('All original pairs, including adverse outcomes, must be retained')
    failed = root/f'block-{len(blocks)}'
    failure = read(failed/'result.json')
    if failure['status'] != 'failed' or failure['trials']:
        raise ValueError('This recovery only covers failed preparation with zero actor trials')
    return dict(blocks=blocks, pairs=pairs, provenance=dict(path=str(root),
        result_sha256=sha256(root/'result.json'), failed_block_result_sha256=sha256(
            failed/'result.json'), preserved_blocks=len(blocks),
        preserved_actual_trials=4*len(blocks), outcome_selection=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block0', required=True)
    p.add_argument('--completed-prefix-block1',
                   help='Terminal failed block1: keep completed independent trials only')
    p.add_argument('--completed-prefix-study',
                   help='Keep every completed audited block of a terminal failed study')
    p.add_argument('--plan', required=True)
    p.add_argument('--site-packages', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if not a.execute:
        raise ValueError('Explicit execute required for remaining predeclared physical trials')
    plan = read(a.plan)
    expected_sha = sha256(a.plan)
    first = read(Path(a.block0) / 'result.json')
    assert first['status'] == 'completed_physical_yaw_block' and first['block'] == 0
    assert read(Path(a.block0) / 'manifest.json')['config']['plan'] == plan
    run = begin_run(a.output, dict(arguments=vars(a), plan=plan, plan_sha256=expected_sha,
        existing_block0_result_sha256=sha256(Path(a.block0) / 'result.json')))
    result = dict(status='failed', goal_achieved=False, full_study_complete=False,
                  physical_yaw_disturbance_iae_gate_passed=False, blocks=[], pairs=[])
    if a.completed_prefix_study:
        if a.completed_prefix_block1:
            raise ValueError('Choose one explicit completed-prefix recovery scope')
        prefix = verified_completed_prefix(a.completed_prefix_study, plan, a.block0)
        result.update(blocks=list(prefix['blocks']), pairs=list(prefix['pairs']),
                      preserved_failed_study=prefix['provenance'])

    def stage(name, module, options):
        assert sha256(a.plan) == expected_sha
        output = run / name
        with (run / (name+'.log')).open('x', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-m', module, *options,
                '--site-packages', a.site_packages, '--output', str(output)],
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(run / 'progress.json', dict(stage=name, pid=child.pid,
                                                  completed_blocks=result['blocks']))
            print(json.dumps(dict(stage=name, pid=child.pid)), flush=True)
            child.wait()
        receipt = read(output / 'result.json')
        if child.returncode:
            raise RuntimeError('Owned study stage failed: '+name+'; '+str(receipt))
        return output, receipt

    try:
        for block_index in range(3):
            if block_index < len(result['blocks']):
                continue
            if block_index == 0:
                original = Path(a.block0)
            else:
                prefix = (['--completed-prefix-block', a.completed_prefix_block1]
                          if block_index == 1 and a.completed_prefix_block1 else [])
                original, receipt = stage(f'block-{block_index}',
                    'scripts.run_flight_physical_yaw_block',
                    ['--plan', a.plan, '--block', str(block_index), '--execute', *prefix])
                assert receipt['status'] == 'completed_physical_yaw_block'
            audit, receipt = stage(f'audit-{block_index}',
                'scripts.audit_flight_physical_yaw_block', ['--block', str(original)])
            assert receipt['status'] == 'completed_physical_yaw_block_audit'
            result['blocks'].append(dict(index=block_index, original=str(original),
                                         audit=str(audit),
                                         audit_sha256=sha256(audit / 'result.json')))
            result['pairs'].extend(receipt['pairs'])
            write_json(run / 'partial-result.json', result)
        assert len(result['pairs']) == 6
        valid = all(pair['pair_valid'] for pair in result['pairs'])
        fractions = [pair['iae_improvement_fraction'] for pair in result['pairs']]
        positive = sum(value is not None and value > 0 for value in fractions)
        median = float(np.median(fractions)) if all(v is not None for v in fractions) else None
        spec = plan['evaluation']
        passed = (valid and positive >= spec['required_positive_pairs'] and median is not None
                  and median >= spec['minimum_median_iae_improvement_fraction']
                  and len(result['blocks']) == spec['required_blocks'])
        result.update(status='completed_physical_yaw_study', full_study_complete=True,
            all_pairs_valid=valid, positive_iae_pairs=positive,
            median_iae_improvement_fraction=median,
            physical_yaw_disturbance_iae_gate_passed=passed,
            scope='three blocks, both signed small pulses at6kmh with engineering pedals only',
            navigation_verified=False, physiological_equivalence_verified=False,
            learned_longitudinal_control_verified=False, actor_qualified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
