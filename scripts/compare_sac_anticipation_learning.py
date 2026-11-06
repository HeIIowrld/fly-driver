"""Compare completed, audited fixed-data credit-horizon learning arms.

Consumed development metrics only. The gate never qualifies live actor use.
"""
import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['control', 'candidate', 'control-audit', 'candidate-audit', 'trace', 'output']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    roots = [Path(a.control), Path(a.candidate)]
    results = [read(root / 'result.json') for root in roots]
    configs = [read(root / 'manifest.json')['config'] for root in roots]
    audits = [read(Path(path) / 'result.json') for path in [a.control_audit, a.candidate_audit]]
    trace = read(Path(a.trace) / 'result.json')
    run = begin_run(a.output, dict(arguments=vars(a), seed=780310, program_sha256=sha256(__file__),
        fit_result_sha256=[sha256(root / 'result.json') for root in roots],
        audit_result_sha256=[sha256(Path(path) / 'result.json') for path in [a.control_audit, a.candidate_audit]],
        gate='2% lower normal weighted MSE than both continuation and original; every axis RMSE <=1.02 x continuation',
        actor_deployed=False, optimizer_updates=0))
    report = dict(status='failed', actor_qualified=False, vehicle_control_sent=False, optimizer_updates=0)
    try:
        for key in ['program_sha256', 'checkpoint_sha256', 'split_sha256', 'episodes',
                    'sealed_learning_plan_sha256', 'fixed_fit', 'seed']:
            if configs[0][key] != configs[1][key]:
                raise ValueError('Paired experiment differs: ' + key)
        for i, arm in enumerate(['continuation', 'credit32']):
            if (results[i]['status'] != 'completed_fixed_anticipation_learning'
                    or list(results[i]['arms']) != [arm]
                    or audits[i]['status'] != 'completed_anticipation_evidence_audit'
                    or audits[i]['fit_updates'] != results[i]['optimizer_updates']
                    or not results[i]['original_checkpoint_unchanged']
                    or sha256(roots[i] / f'{arm}.pt') != results[i]['arms'][arm]['checkpoint_sha256']):
                raise ValueError('Completed independently audited paired arms required')
        original = trace['metrics'][2]
        control, candidate = results[0]['arms']['continuation']['dev'], results[1]['arms']['credit32']['dev']
        gate = (candidate['weighted_mse'] <= .98 * min(original['weighted_mse'], control['weighted_mse'])
                and np.all(np.asarray(candidate['rmse']) <= 1.02 * np.asarray(control['rmse'])))
        report.update(status='completed_fixed_credit_learning_comparison',
            original=original, continuation8=control, credit32=candidate,
            candidate_weighted_mse_change_from_original_percent=100*(candidate['weighted_mse']/original['weighted_mse']-1),
            candidate_weighted_mse_change_from_continuation_percent=100*(candidate['weighted_mse']/control['weighted_mse']-1),
            candidate_axis_rmse_change_from_continuation_percent=(100*(np.asarray(candidate['rmse'])/control['rmse']-1)).tolist(),
            consumed_development_gate_passed=bool(gate),
            real_optimizer_updates=[r['optimizer_updates'] for r in results],
            scope='One consumed Monza development episode, same two train episodes/data budget, unequal optimizer count inherent in window change. No new test, causal visual improvement or autonomous qualification.',
            next='Whole-episode visual/body information interventions and new conditions before promotion' if gate else
                 'Do not promote; preserve both outcomes and isolate next train-only hypothesis',
            termination_reason='Paired normal-input metrics and audited checkpoint identities compared')
    except Exception as error:
        report.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit comparison failure')
    write_json(run / 'result.json', report)
    print(report)
    if report['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
