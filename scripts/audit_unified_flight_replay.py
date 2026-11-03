"""Audit saved unified replay clocks and counterfactual arithmetic, without rerunning the model."""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','replay','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.candidate)/'recording';replay=Path(a.replay)
    result, config=read(replay/'result.json'),read(replay/'manifest.json')['config']
    data=replay/'intents.npz'; events=root/'events.jsonl'
    if (result['status'] != 'completed_unified_flight_offline_replay'
            or sha256(data) != result['intents_sha256'] or sha256(events) != config['events_sha256']):
        raise ValueError('Exact original and unified replay required')
    rows=[json.loads(line) for line in events.read_text(encoding='utf8').splitlines()]
    with np.load(data,allow_pickle=False) as f:
        intents,clocks=f['intents'].copy(),f['decision_clock'].copy()
    if (intents.shape != (2,len(rows),2) or not np.isfinite(intents).all()
            or not np.array_equal(clocks,[r['decision_monotonic_seconds'] for r in rows])):
        raise ValueError('Whole original clocks and two finite intent series required')
    delta=np.abs(intents[0]-intents[1])
    prefix = 'preview_motion_mean' if result.get('visual_preview_command_used') else 'visual_zero'
    np.testing.assert_allclose(delta.mean(0),result[prefix+'_mean_abs_intent_delta'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(delta.max(0),result[prefix+'_max_abs_intent_delta'],rtol=0,atol=1e-14)
    speed=sorted(set(round(row['external_task_at_receipt']['desired_speed_m_s']*3.6,3) for row in rows))
    if speed != result['distinct_task_speeds_kmh']:
        raise ValueError('Original external task speeds differ')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(events),
        intents_sha256=sha256(data),result_sha256=sha256(replay/'result.json'),scope=__doc__))
    out=dict(status='completed',decisions=len(rows),original_clocks_and_tasks_match=True,
        intervention_arithmetic_matches=True,independent_model_execution=False,
        fit_allowed=False,physiology_or_closed_loop_driving_verified=False)
    write_json(run/'result.json',out);print(json.dumps(out))


if __name__ == '__main__': main()
