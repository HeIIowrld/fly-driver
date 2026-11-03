"""Compare original and registered readout neutral on saved intent, offline only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_pedal_learning_v2 import bind_registered_reference
from flybrain_drive.graph import sha256


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','teacher-eval','graph-replay','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    a=p.parse_args();candidate,teacher,replay=map(Path,(a.candidate,a.teacher_eval,a.graph_replay))
    c,t,r=[read(x/'result.json') for x in (candidate,teacher,replay)]
    event_hash=sha256(candidate/'recording/events.jsonl')
    if (c['role'] != a.role or c['fit_allowed'] is not False or t['role'] != a.role or r['role'] != a.role
            or read(teacher/'manifest.json')['config']['events_sha256'] != event_hash
            or read(replay/'manifest.json')['config']['events_sha256'] != event_hash
            or sha256(teacher/'evaluation.npz') != t['evaluation_sha256']
            or sha256(replay/'intents.npz') != r['intents_sha256']
            or sha256(replay/'geometry-readiness.npz') != r['geometry_readiness_sha256']):
        raise ValueError('Original fixed role/clock/evidence required')
    base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    plan=read(base/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model']
    if sha256(plan['checkpoint']) != read(replay/'manifest.json')['config']['source_motor_sha256']:
        raise ValueError('Original physical checkpoint differs')
    reference_path,readout_path=Path(plan['cpu_reference']),Path(plan['cpu_readout'])
    binding=bind_registered_reference(reference_path.read_bytes(),readout_path.read_bytes())
    original=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)['model']
    registered=binding.readout_state
    for name in ('turn_log_gain','steering_mix'):
        if not torch.equal(original['readout.'+name],torch.as_tensor(registered[name])):
            raise ValueError('Original turn gain/mix differs from registered private readout')
    old=original['readout.lateral_reference'];new=torch.tensor(registered['lateral_reference'])
    gain=original['readout.turn_log_gain'].exp();mix=original['readout.steering_mix'].softmax(0)
    shift=(gain*((old-new)@mix)).item()
    # Check algebra independently on representative lateral muscle differences.
    lateral=torch.tensor([[0.,0.],[.001,-.002],[-.001,.002]],dtype=torch.float32)
    direct=(gain*((lateral-new)@mix)).tanh()
    original_value=(gain*((lateral-old)@mix)).tanh()
    reconstructed=(torch.atanh(original_value)+shift).tanh()
    error=float((direct-reconstructed).abs().max())
    if error>2e-6:raise ValueError('Readout-only counterfactual algebra mismatch')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,event_hash=event_hash,
        source_motor_sha256=sha256(plan['checkpoint']),cpu_reference_sha256=sha256(reference_path),
        cpu_readout_sha256=sha256(readout_path),teacher_sha256=t['evaluation_sha256'],
        replay_sha256=r['intents_sha256'],old_lateral=old.tolist(),registered_lateral=new.tolist(),
        shift_preactivation=shift,scope='output-only static counterfactual, no model mutation or calibration'))
    with np.load(teacher/'evaluation.npz',allow_pickle=False) as f:
        target=f['target'][:,2,1].copy();eligible=f['informative'].copy();clock=f['decision_clock'].copy()
    with np.load(replay/'intents.npz',allow_pickle=False) as f:
        intents=f['intents'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    with np.load(replay/'geometry-readiness.npz',allow_pickle=False) as f:
        ready=f['ready'].copy();np.testing.assert_array_equal(clock,f['decision_clock'])
    values=intents[0,:,0]
    if np.any(np.abs(values)>=1):raise ValueError('Finite inverse tanh reconstruction required')
    alternative=np.tanh(np.arctanh(values)+shift)
    keep=eligible & ready
    expected=-np.sign(target[keep])
    result=dict(status='completed_neutral_reference_counterfactual',role=a.role,
        observations=int(keep.sum()),original_match=int((np.sign(values[keep])==expected).sum()),
        alternative_match=int((np.sign(alternative[keep])==expected).sum()),
        original_mean=float(values[keep].mean()) if keep.any() else None,
        alternative_mean=float(alternative[keep].mean()) if keep.any() else None,
        preactivation_shift=shift,algebra_max_error=error,
        full_neutral_recalibration_or_actual_actor_verified=False,
        warning='Registered quiet reference was qualified under a different no-vision physical setup; '
                'this output-only counterfactual does not establish a valid unified-model neutral.',
        fit_allowed=False,optimizer_updates=0,vehicle_control_sent=False,actor_deployed=False)
    write_json(run/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
