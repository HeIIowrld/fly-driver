"""Source-bound full-history gradient finite differences with consistent autograd mode."""
import argparse,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.functional_priority_experiment import Experiment

def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.diagnostic);r=read(root/'result.json');m=read(root/'manifest.json')['config'];e=Experiment(m['plan']);previous=torch.load(m['plan']['previous_checkpoint'],map_location='cpu',weights_only=True);out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),diagnostic_result_sha256=sha256(root/'result.json'),finite_difference_grad_mode='gain requires_grad=True for both analytic and numeric forward passes',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');task=m['selected_synthetic_tasks'][0];reports=[];start=time.perf_counter()
    try:
        with np.load(root/'gradients.npz') as z:
            for point,value in [('parent',torch.ones_like(e.gain.value)),('previous_gain',previous['temporal_gain']['value'])]:
                with torch.no_grad():e.gain.value.copy_(value)
                losses,_=e.synthetic(task,'episode_full');gradient=torch.autograd.grad(sum(losses),e.gain.value)[0].detach();np.testing.assert_allclose(gradient.numpy(),z[f'{point}_episode_full_synthetic_gradients'][0].sum(0),atol=1e-9,rtol=1e-5);free=(value>.35)&(value<1.65);direction=gradient*free;assert float(direction.norm())>1e-12;direction=direction/direction.norm();analytic=float(gradient@direction);checks=[]
                for epsilon in (.05,.01,.002):
                    values=[]
                    for sign in (1,-1):
                        with torch.no_grad():e.gain.value.copy_(value+sign*epsilon*direction)
                        current,_=e.synthetic(task,'episode_full');values.append(float(sum(current).detach()))
                    numeric=(values[0]-values[1])/(2*epsilon);checks.append(dict(epsilon=epsilon,numerical=numeric,analytic=analytic,absolute_error=abs(numeric-analytic),relative_error=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-12)))
                reports.append(dict(point=point,free_gain_columns=int(free.sum()),checks=checks))
        e.verify_fixed();write_json(out/'result.json',dict(status='completed_full_history_gradient_step_validation',reports=reports,all_largest_step_relative_errors_below_05=all(v['checks'][0]['relative_error']<.05 for v in reports),optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,elapsed_seconds=time.perf_counter()-start,termination_reason='Same full-prefix numerical graph and autograd mode at analytic/numeric evaluation; three finite difference scales on first TRAIN synthetic pair at two fixed gains. F32 small-step cancellation retained; no actual vehicle or global derivative accuracy claim.'))
    finally:e.close()

if __name__=='__main__':main()
