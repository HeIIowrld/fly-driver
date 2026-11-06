"""Original TRAIN operating point and full-history projection gradient check; no fit."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.sac_unified_student import read
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.descending_intent_experiment import DescendingExperiment

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(a.plan);e=DescendingExperiment(plan);out=begin_run(a.output,dict(plan=plan,plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),scope=__doc__,vehicle_control_sent=False));w=e.dn.projection.weight;initial=w.detach().clone();rows=[]
 try:
  for i in (0,1):
   features=e.extras[i];raw=e.dn.projection(((features-e.dn.reference)/e.dn.scale).clamp(-5,5)).detach().numpy()[:,0];d=e.items[i][0]
   rows.append(dict(episode=i,events=len(raw),fit=int(d['fit_mask'].sum()),pre=int(e.pre[i].sum()),raw_quantiles=np.quantile(raw,[0,.01,.1,.5,.9,.99,1]).tolist(),contrast_quantiles=np.quantile(8*np.tanh(raw),[0,.1,.5,.9,1]).tolist(),fraction_negative=float(np.mean(raw<0)),fraction_abs_raw_gt2=float(np.mean(abs(raw)>2)),pre_raw_quantiles=np.quantile(raw[e.pre[i]],[0,.1,.5,.9,1]).tolist(),slope_fraction_quantiles=np.quantile(1-np.tanh(raw)**2,[0,.1,.5,.9,1]).tolist()))
  task=dict(source_slot=0,pair=0,temperature=80.,group=1);parts,pred=e.synthetic(task,'episode_full');loss=sum(parts);loss.backward();grad=w.grad.detach().clone();norm=float(grad.norm());direction=grad/grad.norm().clamp_min(1e-12);analytic=float((grad*direction).sum());fd=[]
  for step in (.02,.01):
   values=[]
   for sign in (-1,1):
    with torch.no_grad():w.copy_(initial+sign*step*direction)
    parts,_=e.synthetic(task,'episode_full');values.append(float(sum(parts).detach()))
   estimate=(values[1]-values[0])/(2*step);fd.append(dict(step=step,analytic=analytic,finite_difference=estimate,relative_error=abs(estimate-analytic)/max(abs(analytic),1e-10)))
  with torch.no_grad():w.copy_(initial)
  e.verify_fixed();result=dict(status='completed_descending_intent_train_diagnostic',train_operating_points=rows,synthetic_task=task,gradient_norm=norm,finite_differences=fd,gradient_validated=norm>0 and min(v['relative_error'] for v in fd)<.1,optimizer_updates=0,development_read_for_statistics=False,actor_qualified=False,vehicle_control_sent=False,scope='TRAIN-only statistics; synthetic FD validates this point/direction, not global dynamics or physical control')
  write_json(out/'result.json',result);print(json.dumps(result))
 finally:e.close()
if __name__=='__main__':main()
