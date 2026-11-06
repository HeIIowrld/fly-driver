"""Independent source, split, recorded-gradient Adam and final-head audit."""
import argparse,time
from pathlib import Path
import numpy as np,torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read

def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.fit);m=read(root/'manifest.json')['config'];r=read(root/'result.json');out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256=sha256(root/'result.json'),vehicle_control_sent=False,optimizer_updates=0));assert r['status']=='completed_fixed_schedule_flight_pedal_learning' and m['development_used_for_fit'] is False and m['steps']==r['optimizer_updates']==3000;assert all(sha256(f)==h for f,h in m['source_sha256'].items());assert sha256(m['split'])==m['split_sha256'];assert sha256(m['parent_checkpoint'])==m['parent_checkpoint_sha256']
 for ep,source in zip(read(m['split'])['episodes'],m['receipts']):
  assert ep['role']==source['role'] and sha256(Path(source['data'])/'actor-only.npz')==source['arrays_sha256']==ep['arrays_sha256']
  original=read(Path(source['data'])/'manifest.json')['config'];assert sha256(Path(original['arguments']['recording'])/'teacher/events.jsonl')==source['teacher_events_sha256']
 assert sha256(root/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'];assert sha256(root/'pedals.pt')==r['checkpoint_sha256'];maximum=0.;projection_max=0.
 with np.load(root/'optimizer-trajectory.npz') as z:
  before,gradient,proposal,after=[z[k].copy() for k in ['before','gradient','proposal','after']];assert before.shape==(3000,21);assert np.isfinite(np.concatenate([before,gradient,proposal,after])).all();assert np.max(abs(before[1:]-after[:-1]))==0;assert np.max(abs(gradient[:,:6]))==0 and np.max(abs(gradient[:,18]))==0
  moment=np.zeros(21,dtype=np.float64);variance=moment.copy()
  for i,g in enumerate(gradient.astype(float)):
   moment=.9*moment+.1*g;variance=.999*variance+.001*g*g;expected=before[i].astype(float)-m['learning_rate']*(moment/(1-.9**(i+1)))/(np.sqrt(variance/(1-.999**(i+1)))+1e-8);maximum=max(maximum,float(abs(expected-proposal[i]).max()));projected=proposal[i].clip(-10,10);projected[10:12]=projected[10:12].clip(.0001,10);projected[16:18]=projected[16:18].clip(-10,-.0001);projected[:6]=before[0,:6];projected[18]=before[0,18];projection_max=max(projection_max,float(abs(projected-after[i]).max()))
  cp=torch.load(root/'pedals.pt',map_location='cpu',weights_only=True);final=np.concatenate([cp['pedal_projection']['weight'].numpy().reshape(-1),cp['pedal_projection']['bias'].numpy()]);assert np.max(abs(final-after[-1]))==0
 assert maximum<3e-5 and projection_max==0
 write_json(out/'result.json',dict(status='completed_independent_pedal_optimizer_audit',steps=3000,maximum_adam_proposal_difference=maximum,maximum_projection_difference=projection_max,gradient_steering_zero=True,trajectory_continuous=True,final_checkpoint_exact=True,source_and_split_verified=True,development_used_for_fit=False,optimizer_updates=0,vehicle_control_sent=False))
if __name__=='__main__':main()
