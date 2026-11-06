"""TRAIN-only weighted six-muscle head; no new neural or direct image readout."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import MuscleVehicleReadout,read
from scripts.sac_steering_experiment import dataset,onsets,SOURCE_SHA
from scripts.improve_sac_anticipation import command_metrics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
 entries,items,extras=dataset();parent=Path(a.parent);r=read(parent/'result.json');assert r['status']=='completed_fixed_flight_steering_focus'
 if sha256(parent/'steering.pt')!=r['checkpoint_sha256']:raise ValueError('Parent changed')
 cp=torch.load(parent/'steering.pt',map_location='cpu',weights_only=True);head=MuscleVehicleReadout(6);head.load_state_dict(cp['readout'])
 run=begin_run(a.output,dict(arguments=vars(a),seed=780505,program_sha256=sha256(__file__),source_checkpoint_sha256=SOURCE_SHA,parent_checkpoint_sha256=r['checkpoint_sha256'],entries=entries,preweight=16,regularization=.001,scope='Exploratory TRAIN-only pre-transition weighted head, consumed development; parent neural states unchanged',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False))
 with np.load(parent/'predictions.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 x=[];y=[];ws=[]
 def norm(mu):return ((torch.tensor(mu)-head.training_reference)/head.training_scale).numpy().astype(float)
 for i in [0,1]:
  d=items[i][0];pre=np.zeros(len(d['labels']),bool)
  for w in onsets(d):pre|=(d['decision_time']>=w['time']-.8)&(d['decision_time']<=w['time']-.08)&(abs(d['body'][:,4])<.07)
  x.append(norm(arrays[f'train_{i}_muscles'])[d['fit_mask']]);y.append(np.arctanh(d['labels'][d['fit_mask'],0]));ws.append(np.where(pre[d['fit_mask']],16.,1.))
 x=np.concatenate(x);y=np.concatenate(y);w=np.concatenate(ws);mu=(x*w[:,None]).sum(0)/w.sum();ym=(y*w).sum()/w.sum();sd=x.std(0);sd=np.where(sd>1e-8,sd,1.);v=(x-mu)/sd
 beta=np.linalg.solve(v.T@(w[:,None]*v)+.001*w.sum()*np.eye(6),v.T@(w*(y-ym)))/sd;bias=ym-mu@beta
 old={k:v.clone() for k,v in head.state_dict().items()}
 with torch.no_grad():head.projection.weight[0]=torch.tensor(beta,dtype=torch.float32);head.projection.bias[0]=float(bias)
 reports=[]
 for i,key in [(0,'train_0'),(1,'train_1'),(2,'dev')]:
  pr=arrays[key].copy();mu=arrays[f'{key}_muscles'];pr[:,0]=head(torch.tensor(mu)).detach().numpy()[:,0]
  assert np.array_equal(pr[:,1:],arrays[key][:,1:]);arrays[key]=pr;reports.append(command_metrics(pr,items[i][0]['labels'],items[i][0]['fit_mask']))
 for key in ['projection.weight','projection.bias']:assert torch.equal(head.state_dict()[key][1:],old[key][1:])
 cp.update(readout=head.state_dict(),parent_checkpoint_sha256=r['checkpoint_sha256'],head_preweight=16,optimizer_updates=0,actor_qualified=False)
 torch.save(cp,run/'steering.pt');np.savez_compressed(run/'predictions.npz',**arrays)
 result=dict(status='completed_fixed_flight_steering_focus',optimizer_updates=0,closed_form_readout_fits=1,derivation='TRAIN-only preweight16 six-muscle steering ridge; every neural state reused exact from parent',train=reports[:2],dev=reports[2],checkpoint_sha256=sha256(run/'steering.pt'),predictions_sha256=sha256(run/'predictions.npz'),actor_deployed=False,vehicle_control_sent=False,termination_reason='Fixed weighted head and all stored parent muscle predictions evaluated; no new neural training or live control')
 write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
