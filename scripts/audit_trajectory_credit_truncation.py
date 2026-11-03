"""Quantify vehicle-only truncated trajectory gradients on saved fixed histories."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from flybrain_drive.differentiable_planar_vehicle import differentiable_bicycle_step

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=Path('D:/CodexResearchRuns/flybrain-neutral-trajectory-fit-20261102-01');res=json.loads((p/'result.json').read_text());evaluation=p/'evaluation.json';assert res['status']=='completed_fixed_history_neutral_training' and sha(evaluation)==res['evaluation_sha256'];scores=json.loads(evaluation.read_text());source=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-closed-loop-scene-reconstruction-1/trajectory.npz');plan=json.loads(Path('D:/CodexResearchRuns/flybrain-neutral-trajectory-plan-20261102-01/plan.json').read_text());assert sha(source)==plan['trajectory_sha256']
rows=[];arrays={}
with np.load(source,allow_pickle=False) as z:
 for bend in (-1,0,1):
  hist=z[f'gain1_{plan["eval_seed"]}_{bend}'];record=scores[f'baseline_eval_{bend}'];inputs=np.array(record['intent']);assert len(inputs)==len(hist)==81
  grads=[];worlds=[];losses=[]
  for truncate in (False,True):
   control=torch.tensor(inputs,dtype=torch.float64,requires_grad=True);world=torch.tensor([[float(hist[0,1]),float(hist[0,2]),float(hist[0,3]),float(hist[0,4])]],dtype=torch.float64);costs=[];trajectory=[]
   for i,row in enumerate(hist):
    world=differentiable_bicycle_step(world,control[i:i+1]*.35,torch.tensor([float(row[9])],dtype=torch.float64),dt_s=.04);x=world[:,0];offset=(x-12.).clamp_min(0.);center=float(bend)*.018*offset.square();heading=torch.atan(float(bend)*.036*offset);cost=((world[:,1]-center)/.3).square()+plan['heading_weight']*((world[:,2]-heading)/.15).square();costs.append(cost.mean());trajectory.append(world.detach().numpy()[0].copy())
    if truncate and (i+1)%plan['tbptt_steps']==0:world=world.detach()
   loss=torch.stack(costs).mean();loss.backward();grads.append(control.grad.detach().numpy().copy());worlds.append(np.stack(trajectory));losses.append(float(loss.detach()))
  assert np.allclose(worlds[0],worlds[1],atol=1e-12,rtol=0)
  full,truncated=grads;norm=np.linalg.norm(full);tnorm=np.linalg.norm(truncated);cosine=float(np.dot(full,truncated)/(norm*tnorm)) if norm and tnorm else None;gap=float(np.max(np.abs(worlds[0]-np.array(record['world']))));arrays[f'{bend}_full']=full;arrays[f'{bend}_truncated']=truncated
  rows.append({'bend':bend,'loss':losses[0],'full_gradient_norm':float(norm),'truncated_gradient_norm':float(tnorm),'norm_ratio_truncated_to_full':float(tnorm/norm) if norm else None,'gradient_cosine':cosine,'saved_float32_trajectory_max_difference':gap})
out=Path('D:/CodexResearchRuns/flybrain-trajectory-credit-audit-20261102-01');out.mkdir(exist_ok=False);np.savez_compressed(out/'gradients.npz',**arrays);r={'status':'completed_vehicle_only_gradient_audit','rows':rows,'gradient_sha256':sha(out/'gradients.npz'),'evaluation_sha256':sha(evaluation),'trajectory_sha256':sha(source),'program_sha256':sha(__file__),'interpretation':'Posthoc fixed saved steering sequence, differentiable synthetic vehicle only. The full/unrolled versus nine-step truncated gradient does not measure neural network gradients, real driving, or physiology.'};(out/'result.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
