"""Hash-bound original full flight trajectories; decoder inputs are muscle6 only."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.paired_flight_vehicle_readout import PairedFlightVehicleReadout
from scripts.train_qualified_sac_unified import load_episode
from scripts.fit_bounded_sac_steering import pre_mask

def load_data(plan):
 fit=Path(plan['parent_fine_fit']);r=read(fit/'result.json');m=read(fit/'manifest.json')['config'];assert sha256(fit/'predictions.npz')==r['predictions_sha256']==plan['muscle_trajectories_sha256'];entries=m['entries'];items=[load_episode(e['data']) for e in entries]
 with np.load(fit/'predictions.npz') as z:muscles=[z[f'muscles_{e}'].copy() for e in range(3)];baseline=[z[f'prediction_{e}'].copy() for e in range(3)]
 for e,(d,c,provenance) in enumerate(items):assert provenance['arrays_sha256']==entries[e]['arrays_sha256'] and muscles[e].shape==(len(d['scene']),6) and np.isfinite(muscles[e]).all() and (muscles[e]>=0).all() and (muscles[e]<=1).all()
 return entries,items,muscles,baseline,[pre_mask(d,-1) for d,_,_ in items]

def normalization(muscles,items,floor):
 x=np.concatenate([muscles[e][items[e][0]['fit_mask']] for e in (0,1)]).astype(float);difference=x[:,1::2]-x[:,::2];common=.5*(x[:,1::2]+x[:,::2]);context=np.concatenate([abs(difference[:,:2]),common[:,:2]],1)
 return dict(difference_scale=torch.tensor(np.maximum(difference.std(0),floor),dtype=torch.float32),context_reference=torch.tensor(context.mean(0),dtype=torch.float32),context_scale=torch.tensor(np.maximum(context.std(0),floor),dtype=torch.float32),power_reference=torch.tensor(common[:,2].mean(),dtype=torch.float32),power_scale=torch.tensor(max(common[:,2].std(),floor),dtype=torch.float32))

def initial_head(stats,items,muscles,plan,monotone):
 head=PairedFlightVehicleReadout(**stats,monotone=monotone,minimum_power_gain=plan['minimum_power_gain'],maximum_gain=plan['maximum_gain']);x=torch.tensor(np.concatenate([muscles[e][items[e][0]['fit_mask']] for e in (0,1)]));target=np.concatenate([items[e][0]['labels'][items[e][0]['fit_mask']] for e in (0,1)]).astype(float)
 with torch.no_grad():
  differential,context,power=[v.numpy().astype(float) for v in head.features(x)];ridge=plan['initial_ridge'];steer=np.arctanh(np.clip(target[:,0],-.999,.999));turn=np.linalg.solve(differential.T@differential+ridge*np.eye(3),differential.T@steer);head.steering_weight.copy_(torch.tensor(turn,dtype=torch.float32));design=np.column_stack([context,np.ones(len(context))]);logits=np.log(np.clip(target[:,1:],.001,.999)/(1-np.clip(target[:,1:],.001,.999)));residual=logits-power*np.array([.1,-.1]);coef=np.linalg.solve(design.T@design+ridge*np.eye(5),design.T@residual);head.context_weight.copy_(torch.tensor(coef[:4].T,dtype=torch.float32));head.long_bias.copy_(torch.tensor(coef[4],dtype=torch.float32));head.project()
 return head
