"""Synthetic diagnostic probes of bilateral features and their trained ports."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.bilateral_retinal_contrast_boundary import BilateralRetinalContrastBoundary
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'));pp=B/'20261101-bilateral-information-probe-plan/plan.json';plan=read(pp);cp=B/plan['candidate'];cr=read(cp.parent/'result.json')
if plan['no_real_fit'] is not True or plan['actor_qualified'] is not False or sha256(cp)!=cr['candidate_sha256'] or set(x['seed'] for x in plan['train_sources'])&set(x['seed'] for x in plan['evaluation_sources']):raise ValueError('declared isolation required')
c=torch.load(cp,map_location='cpu',weights_only=True)
if c['actor_qualified'] is not False:raise ValueError('diagnostic candidate required')
model=BilateralRetinalContrastBoundary(c['spatial_model']['projection.weight'].shape[0],normalization_floor=c['normalization_floor']).eval();model.load_state_dict(c['spatial_model'],strict=True)
mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];spec=read(Path(mp['spec']))
with load_array(spec['scene_registration_file'],allow_pickle=False) as z:uv=z['image_uv'].copy()
pool=FlightScenePool(uv,spec['scene_basis']).eval();run=begin_run(B/'20261101-bilateral-information-probe-1',dict(program_sha256=sha256(__file__),plan_sha256=sha256(pp),candidate_sha256=sha256(cp),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
def fetch(item):
 root=B/item['folder'];rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
 if cfg['texture_seed']!=item['seed'] or rr['status']!='completed_causal_synthetic_road_with_yaw' or sha256(f)!=rr['responses_sha256']:raise ValueError('source provenance mismatch')
 with np.load(f,allow_pickle=False) as z:
  ret=z['retina'][:,:,-10:].copy();bend=z['bend'].copy();yaw=z['yaw_rate'].copy()
 if ret.shape!=(3,3,10,4,721) or not np.array_equal(bend,[0,-1,1]) or not np.allclose(yaw,[0,-.12,.12]):raise ValueError('complete declared grid required')
 x=np.zeros((90,4,9,721),dtype='float32');x[:,:,0]=ret.reshape(90,4,721)
 with torch.no_grad():
  sc=pool(torch.from_numpy(x));valid=torch.ones((90,4),dtype=torch.bool);features=model.features(sc,valid).numpy().reshape(3,3,10,64);ports=model(sc,valid).numpy().reshape(3,3,10,-1)
 return {'bilateral_features_64':features.astype('float64'),'trained_spatial_ports_42':ports.astype('float64')},sha256(f)
try:
 training=[fetch(x) for x in plan['train_sources']];evaluation=[(item,*fetch(item)) for item in plan['evaluation_sources']];labels=np.tile(np.repeat([0.,-1.,1.],10),len(training));probes={};reports={};arrays={}
 for name in plan['representations']:
  x=np.concatenate([d[name][:,0].reshape(30,-1) for d,h in training]);mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),1e-6);z=(x-mean)/std;D=np.column_stack((z,np.ones(len(z))));w=np.linalg.solve(D.T@D+np.diag([1.]*(D.shape[1]-1)+[0.]),D.T@labels);probes[name]=(mean,std,w);arrays[name+'_mean']=mean;arrays[name+'_scale']=std;arrays[name+'_weights']=w
  rows=[]
  for item,d,h in evaluation:
   values=d[name];scores=np.column_stack(((values.reshape(90,-1)-mean)/std,np.ones(90)))@w;scores=scores.reshape(3,3,10);corner_feature=values[2,0]-values[1,0];rows.append(dict(seed=item['seed'],source_sha256=h,score_means=scores.mean(axis=2).tolist(),pair_ranking_by_yaw=[float(np.mean(scores[2,j]>scores[1,j])) for j in range(3)],mean_pair_separation_by_yaw=[float(np.mean(scores[2,j]-scores[1,j])) for j in range(3)],opposite_yaw_score_difference_rms=float(np.sqrt(np.mean((scores[:,2]-scores[:,1])**2))),corner_representation_delta_rms=float(np.sqrt(np.mean(corner_feature**2)))));arrays[f"{item['seed']}_{name}_scores"]=scores
  reports[name]=dict(fit_mse=float(np.mean((D@w-labels)**2)),floored_features=int(np.sum(x.std(axis=0)<1e-6)),evaluation=rows)
 out=run/'probes.npz';np.savez_compressed(out,**arrays);result.update(status='completed_synthetic_bilateral_information_probe',representations=reports,training_source_sha256=[h for d,h in training],plan_sha256=sha256(pp),output_sha256=sha256(out),diagnostic_fits=2,optimizer_updates=0,validation_used_for_fit=False,fit_on_real=False,interpretation='Post-hoc synthetic linear decodability diagnostics, repeated correlated frames and common road generator; no actor, measured fly physiology, closed-loop or real-road proof')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_synthetic_bilateral_information_probe':raise SystemExit(1)
