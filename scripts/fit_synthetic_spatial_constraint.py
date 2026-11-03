"""Synthetic constrained spatial projection pilot; no actor deployment."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
planpath=B/'20261101-spatial-constraint-pilot-plan/plan.json';plan=read(planpath);cp=B/plan['source_candidate'];cr=read(cp.parent/'result.json');sp=B/'20261101-adaptive-pattern-fit/adaptive-pattern-contrast.pt';sr=read(sp.parent/'result.json')
if plan['no_actor_deployment'] is not True or plan['validation_seen_during_fit'] is not False or sha256(cp)!=cr['candidate_sha256'] or sha256(sp)!=sr['checkpoint_sha256'] or set(i['seed'] for i in plan['train_sources'])&set(i['seed'] for i in plan['evaluation_sources']):raise ValueError('exact declared lineage and disjoint seed required')
cand=torch.load(cp,map_location='cpu',weights_only=True);orig=torch.load(sp,map_location='cpu',weights_only=True)
if cand['actor_qualified'] is not False or cand['source_spatial_sha256']!=sha256(sp):raise ValueError('diagnostic candidate required')
mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];spec=read(Path(mp['spec']))
with load_array(spec['scene_registration_file'],allow_pickle=False) as z:uv=z['image_uv'].copy()
pool=FlightScenePool(uv,spec['scene_basis']).eval();boundary=RetinalSpatialContrastBoundary(cand['spatial_weight'].shape[0],reference=orig['reference'],max_contrast_rms=orig['max_contrast_rms']).eval()
run=begin_run(B/'20261101-spatial-constraint-pilot-fit-1',dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),source_candidate_sha256=sha256(cp),optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
try:
 ys=[];cs=[];hashes=[]
 for item in plan['train_sources']:
  root=B/item['folder'];rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];file=root/'responses.npz'
  if cfg['texture_seed']!=item['seed'] or rr['status']!='completed_causal_synthetic_road_with_yaw' or sha256(file)!=rr['responses_sha256']:raise ValueError('training source mismatch')
  with np.load(file,allow_pickle=False) as z:
   if not np.array_equal(z['bend'],[0,-1,1]) or not np.allclose(z['yaw_rate'],[0,-.12,.12]):raise ValueError('declared complete yaw-bend grid required')
   retina=z['retina'][:,:,-10:].copy()
  x=np.zeros((90,4,9,721),dtype='float32');x[:,:,0]=retina.reshape(90,4,721)
  with torch.no_grad():
   scene=pool(torch.from_numpy(x));raw=scene.reshape(-1,4,9,16)[:,:,0];center=raw-raw.mean(dim=-1,keepdim=True)-boundary.reference;rms=(center.square().mean(dim=(1,2),keepdim=True)+1e-24).sqrt();gain=(boundary.max_contrast_rms/rms).clamp(max=1.);feature=(center*gain).numpy().reshape(3,3,10,64)
  y=(feature[0,2]-feature[0,1]).mean(axis=0).astype('float64');c=(feature[1,0]-feature[2,0]).mean(axis=0).astype('float64');yn=np.linalg.norm(y);cn=np.linalg.norm(c)
  if min(yn,cn)<1e-10:raise ValueError('nontrivial train contrasts required')
  ys.append(y/yn);cs.append(c/cn);hashes.append(sha256(file))
 Y=np.stack(ys,axis=1);C=np.stack(cs,axis=1);u,sv,_=np.linalg.svd(C,full_matrices=False);rank=int((sv>sv[0]*1e-4).sum());P=np.eye(64)-u[:,:rank]@u[:,:rank].T;yp=P@Y;W=cand['spatial_weight'].numpy().astype('float64');correction=-(W@Y)@np.linalg.pinv(yp,rcond=1e-4);limit=.5*np.linalg.norm(W);raw_norm=np.linalg.norm(correction);factor=min(1.,limit/raw_norm) if raw_norm else 1.;Wnew=W+factor*correction
 if not np.isfinite(Wnew).all():raise ValueError('nonfinite constrained result')
 path=run/'spatial-constraint-candidate.pt';torch.save(dict(schema='flybrain.experimental.spatial_constraint.v1',spatial_weight=torch.tensor(Wnew,dtype=torch.float32),opponent_weight=cand['opponent_weight'].clone(),actor_qualified=False,source_candidate_sha256=sha256(cp),train_source_sha256=hashes),path)
 arr=run/'constraint-matrices.npz';np.savez_compressed(arr,Y=Y,C=C,corner_basis_singular_values=sv,projected_yaw_singular_values=np.linalg.svd(yp,compute_uv=False),correction=correction)
 result.update(status='completed_synthetic_spatial_constraint_fit',candidate_sha256=sha256(path),output_sha256=sha256(arr),training_sources=hashes,validation_used_for_fit=False,corner_rank=rank,cap_scale=factor,weight_norm=float(np.linalg.norm(W)),uncapped_update_norm=float(raw_norm),actual_update_norm=float(np.linalg.norm(factor*correction)),train_yaw_before=float(np.linalg.norm(W@Y)),train_yaw_after=float(np.linalg.norm(Wnew@Y)),train_corner_projection_drift=float(np.linalg.norm((Wnew-W)@C)),optimizer_updates=0,closed_form_solves=1,interpretation='Synthetic mean-feature linear projection; nonlinear temporal dynamics, other stimuli and biology are not implied')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_synthetic_spatial_constraint_fit':raise SystemExit(1)
