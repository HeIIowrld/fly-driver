"""Generate controlled synthetic appearance interventions through original FlyVis."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch,yaml
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
from flybrain_drive.synthetic_domain_interventions import MODES,intervene
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();planfile=B/'20261101-synthetic-domain-intervention-plan/plan-v2.json';plan=read(planfile);srcplan=read(B/'20261101-joint-vision-yaw-new-plan/plan.json');sources=B/'20261101-joint-vision-yaw-new-sources';model=Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000');cache=Path('C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-data-windows-v1');camera_file=Path('configs/ac-csp-exterior-front-v3-retina.json');camera=read(camera_file)
 if sha256(planfile)!='6d58bc3865542530c411192330a4cbf39ccc5225a79502bbfb3c1d1efe4ce988' or sha256('src/flybrain_drive/synthetic_domain_interventions.py')!=plan['intervention_source_sha256'] or tuple(plan['modes'])!=MODES or sha256('src/flybrain_drive/synthetic_road_visual.py')!=srcplan['road_source_sha256']:raise ValueError('Exact declared plan and interventions required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planfile),checkpoint_sha256=sha256(model/'best_chkpt'),camera_sha256=sha256(camera_file),actor=False,optimizer_updates=0));result={'status':'failed','optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False}
 try:
  os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  torch.set_num_threads(1);torch.manual_seed(863009)
  with torch.device('cuda'):
   net=flyvis.Network(**yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))['config']['network']).to('cuda');net.load_state_dict(torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);net.eval().requires_grad_(False)
   nodes=net.connectome.nodes;receptors=net.stimulus.input_index[0];u=nodes.u[:][receptors];v=nodes.v[:][receptors];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{d}' for k in ('T4','T5') for d in 'abcd']);projector=FlyvisCameraProjector(camera,u,v)
   all_ret=[];all_mot=[];verification=[]
   for seed in plan['source_seeds']:
    entry=next(x for x in srcplan['validation'] if x['seed']==seed);source=sources/f'validation-{seed}';rr=read(source/'result.json');rf=source/'responses.npz'
    if rr['status']!='completed_causal_four_view_synthetic_road' or sha256(rf)!=rr['responses_sha256']:raise ValueError('Exact source response required')
    with np.load(rf,allow_pickle=False) as z:old_ret=z['retina'].copy();old_mot=z['motion'].copy();times=z['times'].copy()
    road=SyntheticRoadVisual(camera,curvature=entry['curvature'],bend_start_m=entry['bend_start_m']);original=[]
    for bend in (0,-1,1):
     original.append([road.render(bend=bend,forward_m=3.*max(0.,i*.04-.4),texture_seed=seed) for i in range(41)])
    seed_ret=[];seed_mot=[]
    for mode in MODES:
     mode_ret=[];mode_mot=[]
     for bend_index in range(3):
      ret=np.stack([np.stack([projector.sample(v) for v in intervene(views,mode)]) for views in original[bend_index]])
      movie=torch.as_tensor(np.transpose(np.repeat(ret[:-1],8,axis=0),(1,0,2))[:,:,None,:],device='cuda')
      with torch.no_grad():
       state=net.steady_state(1.,.005,batch_size=4,value=.5);first=state.nodes.activity[:,indices].cpu().numpy();activity=net.simulate(movie,dt=.005,initial_state=state);motion=np.concatenate([first[None],activity[:,7::8,indices].permute(1,0,2,3).cpu().numpy()],axis=0)
      if ret.shape!=(41,4,721) or motion.shape!=(41,4,8,721) or not np.isfinite(motion).all():raise ValueError('Complete finite four-view history required')
      mode_ret.append(ret);mode_mot.append(motion);del movie,activity
      write_json(run/'progress.json',dict(status='running',seed=seed,mode=mode,bend=[0,-1,1][bend_index]))
     seed_ret.append(np.stack(mode_ret));seed_mot.append(np.stack(mode_mot))
    seed_ret=np.stack(seed_ret);seed_mot=np.stack(seed_mot);er=float(np.max(np.abs(seed_ret[0]-old_ret)));em=float(np.max(np.abs(seed_mot[0]-old_mot)))
    if max(er,em)>2e-5:raise ValueError('Baseline synthetic source did not reproduce')
    verification.append({'seed':seed,'source_sha256':sha256(rf),'baseline_retina_max_abs':er,'baseline_motion_max_abs':em});all_ret.append(seed_ret);all_mot.append(seed_mot)
  out=run/'responses.npz';np.savez_compressed(out,retina=np.stack(all_ret),motion=np.stack(all_mot),times=times,seeds=np.array(plan['source_seeds']),bends=np.array([0,-1,1]));result.update(status='completed_synthetic_domain_interventions',modes=list(MODES),verification=verification,output_sha256=sha256(out),training_admissible=False,physical_or_physiology_verified=False,interpretation='Previously evaluated synthetic roads with deterministic front screen-space perturbations. No 3-D track, weather or biological claim; original held-input FlyVis, no training or actor.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_synthetic_domain_interventions':raise SystemExit(1)
if __name__=='__main__':main()
