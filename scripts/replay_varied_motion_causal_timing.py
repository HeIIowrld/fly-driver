"""Replay saved synthetic receptors under the historical causal timing contract."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch,yaml
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 root=B/'20261101-flyvis-varied-motion-879501';res=read(root/'result.json');f=root/'responses.npz'
 if res['status']!='completed_four_view_varied_synthetic_motion' or sha256(f)!=res['responses_sha256']:raise ValueError('source')
 with np.load(f,allow_pickle=False) as z: retina=z['retina'].copy();original=z['motion'].copy();times=z['times'].copy()
 if retina.shape!=(3,41,4,721) or original.shape!=(3,41,4,8,721) or not np.isfinite(retina).all() or not np.allclose(times,np.arange(41)*.04):raise ValueError('shape')
 model=Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000'); cache=Path('C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-data-windows-v1')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_sha256=sha256(f),model_sha256=sha256(model/'best_chkpt'),policy='initial fixed grey steady state; then hold preceding known receptor for elapsed interval',dt=.005,optimizer_updates=0))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  torch.set_num_threads(1);torch.manual_seed(863009)
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  torch.set_default_device('cuda');cfg=yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**cfg).to('cuda');network.load_state_dict(torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
  nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors]
  indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{d}' for k in ('T4','T5') for d in 'abcd'])
  cases=[]
  with torch.no_grad():
   for i in range(3):
    state=network.steady_state(1.,.005,batch_size=4,value=.5)
    first=state.nodes.activity[:,indices].cpu().numpy()
    movie=torch.tensor(np.transpose(np.repeat(retina[i,:-1],8,axis=0),(1,0,2))[:,:,None,:],device='cuda')
    activity=network.simulate(movie,dt=.005,initial_state=state)
    held=activity[:,7::8,indices].permute(1,0,2,3).cpu().numpy()
    cases.append(np.concatenate([first[None],held],axis=0));del activity,movie,state
    write_json(run/'progress.json',dict(completed=i+1,total=3))
  motion=np.stack(cases)
  if motion.shape!=original.shape or not np.isfinite(motion).all():raise ValueError('bad response')
  output=run/'responses.npz';np.savez_compressed(output,retina=retina,motion=motion,times=times)
  result.update(status='completed_causal_prior_frame_replay',responses_sha256=sha256(output),motion_same_index_mae=float(np.abs(motion-original).mean()),motion_same_index_max=float(np.abs(motion-original).max()),source_motion_one_frame_shift_mae=float(np.abs(motion[:,1:]-original[:,:-1]).mean()),source_motion_one_frame_shift_max=float(np.abs(motion[:,1:]-original[:,:-1]).max()),timing_policy='historical prior-frame hold; grey steady-state initialization',actor_qualified=False)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
