"""Owned GPU FlyVis load on synthetic saved retina; no actor or simulator control."""
import argparse,json,os,time
from pathlib import Path
import numpy as np,torch,yaml
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stop',required=True);a=p.parse_args();r=B/'20261101-flyvis-road-varied-879715';rr=read(r/'result.json');f=r/'responses.npz';model=Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000');cache=Path('C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-data-windows-v1')
 if rr['status']!='completed_causal_four_view_synthetic_road' or sha256(f)!=rr['responses_sha256']:raise ValueError('original synthetic retinal source required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),retina_sha256=sha256(f),flyvis_checkpoint_sha256=sha256(model/'best_chkpt'),scope='repeated synthetic four-view retina; no camera or actor'))
 rows=[];result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False)
 try:
  torch.set_num_threads(1);os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  torch.set_default_device('cuda');config=yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**config).to('cuda');network.load_state_dict(torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
  with np.load(f,allow_pickle=False) as z:ret=z['retina'][0].copy()
  retinal=torch.as_tensor(ret,device='cuda');state=network.steady_state(1.,.005,batch_size=4,value=.5);torch.cuda.synchronize()
  while not Path(a.stop).exists():
   start=time.perf_counter();held=retinal[len(rows)%len(retinal)];movie=held[:,None,None,:].expand(-1,8,-1,-1)
   with torch.no_grad():state=network.simulate(movie,dt=.005,initial_state=state,as_states=True)[-1]
   torch.cuda.synchronize();end=time.perf_counter();rows.append((start,end));
   if len(rows)==1:write_json(run/'ready.json',dict(ready=True,started=start,finished=end))
   delay=.04-(end-start)
   if delay>0:time.sleep(delay)
  if not rows:raise ValueError('no visual work completed')
  out=run/'visual-timing.npz';np.savez_compressed(out,windows=np.asarray(rows));result.update(status='completed_owned_flyvis_load',chunks=len(rows),timing_sha256=sha256(out),scope='repeated synthetic retained retina, 8x5ms ticks, independent process, not a real sensor stream')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
