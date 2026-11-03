"""Replay verified actual stationary four-camera pixels through FlyVis, without control."""
import argparse,hashlib,io,json,os,time
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.frame_persistence_profile_v3 import verify_buffered_persistence_archive
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER,integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();src=Path(a.source);model=Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000');camera=Path('configs/ac-csp-retina.json');data=json.loads((src/'result.json').read_text(encoding='utf8'));events=json.loads((src/'diagnostic-frames.json').read_text(encoding='utf8'));host=[json.loads(line) for line in (src/'live-frame-hashes.jsonl').read_text(encoding='utf8').splitlines()]
 if data['status']!='completed' or sha256(src/'diagnostic-frames.json')!=data['metadata_sha256'] or sha256(src/'live-frame-hashes.jsonl')!=data['host_observed_hashes_sha256'] or len(events)!=len(host):raise ValueError('verified moving diagnostic required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_result_sha256=sha256(src/'result.json'),source_events_sha256=sha256(src/'diagnostic-frames.json'),checkpoint_sha256=sha256(model/'best_chkpt'),model_config_sha256=sha256(model/'_meta.yaml'),camera_config_sha256=sha256(camera),seed=863009,actor=False,training=False,real_time_claim=False,clock='original CSP script frame times; physical capture unverified',counterfactual='first frame held with identical initial state and timing'))
 result=dict(status='failed',vehicle_control_sent=False,actor_qualified=False,physical_exposure_verified=False,training=False)
 try:
  if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
  torch.set_num_threads(1);torch.manual_seed(863009);cache=str(Path(os.environ['LOCALAPPDATA'])/'flybrain-bio-20261029/flyvis-data-windows-v1');os.environ['FLYVIS_ROOT_DIR']=cache;install_datamate_cache_compat(cache)
  import flyvis
  torch.set_default_device('cuda');config=yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'));network=flyvis.Network(**config['config']['network']).to('cuda');network.load_state_dict(torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
  nodes=network.connectome.nodes;ids=network.stimulus.input_index[0];u,v=nodes.u[:][ids],nodes.v[:][ids];projector=FlyvisCameraProjector(json.loads(camera.read_text(encoding='utf8')),u,v);types,all_u,all_v=nodes.type[:].astype(str),nodes.u[:],nodes.v[:];channels=[]
  for name in ('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'):
   layer=np.flatnonzero(types==name);mapping={(int(all_u[i]),int(all_v[i])):i for i in layer};channels.append([mapping[(int(q),int(r))] for q,r in zip(u,v,strict=True)])
  images=[];clocks=[]
  for event,record in zip(events,host,strict=True):
   path=Path(event['source_file']);raw=path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=event['sha256'] or event['sha256']!=record['metadata_sha256']:raise ValueError('frame metadata hash mismatch')
   frame=json.loads(raw);views={v['view_id']:v for v in frame['views']};sampled=[]
   if frame['index']!=event['index'] or frame['training_admissible'] is not False or frame['actor_stream_created'] is not False:raise ValueError('diagnostic frame only')
   for name in VIEW_ORDER:
    file=path.parent/views[name]['file'];raw=file.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['images'][name]['sha256']:raise ValueError('host observed PNG hash mismatch')
    with Image.open(io.BytesIO(raw)) as im:rgb=np.asarray(im.convert('RGB'))
    sampled.append(projector.sample(rgb))
   images.append(np.stack(sampled));clocks.append(float(frame['script_clock']))
  images=np.stack(images);clocks=np.array(clocks);channels=np.asarray(channels)
  if len(images)<2 or not np.all(np.diff(clocks)>0):raise ValueError('Ordered original times required')
  started=time.perf_counter();activities=[];ticks=0
  with torch.no_grad():
   state=network.steady_state(1.,.005,batch_size=8,value=.5);held=None
   for i in range(len(images)):
    if i:
     for dt in integration_intervals(float(clocks[i]-clocks[i-1])):
      state=network.simulate(held[:,None,None,:],dt=dt,initial_state=state,as_states=True)[-1];ticks+=1
    held=torch.as_tensor(np.concatenate([images[i],images[0]],axis=0),device='cuda');activities.append(state.nodes.activity[:,channels].cpu().numpy())
  activity=np.stack(activities);out=run/'responses.npz';np.savez_compressed(out,clock=clocks,retina=images,motion=activity);difference=activity[:,:4]-activity[:,4:]
  result.update(status='completed_offline_flyvis_motion_diagnostic_replay',groups=len(images),ticks=ticks,compute_seconds=time.perf_counter()-started,response_sha256=sha256(out),retina_abs_delta_max=float(np.max(np.abs(images-images[0]))),motion_counterfactual_abs_max=float(np.max(np.abs(difference))),physical_or_physiological_validation=False,interpretation='Actual AI diagnostic camera frames, CSP frame-start timing replay and held-first counterfactual; not synchronized physical exposure, physiology, training, or driving proof')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result));
 if result['status']!='completed_offline_flyvis_motion_diagnostic_replay':raise SystemExit(1)
if __name__=='__main__':main()
