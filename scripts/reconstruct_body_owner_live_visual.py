"""Reconstruct selected original live pixels with the exact stateful FlyVis engine."""
import argparse,copy,gc,hashlib,io,json,os,sys,zipfile
from pathlib import Path
import numpy as np,torch
from PIL import Image
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.flyvis_process import FlyvisEngine
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
 src=Path('D:/CodexResearchRuns/flybrain-body-owner-live-20261102-06');live=read(src/'result.json');manifest=read(src/'manifest.json');audit=read(src/'independent-host-audit.json');events=read(src/'events.json')
 if live['status']!='observation_finished' or audit['status']!='completed_stationary_host_freshness_audit' or audit['run_result_sha256']!=sha256(src/'result.json') or sha256(src/'events.json')!=live['events_sha256'] or manifest['config']['visual_worker_sha256']!=sha256('src/flybrain_drive/flyvis_process.py'):raise ValueError('Exact completed source and visual engine required')
 events=[x for x in events if x['status']=='scene_published']
 if len(events)!=201 or len({x['capture_id'] for x in events})!=len(events):raise ValueError('Original selected scene coverage differs')
 archive=src/'vision';bundle=read(archive/'bundle.json');files=read(archive/'archive.json');zp=archive/'camera.zip'
 if sha256(zp)!=bundle['sha256'] or sha256(archive/'archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('Original archive changed')
 expected={x['file']:x for x in files};cfg=read('configs/flight-stream-batched-v1-20261029.json');camera=read(cfg['camera_config']);modeldir=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root'])
 if sha256(cfg['camera_config'])!=cfg['camera_config_sha256'] or sha256(modeldir/'best_chkpt')!=cfg['author_weights_sha256']:raise ValueError('Exact visual checkpoint required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),live_events_sha256=sha256(src/'events.json'),archive_sha256=sha256(zp),visual_worker_sha256=sha256('src/flybrain_drive/flyvis_process.py'),checkpoint_sha256=sha256(modeldir/'best_chkpt'),optimizer_updates=0,actor=False))
 result={'status':'failed','optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False};engine=None
 try:
  torch.set_num_threads(1);torch.manual_seed(863009)
  pool=FlightScenePool(np.random.default_rng(1).random((721,2)),dict(rows=4,columns=4,sigma_image_uv=.2))
  engine=FlyvisEngine(camera,modeldir,cache,pool,device='cuda');state=engine.state;held=None;last=None;retinas=[];motions=[]
  with zipfile.ZipFile(zp) as z,torch.no_grad(),torch.device('cuda'):
   for i,event in enumerate(events):
    images=[]
    for view in ('front','left','right','rear'):
     name=f"{event['capture_id']:04d}-{view}.bmp";raw=z.read(name);entry=expected[name]
     if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Original selected pixel hash differs')
     images.append(np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')).copy())
    now=event['visual_snapshot'];current=torch.as_tensor(np.stack([engine.projector.sample(v) for v in images]),device='cuda')
    if last is not None:
     elapsed=now-last
     if not 0<elapsed<=.15:raise ValueError('Original visual interval invalid')
     for dt,count in held_input_batches(elapsed):
      while count:
       step=min(count,16);state=engine.network.simulate(held[:,None,None,:].expand(-1,step,-1,-1),dt=dt,initial_state=state,as_states=True)[-1];count-=step
    engine._synchronize();retinas.append(current.cpu().numpy());motions.append(state.nodes.activity[:,engine.indices].cpu().numpy());held=current.detach().clone();last=now
    if i%25==0:write_json(run/'progress.json',dict(frame=i,total=len(events)))
  out=run/'visual-responses.npz';np.savez_compressed(out,capture_ids=np.asarray([x['capture_id'] for x in events],dtype=np.int64),retina=np.stack(retinas),motion=np.stack(motions))
  del current,held,state;engine.close();engine=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  result.update(status='completed_original_visual_reconstruction',frames=len(events),responses_sha256=sha256(out),source_run_sha256=sha256(src/'result.json'),device_released=True,interpretation='Original selected pixels and host intervals reconstructed; not yet output parity, physical synchronization or driving')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if engine is not None:
   try:engine.close()
   except Exception as e:result['cleanup_error']=str(e)
  write_json(run/'result.json',result)
 print(json.dumps(result),flush=True)
 if result['status']!='completed_original_visual_reconstruction':raise SystemExit(1)
if __name__=='__main__':
 main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
