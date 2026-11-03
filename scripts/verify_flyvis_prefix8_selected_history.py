"""Replay selected original camera history through reference and speculative FlyVis."""
import argparse,hashlib,io,json,os,zipfile
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.graph import sha256
from flybrain_drive.held_visual_prefix import HeldVisualPrefix
from flybrain_drive.retinotopic_readout import column_indices


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 src=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-held-prefix');live=read(src/'result.json');events=read(src/'events.json')
 if live['status']!='observation_finished' or not live['held_prefix'] or sha256(src/'events.json')!=live['events_sha256'] or live['held_prefix_source_sha256']!=sha256('src/flybrain_drive/held_visual_prefix.py'):raise ValueError('Exact live selected events and prefix required')
 archive=src/'vision';bundle=read(archive/'bundle.json');manifest=read(archive/'archive.json');zp=archive/'camera.zip'
 if sha256(zp)!=bundle['sha256'] or sha256(archive/'archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('Original archive changed')
 expected={x['file']:x for x in manifest};groups=[]
 with zipfile.ZipFile(zp) as z:
  for event in events:
   group=[]
   for view in ('front','left','right','rear'):
    name=f"{event['capture_id']:04d}-{view}.png";raw=z.read(name)
    if len(raw)!=expected[name]['bytes'] or hashlib.sha256(raw).hexdigest()!=expected[name]['sha256']:raise ValueError('Original selected pixel changed')
    group.append(np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')).copy())
   groups.append(group)
 cfg=read('configs/flight-stream-batched-v1-20261029.json');camera=read(cfg['camera_config']);modeldir=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root'])
 if sha256(cfg['camera_config'])!=cfg['camera_config_sha256'] or sha256(modeldir/'best_chkpt')!=cfg['author_weights_sha256']:raise ValueError('Exact visual sources required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),prefix_sha256=sha256('src/flybrain_drive/held_visual_prefix.py'),live_events_sha256=sha256(src/'events.json'),archive_sha256=sha256(zp),checkpoint_sha256=sha256(modeldir/'best_chkpt'),actor=False,optimizer_updates=0))
 result={'status':'failed','vehicle_control_sent':False,'actor_deployed':False}
 try:
  os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  torch.set_num_threads(1);torch.manual_seed(863009)
  with torch.no_grad(),torch.device('cuda'):
   net=flyvis.Network(**yaml.safe_load((modeldir/'_meta.yaml').read_text())['config']['network']).to('cuda');net.load_state_dict(torch.load(modeldir/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);net.eval().requires_grad_(False)
   nodes=net.connectome.nodes;recep=net.stimulus.input_index[0];u=nodes.u[:][recep];v=nodes.v[:][recep];projector=FlyvisCameraProjector(camera,u,v);indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{kind}{d}' for kind in ('T4','T5') for d in 'abcd'])
   base=net.steady_state(1.,.005,batch_size=4,value=.5);candidate=base;held=None;last=None;rows=[]
   for index,(event,group) in enumerate(zip(events,groups,strict=True)):
    when=event['visual_snapshot'];current=torch.as_tensor(np.stack([projector.sample(x) for x in group]),device='cuda')
    if held is not None:
     elapsed=when-last
     if not 0<elapsed<=.15:raise ValueError('Original visual snapshot interval invalid')
     prefix=HeldVisualPrefix(net,candidate,held,steps=8);candidate=prefix.commit(elapsed)
     for dt,n in held_input_batches(elapsed):base=net.simulate(held[:,None,None,:].expand(-1,n,-1,-1),dt=dt,initial_state=base,as_states=True)[-1]
    else:elapsed=None
    torch.cuda.synchronize();state_diff=float((candidate.nodes.activity-base.nodes.activity).abs().max().item());motion_diff=float((candidate.nodes.activity[:,indices]-base.nodes.activity[:,indices]).abs().max().item())
    rows.append({'index':index,'capture_id':event['capture_id'],'elapsed':elapsed,'state_max_abs':state_diff,'motion_max_abs':motion_diff})
    held=current;last=when
   if max(x['state_max_abs'] for x in rows)>2e-5:raise ValueError('Accumulated state parity exceeded 2e-5')
  result.update(status='completed_selected_live_history_prefix_parity',frames=len(rows),intervals=len(rows)-1,state_max_abs=max(x['state_max_abs'] for x in rows),motion_max_abs=max(x['motion_max_abs'] for x in rows),comparisons=rows,original_skipped_frames=live['camera_close']['frames']-len(rows),interpretation='Two offline replays of actual selected images and visual snapshot intervals; neither reproduces live outputs, physical exposure timing, physiology or driving')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},allow_nan=False))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
