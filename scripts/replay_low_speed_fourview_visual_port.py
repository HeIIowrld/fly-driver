"""Offline replay of audited low-speed camera/body history through fixed FlyVis and geometry port."""
import argparse,base64,hashlib,io,json,os
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_geometry_feedback import CausalGeometryVisualPort
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool
from scripts.replay_ac_camera_motion_probe import diagnostic_steps

p=argparse.ArgumentParser(description=__doc__)
for k in ('probe','host-audit','body-audit','model-dir','cache-root','output'):p.add_argument('--'+k,required=True)
a=p.parse_args();root=Path(a.probe);model=Path(a.model_dir)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
r=read(root/'result.json');ha=Path(a.host_audit);ba=Path(a.body_audit);h=read(ha/'result.json');b=read(ba/'result.json')
ehash=sha256(root/'events.jsonl')
if (r['status']!='completed_low_speed_pulse_diagnostic' or not r['post_stop_verified'] or h['status']!='completed_camera_native_host_audit'
 or b['status']!='completed_low_speed_raw_audit' or read(ha/'manifest.json')['config']['events_sha256']!=ehash
 or read(ba/'manifest.json')['config']['source_events_sha256']!=ehash or not torch.cuda.is_available()):raise ValueError('Exact independently audited input and CUDA required')
events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines() if json.loads(x)['kind']=='native']
if len(events)!=h['native_packets'] or len(events)!=b['samples']:raise ValueError('Complete audited native history required')
os.environ['FLYVIS_ROOT_DIR']=a.cache_root;install_datamate_cache_compat(a.cache_root)
import flyvis
torch.manual_seed(863009);torch.set_num_threads(1);torch.set_default_device('cuda')
cfg=yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'));net=flyvis.Network(**cfg['config']['network']).to('cuda')
weights=torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network'];net.load_state_dict(weights,strict=True);net.eval().requires_grad_(False)
nodes=net.connectome.nodes;idx=net.stimulus.input_index[0];u,v=nodes.u[:][idx],nodes.v[:][idx]
camera=read('configs/ac-csp-yaw-retina.json');projector=FlyvisCameraProjector(camera,u,v);pool=VisualSelfMotionPool(camera,projector.directions_body)
types,all_u,all_v=nodes.type[:].astype(str),nodes.u[:],nodes.v[:];channels=[]
for name in ('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'):
 layer=np.flatnonzero(types==name);mapping={(int(all_u[i]),int(all_v[i])):i for i in layer};channels.append([mapping[(int(q),int(r))] for q,r in zip(u,v,strict=True)])
channels=np.asarray(channels);clocks=np.array([x['native']['received_monotonic_seconds'] for x in events])
if not np.all(np.diff(clocks)>0):raise ValueError('Advancing original perf-counter receipts required')
run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=ehash,host_audit_sha256=sha256(ha/'result.json'),body_audit_sha256=sha256(ba/'result.json'),checkpoint_sha256=sha256(model/'best_chkpt'),camera_config_sha256=sha256('configs/ac-csp-yaw-retina.json'),seed=863009,clock='host performance-counter receipt, physical time unverified',actor=False,training_admissible=False))
cache={};images=[]
with CameraArchive(root/'vision') as archive:
 if archive.bundle_sha256!=r['camera_archive']['bundle_sha256']:raise ValueError('Source archive changed')
 for event in events:
  cam=event.get('camera')
  if cam is None:images.append(None);continue
  capture=cam['capture_id']
  if capture not in cache:
   by_view={v['view_id']:v for v in cam['views']};samples=[]
   if set(by_view)!=set(VIEW_ORDER):raise ValueError('Four views required')
   for name in VIEW_ORDER:
    view=by_view[name];raw=archive.read(Path(view['file']).name)
    if hashlib.sha256(raw).hexdigest()!=view['file_sha256']:raise ValueError('Original image mismatch')
    with Image.open(io.BytesIO(raw)) as image:rgb=np.asarray(image.convert('RGB'))
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=view['rgb_sha256']:raise ValueError('Decoded pixels changed')
    samples.append(projector.sample(rgb))
   cache[capture]=np.stack(samples)
  images.append(cache[capture])
port=CausalGeometryVisualPort(torch.nn.Identity(),pool,2).to('cuda');rows=[];motion=[];features=[];ticks=0;held=None;first=None
with torch.no_grad():
 state=net.steady_state(1.,.005,batch_size=8,value=.5)
 for i,(event,clock) in enumerate(zip(events,clocks,strict=True)):
  if i:
   for dt in diagnostic_steps(float(clock-clocks[i-1])):
    state=net.simulate(held[:,None,None,:],dt=dt,initial_state=state,as_states=True)[-1];ticks+=1
  img=images[i]
  if img is not None:
   if first is None:first=img.copy()
   held=torch.as_tensor(np.concatenate((img,first)),device='cuda')
  elif held is None:held=torch.full((8,721),.5,device='cuda')
  x=state.nodes.activity[:,channels].cpu().numpy();motion.append(x);features.append(np.stack([pool.features(x[:4]),pool.features(x[4:])]))
  native=event['native'];data=base64.b64decode(native['prefix_base64'],validate=True)
  if hashlib.sha256(data).hexdigest()!=native['prefix_sha256']:raise ValueError('Native bytes changed')
  sample=decode_physics_prefix(data,float(native['received_monotonic_seconds']));cam=event.get('camera')
  if cam is None:
   port.current=None
   if port.reference.value is None:port.reference.reset()
   rows.append(dict(index=i,capture_id=None,ready=False,reason='camera_unavailable',reference_ready=port.reference.value is not None,reference_rows=len(port.reference.rows)));continue
  ready=port.observe(x[:4],np.ones(4,dtype=bool),sample,capture_id=int(cam['capture_id']),decision_time=float(clock),camera_receipt_time=float(cam['received']))
  rows.append(dict(index=i,capture_id=cam['capture_id'],ready=ready,reason=port.reason,reference_ready=port.reference.value is not None,reference_rows=len(port.reference.rows),geometry_yaw=float(port.current[0,2]) if ready else None))
np.savez_compressed(run/'responses.npz',clock=clocks,motion=np.asarray(motion),features=np.asarray(features));write_json(run/'port-rows.json',rows)
result=dict(status='completed_audited_low_speed_visual_replay',decisions=len(rows),visual_ticks=ticks,distinct_images=len(cache),
 reference_first_index=next((x['index'] for x in rows if x['reference_ready']),None),first_ready_index=next((x['index'] for x in rows if x['ready']),None),
 ready_count=sum(x['ready'] for x in rows),reasons={k:sum(x['reason']==k for x in rows) for k in sorted(set(x['reason'] for x in rows))},
 port_rows_sha256=sha256(run/'port-rows.json'),responses_sha256=sha256(run/'responses.npz'),
 physical_exposure_or_sensor_sync_verified=False,physiology_verified=False,actor_deployed=False,training_admissible=False,optimizer_updates=0,vehicle_control_sent=False)
write_json(run/'result.json',result);print(json.dumps(result))
