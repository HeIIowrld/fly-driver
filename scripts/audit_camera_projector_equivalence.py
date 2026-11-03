"""Compare camera-local retina projector contracts on original archived AC pixels."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
capture=B/'20261101-stationary-native-only-fourview-capture';replay=B/'20261101-stationary-native-flyvis-geometry-replay-v2';rr=read(replay/'result.json');data=replay/'responses.npz';archive=read(capture/'archive.json');r=read(capture/'result.json')
if r['status']!='completed' or r['training_admissible'] is not False or rr['status']!='completed_offline_visual_probe' or sha256(data)!=rr['responses_sha256']:raise ValueError('verified offline source required')
configs=[Path('configs/ac-csp-yaw-retina.json'),Path('configs/ac-csp-exterior-front-v3-retina.json')];cfg=[read(p) for p in configs]
registration=Path('runs/ac-flyvis-continuous-camera-replay-20261029-t1029/retina-registration.npz')
with load_array(registration,allow_pickle=False) as z:u=z['u'].copy();v=z['v'].copy()
projectors=[FlyvisCameraProjector(c,u,v) for c in cfg];events=[json.loads(x) for x in (capture/'client/events.jsonl').read_text(encoding='utf8').splitlines()];arch={x['file']:x['sha256'] for x in archive}
run=begin_run(B/'20261101-camera-projector-equivalence-audit-1',dict(program_sha256=sha256(__file__),actual_responses_sha256=sha256(data),archive_sha256=sha256(capture/'archive.json'),config_sha256=[sha256(p) for p in configs],optimizer_updates=0));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
try:
 with np.load(data,allow_pickle=False) as z:stored=z['retina'].copy()
 if stored.shape!=(24,4,721) or len(events)!=24:raise ValueError('declared complete archive required')
 rows=[];allerrors=[];archive_errors=[]
 for i,e in enumerate(events):
  views={v['view_id']:v for v in e['image_sources']}
  if set(views)!=set(VIEW_ORDER):raise ValueError('four original views required')
  for j,name in enumerate(VIEW_ORDER):
   entry=views[name];p=capture/'camera'/entry['file'];payload=p.read_bytes();h=hashlib.sha256(payload).hexdigest()
   if h!=entry['sha256'] or h!=arch['camera/'+entry['file']]:raise ValueError('image integrity failure')
   with Image.open(p) as img:rgb=np.asarray(img.convert('RGB'))
   x=projectors[0].sample(rgb);y=projectors[1].sample(rgb);diff=float(np.max(np.abs(x-y)));archdiff=float(np.max(np.abs(x-stored[i,j])));allerrors.append(diff);archive_errors.append(archdiff)
   if i in (0,1,23):rows.append(dict(index=i,view=name,projector_difference=diff,stored_difference=archdiff))
 metadata=[]
 for name in VIEW_ORDER:
  a,b=cfg[0]['views'][name],cfg[1]['views'][name];metadata.append(dict(view=name,projection_equal=all(a[k]==b[k] for k in ('projection_x','projection_y','forward_body','right_body','up_body')),yaw_config_origin=a.get('origin_body_m'),exterior_config_origin=b.get('origin_body_m')))
 result.update(status='completed_camera_projector_equivalence_audit',frames=24,view_images=len(allerrors),projector_max_error=max(allerrors),stored_retina_max_error=max(archive_errors),uv_max_error=float(np.max(np.abs(projectors[0].uv-projectors[1].uv))),sampling_points_max_error=float(np.max(np.abs(projectors[0].samples-projectors[1].samples))),body_directions_max_error=float(np.max(np.abs(projectors[0].directions_body-projectors[1].directions_body))),camera_metadata=metadata,sample_rows=rows,physical_camera_origin_equivalence_verified=False,physical_exposure_verified=False,interpretation='Same archived RGB pixels and lattice under two configs; camera-local sampling equivalence does not prove camera origins, rendered world geometry or physical-time alignment')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_camera_projector_equivalence_audit':raise SystemExit(1)
