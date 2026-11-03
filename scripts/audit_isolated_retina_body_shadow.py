"""Replay original archived pixels and native prefixes through the recorded retina."""
import argparse,base64,hashlib,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.frame_persistence_profile_v3 import verify_buffered_persistence_archive
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.source)
 result=json.loads((src/'result.json').read_text(encoding='utf8'));events=json.loads((src/'events.json').read_text(encoding='utf8'))
 if result['status']!='completed_stationary_owned_mmf_resume_diagnostic' or sha256(src/'events.json')!=result['events_sha256'] or sha256(src/'sensory.npz')!=result['sensory_sha256']:raise ValueError('original successful diagnostic and hashes required')
 manifest=json.loads((src/'manifest.json').read_text(encoding='utf8'));config='configs/ac-csp-exterior-front-v3-retina.json'
 if sha256(config)!=manifest['config']['sensory_config_sha256'] or sha256('src/flybrain_drive/ac_csp_retina.py')!=manifest['config']['retina_sha256']:raise ValueError('same original transducer and configuration required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_result_sha256=sha256(src/'result.json'),source_events_sha256=sha256(src/'events.json'),source_sensory_sha256=sha256(src/'sensory.npz'),actor=False,training=False));retina=CSPCameraRetina(json.loads(Path(config).read_text(encoding='utf8')));computed=[]
 with CameraArchive(src/'vision') as archive:
  rows=verify_buffered_persistence_archive(archive,expected_mmf=True)
  for event in events:
   row=rows[event['capture_id']];samples={}
   for view in row['views']:
    name=view['view_id'];raw=archive.read(view['file'].removeprefix('vision/'))
    if hashlib.sha256(raw).hexdigest()!=view['file_sha256']:raise ValueError('archive image file mismatch')
    with Image.open(io.BytesIO(raw)) as im:rgb=np.asarray(im.convert('RGB'))
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=view['rgb_sha256']:raise ValueError('original RGB mismatch')
    samples[name]=dict(rgb=rgb,capture_id=event['capture_id'],generation=event['generation'],received=event['camera_received'],render_started_lower=view['render_started_lower'])
   raw=base64.b64decode(event['native_prefix_base64'],validate=True)
   if hashlib.sha256(raw).hexdigest()!=event['native_prefix_sha256']:raise ValueError('original body prefix hash mismatch')
   body=decode_physics_prefix(raw,event['body_received'])
   if body.packet_id!=event['body_packet']:raise ValueError('body packet mismatch')
   computed.append(retina.pack(samples,decision_time=event['decision'],native_scalars={k:body.fields[k] for k in retina.required_native_fields},native_receipt=body.received_monotonic_seconds))
 with np.load(src/'sensory.npz',allow_pickle=False) as recorded:
  differences={k:float(np.max(np.abs(np.stack([item[k] for item in computed])-recorded[k]))) for k in recorded.files}
 if any(v!=0. for v in differences.values()):raise ValueError('retina replay differs: '+str(differences))
 write_json(run/'result.json',dict(status='completed_exact_stationary_retina_replay',groups=len(events),archive_groups=len(rows),max_abs_differences=differences,physical_exposure_verified=False,flyvis_motion_verified=False,actor_qualified=False,vehicle_control_sent=False))
 print(json.dumps(differences))
if __name__=='__main__':main()
