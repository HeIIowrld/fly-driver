"""Replay actual diagnostic native packets through existing body encoder, no policy."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix
OUT=Path('D:/CodexResearchRuns/flybrain-real-acceleration-mask-audit-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 outputs={}
 for suffix in ('01','02'):
  root=Path(f'D:/CodexResearchRuns/flybrain-marker-motion-20261102-{suffix}');result=json.loads((root/'result.json').read_text(encoding='utf8'));manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'));file=root/'native-copy-windows.json'
  if result['status']!='completed' or sha(file)!=result['native_copy_sha256'] or manifest['config']['before']['car_id']!='bmw_z4_gt3':raise ValueError('verified diagnostic source and car required')
  identity={'car_model':'bmw_z4_gt3','shared_memory_version':'1.7','ac_version':'1.16.4'}
  encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);seen=set();count=nonzero=0;mask_violations=[];speed_valid=0
  for row in json.loads(file.read_text(encoding='utf8')):
   auth=row['authority'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
   if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('native prefix hash mismatch')
   pid=auth['packet_id']
   if pid in seen:continue
   seen.add(pid);t=float(row['copy']['second_copy_end']);sample=decode_navigation_body_prefix(raw[:PHYSICS_PREFIX.size],t)
   if sample.legacy.packet_id!=pid:raise ValueError('packet identity mismatch')
   out=encoder.encode(sample,episode_id=suffix,decision_monotonic_seconds=t)
   count+=1;nonzero+=int(any(abs(x)>1e-8 for x in sample.legacy.acceleration_native_g));speed_valid+=int(out.valid[16])
   if out.valid[5:8].any() or np.any(out.values[5:8]!=0):mask_violations.append(pid)
  outputs[suffix]={'unique_packets':count,'packets_with_nonzero_native_acceleration':nonzero,'speed_valid':speed_valid,'unregistered_acceleration_leaks':mask_violations,'source_result_sha256':sha(root/'result.json'),'native_source_sha256':sha(file)}
 report={'status':'completed_actual_native_acceleration_mask_audit','runs':outputs,'program_sha256':sha(Path(__file__)),'actor_qualified':False,'physical_time_verified':False,'training_admissible':False,'limits':'Recorded diagnostics, encoder replay only; acceleration axis, gravity and exposure sync remain unverified. No change to biological circuit or real driving.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
 if any(x['unregistered_acceleration_leaks'] for x in outputs.values()):raise SystemExit(1)
if __name__=='__main__':main()
