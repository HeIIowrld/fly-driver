"""Compare archived visual-path interventions with contemporaneous observed body yaw."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from scripts.audit_causal_copy_body_context import first_observed_packets

B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 ab=B/'20261101-unified-moving-teacher-path-ablations';root=B/'20261031-process-body-owned-pause-live';ar=read(ab/'result.json');orig=read(root/'result.json');data=ab/'ablations.npz'
 if ar['status']!='completed_moving_teacher_shadow_path_ablations' or ar['training_admissible'] is not False or orig['training_admissible'] is not False or sha256(data)!=ar['output_sha256'] or sha256(root/'result.json')!=ar['source_original_sha256']:raise ValueError('exact diagnostic-only source required')
 for name,key in [('body-ticker.json','ticker_sha256'),('native-copy-windows.json','native_sha256'),('bridge-events.json','bridge_sha256')]:
  if sha256(root/name)!=orig[key]:raise ValueError('raw evidence mismatch')
 ticks=read(root/'body-ticker.json'); packets=first_observed_packets(read(root/'native-copy-windows.json'));by_packet={x['packet']:x for x in packets}
 with np.load(data,allow_pickle=False) as z:intacts=z['observed_intent'].copy();spatial=z['visual_spatial_intent'].copy();opponent=z['visual_opponent_intent'].copy()
 if len(ticks)!=len(intacts) or not np.isfinite(intacts).all() or not np.isfinite(spatial).all() or not np.isfinite(opponent).all():raise ValueError('complete finite sequences required')
 identity=read('configs/ac-virtual-body-v1.json')['native_identity'];encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);ys=[];vs=[];speeds=[];ages=[];image_ages=[];capture=[];clock=[]
 for row in ticks:
  packet=by_packet[row['packet']]; t=float(row['decision'])
  if packet['first_end']!=row['body_receipt'] or row['visual_published']>t:raise ValueError('original causal receipt mismatch')
  body=encoder.encode(packet['sample'],episode_id=orig['camera_request_id'],decision_monotonic_seconds=t)
  ys.append(float(body.values[4]));vs.append(bool(body.valid[4] and body.valid[16]));speeds.append(float(body.values[16]));ages.append(t-body.original_received_monotonic_seconds);image_ages.append(t-row['visual_published']);capture.append(row['visual_frame_index']);clock.append(t)
 yaw=np.array(ys); valid=np.array(vs);speed=np.array(speeds);age=np.array(ages);image_age=np.array(image_ages);clock=np.array(clock);capture=np.array(capture)
 if not np.all(np.diff(clock)>0) or not np.isfinite(yaw).all() or not np.isfinite(speed).all() or np.min(age)<0 or np.min(image_age)<0:raise ValueError('ordered and finite original receipts required')
 results={}
 for name,threshold in [('positive',.02),('negative',-.02)]:
  mask=valid & (speed>1.) & ((yaw>threshold) if name=='positive' else (yaw<threshold))
  report={'decisions':int(mask.sum()),'unique_visual_frames':len(set(capture[mask].tolist()))}
  for label,other in [('spatial',spatial),('opponent',opponent)]:
   delta=(intacts[:,0]-other[:,0])[mask]; y=yaw[mask];product=delta*y
   report[label]={'mean_signed_full_minus_lesion':float(delta.mean()) if len(delta) else None,'mean_abs_effect':float(np.abs(delta).mean()) if len(delta) else None,'fraction_opposes_body_yaw':float(np.mean(product<0)) if len(delta) else None}
  results[name]=report
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),raw_source_sha256=sha256(root/'result.json'),ablation_sha256=ar['output_sha256'],yaw_threshold_rad_s=.02,speed_threshold_m_s=1.,training_admissible=False,optimizer_updates=0))
 np.savez_compressed(run/'aligned.npz',yaw=yaw,valid=valid,speed=speed,body_age=age,visual_publication_age=image_age,decision=clock,capture_id=capture,spatial_effect=intacts[:,0]-spatial[:,0],opponent_effect=intacts[:,0]-opponent[:,0])
 out={'status':'completed_archived_visual_body_path_alignment','reports':results,'decisions':len(ticks),'unique_frames':len(set(capture.tolist())),'body_age_max_seconds':float(age.max()),'publication_age_max_seconds':float(image_age.max()),'arrays_sha256':sha256(run/'aligned.npz'),'optimizer_updates':0,'vehicle_control_sent':False,'training_admissible':False,'limitations':'Same-time native receipt and published-frame host timing only; physical exposure alignment unverified; consecutive samples and frames correlated, not independent, not local derivative or correct driving.'}
 write_json(run/'result.json',out);print(json.dumps(out))
if __name__=='__main__':main()
