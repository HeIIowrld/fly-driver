import json,hashlib,math
from pathlib import Path
import numpy as np
root=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-frame-clock-ai-probe')
audit=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-frame-clock-ai-audit/result.json')
result=json.loads((root/'result.json').read_text(encoding='utf8')); rows=json.loads((root/'diagnostic-frames.json').read_text(encoding='utf8'));a=json.loads(audit.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert result['status']=='completed' and result['metadata_sha256']==sha(root/'diagnostic-frames.json') and a['frames']==len(rows)==250 and a['exposure_sync_verified'] is False
x=[];y=[];speeds=[]
for row in rows:
 p=Path(row['source_file']);assert sha(p)==row['sha256'];frame=json.loads(p.read_text(encoding='utf8'));d=frame['airflow_diagnostic'];x.append(float(d['finished_source_clock']));y.append(float(d['car_physics_timestamp_ms'])/1000);speeds.append(float(np.linalg.norm(d['vehicle_velocity_candidate_m_s'])))
x=np.asarray(x);y=np.asarray(y);speed=np.asarray(speeds);elapsed=x-x[0];relative=(y-y[0])-elapsed;dx=np.diff(x);dy=np.diff(y)
assert np.all(dx>0) and np.all(dy>0) and np.isfinite(relative).all()
report={'status':'completed_descriptive_candidate_clock_comparison','frames':len(rows),'moving_frames_speed_gt_1':int((speed>1).sum()),'source_camera_diagnostic_metadata_sha256':sha(root/'diagnostic-frames.json'),'prior_audit_sha256':sha(audit),'relative_error_ms_quantiles':{str(q):float(np.quantile(relative,q)*1000) for q in (0,.05,.5,.95,1)},'successive_increment_mismatch_ms_quantiles':{str(q):float(np.quantile(dy-dx,q)*1000) for q in (0,.05,.5,.95,1)},'duration_source_s':float(elapsed[-1]),'duration_car_candidate_s':float(y[-1]-y[0]),'source_clock_origin_unknown':True,'physics_timestamp_semantics_verified':False,'physical_exposure_sync_verified':False,'training_admissible':False,'actor_deployed':False,'optimizer_updates':0,'interpretation':'Elapsed-time comparison anchored at the first frame is descriptive only: candidate car timestamp and source clock have unverified origins and source timestamp semantics, graphics-state timing is not pixel exposure. Cannot establish synchronization or admission.'}
out=Path('D:/CodexResearchRuns/flybrain-candidate-clock-drift-audit-20261102-01');out.mkdir(exist_ok=False);np.savez_compressed(out/'clock-series.npz',source=x,car_candidate=y,relative_seconds=relative,speed=speed);report['arrays_sha256']=sha(out/'clock-series.npz');(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
