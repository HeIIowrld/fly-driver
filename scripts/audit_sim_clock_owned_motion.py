"""Audit original CSP clock readings and nearby diagnostic AI telemetry; no admission."""
from pathlib import Path
import hashlib,json
import numpy as np
from flybrain_drive.diagnostic_sim_clock_pairs import summarize_clock_pairs

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=Path('D:/CodexResearchRuns/flybrain-simclock-motion-probe-20261102-02');r=json.loads((root/'result.json').read_text(encoding='utf8'));receipt=json.loads((root/'control/api-receipt.json').read_text(encoding='utf8'));rows=json.loads((root/'diagnostic-frames.json').read_text(encoding='utf8'));assert r['status']=='completed' and r['metadata_sha256']==sha(root/'diagnostic-frames.json') and len(rows)==r['frames'] and receipt['status']=='api_returned_true' and receipt['motion_probe_passed'] is True
tele=receipt['telemetry'];times=np.array([x['script_clock'] for x in tele]);speeds=np.array([x['speed_kmh'] for x in tele]);assert np.isfinite(times).all() and np.all(np.diff(times)>=0)
all_pairs=[];moving=[]
for row in rows:
 p=Path(row['source_file']);assert sha(p)==row['sha256'];f=json.loads(p.read_text(encoding='utf8'));d=summarize_clock_pairs(f);pair=d['views'];all_pairs.extend(pair)
 t=f['script_clock'];idx=int(np.argmin(np.abs(times-t)))
 if abs(times[idx]-t)<=.05 and speeds[idx]>1.:moving.extend(pair)
if not moving:raise ValueError('No nearby moving telemetry')
summary=lambda values:{'min':float(np.min(values)),'median':float(np.median(values)),'max':float(np.max(values))}
deltas=lambda z:[v[k] for v in z for k in ('before_car_minus_sim_ms','after_car_minus_sim_ms')]
all_reads=[1000*v[k] for v in all_pairs for k in ('before_read_duration_s','after_read_duration_s')]
out=Path('D:/CodexResearchRuns/flybrain-simclock-motion-analysis-20261102-01');out.mkdir(exist_ok=False)
result={'status':'completed_motion_clock_diagnostic','frames':len(rows),'view_pairs':len(all_pairs),'moving_telemetry_nearby_view_pairs':len(moving),'maximum_speed_kmh':receipt['maximum_speed_kmh'],'all_car_minus_sim_ms':summary(deltas(all_pairs)),'moving_car_minus_sim_ms':summary(deltas(moving)),'read_duration_ms':summary(all_reads),'clock_reversal_count':sum(v['sim_clock_reversed_during_render'] or v['car_clock_reversed_during_render'] for v in all_pairs),'source_metadata_sha256':sha(root/'diagnostic-frames.json'),'control_receipt_sha256':sha(root/'control/api-receipt.json'),'program_sha256':sha(__file__),'telemetry_matching':'nearest CSP script clock within 50 ms, speed > 1 km/h, diagnostic association only','physical_exposure_sync_verified':False,'training_admissible':False,'actor_deployed':False,'interpretation':'Camera frame metadata clock readings with nearby AI diagnostic telemetry; not physical exposure, atomicity, or host-clock conversion.'}
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
