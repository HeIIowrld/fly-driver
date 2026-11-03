"""Validate a completed stationary CSP state-clock capture, with no synchronization claim."""
import hashlib,json
from pathlib import Path
from flybrain_drive.diagnostic_cached_body_clock import summarize_cached_body_around_render
ROOT=Path('D:/CodexResearchRuns/flybrain-stateclock-probe-20261102-01')
OUT=Path('D:/CodexResearchRuns/flybrain-stateclock-analysis-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 result=json.loads((ROOT/'result.json').read_text(encoding='utf8'));archive=json.loads((ROOT/'archive.json').read_text(encoding='utf8'))
 if result['status']!='completed' or result['training_admissible'] is not False or result['actor_stream_created'] is not False:raise ValueError('diagnostic result required')
 paths={x['file']:x['sha256'] for x in archive}
 for name,digest in paths.items():
  if sha(ROOT/name)!=digest:raise ValueError('archive changed '+name)
 rows=[]
 for index in range(4):
  name=f'camera/frame-{index:04d}.json'
  if name not in paths:raise ValueError('frame not archived')
  frame=json.loads((ROOT/name).read_text(encoding='utf8'))
  if frame['index']!=index or frame['request_id']!=result['request_id']:raise ValueError('identity mismatch')
  rows.extend(summarize_cached_body_around_render(frame)['views'])
 deploy=Path('D:/CodexResearchRuns/flybrain-stateclock-deploy-20261102-01');native=json.loads((deploy/'native-before.json').read_text(encoding='utf8'))
 import base64
 raw=base64.b64decode(native['prefix_base64'],validate=True)
 if hashlib.sha256(raw).hexdigest()!=native['prefix_sha256']:raise ValueError('native prefix hash mismatch')
 speeds=[x['speed_before_kmh'] for x in rows];OUT.mkdir(parents=True,exist_ok=False)
 report={'status':'completed_stationary_cached_body_diagnostic','frames':4,'view_pairs':len(rows),'speed_min_kmh':min(speeds),'speed_max_kmh':max(speeds),'maximum_abs_within_render_speed_delta_kmh':max(abs(x['speed_difference_kmh']) for x in rows),'native_predeployment_speed_kmh':native['fields']['speedKmh'],'native_predeployment_sample_not_simultaneous':True,'source_result_sha256':sha(ROOT/'result.json'),'archive_sha256':sha(ROOT/'archive.json'),'native_before_sha256':sha(deploy/'native-before.json'),'program_sha256':sha(Path(__file__)),'physical_exposure_sync_verified':False,'native_packet_identity_verified':False,'training_admissible':False,'actor_deployed':False,'vehicle_control_sent':False,'rows':rows}
 (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
