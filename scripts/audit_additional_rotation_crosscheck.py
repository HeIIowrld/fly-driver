"""Cross-check three saved low-speed integrated rotation audits and source identities."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();sources=['20261101-fast-body-fourview-probe','20261101-long-reference-new-history-probe','20261101-low-speed-fourview-prefix-reverse-probe'];reports=[];bundles=[]
 for i,name in enumerate(sources,1):
  source=B/name;d=B/f'20261101-additional-rotation-audit-{i}';r=read(d/'result.json');manifest=read(d/'manifest.json')['config'];orig=read(source/'result.json');bundle=read(source/'vision/bundle.json');arr=d/'angular-comparison.npz'
  if r['status']!='completed_additional_low_speed_integrated_rotation' or r['training_admissible'] is not False or orig['training_admissible'] is not False or sha256(arr)!=r['comparison_sha256'] or sha256(source/'events.jsonl')!=manifest['events_sha256'] or sha256(source/'vision/camera.zip')!=bundle['sha256'] or bundle['sha256']!=manifest['camera_bundle_sha256']:raise ValueError('original matched source required')
  with np.load(arr,allow_pickle=False) as z:
   x=z['native'];y=z['renderer']
   if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)!=r['paired_windows'] or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('finite matched arrays required')
   errors=np.sqrt(np.mean((x-y)**2,axis=0))
   for j,axis in enumerate(('roll','pitch','yaw')):
    if not np.isclose(errors[j],r['metrics'][axis]['rmse'],rtol=1e-9,atol=1e-12):raise ValueError('independent RMSE mismatch')
  bundles.append(bundle['sha256']);reports.append({'source':name,'windows':len(x),'metrics':r['metrics'],'availability_delay_median_ms':r['availability_delay_median_ms'],'availability_delay_max_ms':r['availability_delay_max_ms'],'events_sha256':manifest['events_sha256'],'camera_sha256':bundle['sha256'],'comparison_sha256':r['comparison_sha256']})
 if len(set(bundles))!=3:raise ValueError('distinct source archives required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_comparison_sha256=[x['comparison_sha256'] for x in reports],optimizer_updates=0));result={'status':'completed_additional_rotation_audits_crosscheck','reports':reports,'source_recordings_distinct':True,'independent_physics_or_new_vehicle_verified':False,'source_clock_verified':False,'physical_exposure_sync_verified':False,'axis_validity_changed':False,'optimizer_updates':0,'vehicle_control_sent':False,'caveat':'same car and session, previously examined, overlapping windows, offline endpoint packet availability; numeric gates do not grant actor sensor admission'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
