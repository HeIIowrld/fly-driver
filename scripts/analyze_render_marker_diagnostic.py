"""Check diagnostic render callbacks against alternating captured pixels."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('D:/CodexResearchRuns/flybrain-marker-probe-20261102-01');OUT=Path('D:/CodexResearchRuns/flybrain-marker-analysis-20261102-01');PLAN=Path('D:/CodexResearchRuns/flybrain-render-marker-plan-20261102-01/plan.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 result=json.loads((ROOT/'result.json').read_text(encoding='utf8'));manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf8'));archive=json.loads((ROOT/'archive.json').read_text(encoding='utf8'));plan=json.loads(PLAN.read_text(encoding='utf8'))
 if result['status']!='completed' or result['training_admissible'] is not False or result['actor_stream_created'] is not False or manifest['config']['deployed_lua_sha256']!=plan['candidate_sha256']:raise ValueError('exact diagnostic capture required')
 for row in archive:
  if sha(ROOT/row['file'])!=row['sha256']:raise ValueError('archive mismatch')
 images={n:[] for n in ('front','left','right','rear')};events=[]
 for i in range(4):
  frame=json.loads((ROOT/f'camera/frame-{i:04d}.json').read_text(encoding='utf8'))
  if frame['index']!=i or frame['request_id']!=result['request_id'] or frame['training_admissible'] is not False or frame['aid_probe_only'] is not True:raise ValueError('diagnostic frame identity')
  if {v['view_id'] for v in frame['views']}!=set(images):raise ValueError('four views required')
  for view in frame['views']:
   name=view['view_id'];event=view.get('render_marker_diagnostic')
   if not isinstance(event,dict) or event['view_id']!=name or event['index']!=i or event['parity']!=i%2 or not view['render_started_clock']<=event['source_clock']<=view['render_call_finished_clock']:raise ValueError('callback event mismatch')
   with Image.open(ROOT/'camera'/view['file']) as im:a=np.asarray(im.convert('RGB'),dtype=np.float64)
   if a.shape!=(180,320,3):raise ValueError('unexpected image shape')
   images[name].append(a[:,:,0]-a[:,:,1]);events.append({'index':i,'view':name,'callback_after_render_start_ms':1000*(event['source_clock']-view['render_started_clock']),'render_call_duration_ms':1000*(view['render_call_finished_clock']-view['render_started_clock'])})
 effects={}
 for name,im in images.items():
  d=((im[0]+im[2])-(im[1]+im[3]))/2;effects[name]={'red_minus_green_difference_pixels_over_80':int((d>80).sum()),'maximum_difference':float(d.max())}
 report={'status':'completed_marker_pixel_diagnostic','plan_sha256':sha(PLAN),'archive_sha256':sha(ROOT/'archive.json'),'source_result_sha256':sha(ROOT/'result.json'),'program_sha256':sha(Path(__file__)),'frames':4,'view_callbacks':len(events),'events':events,'posthoc_paired_pixel_effects':effects,'physical_exposure_sync_verified':False,'training_admissible':False,'actor_qualified':False,'interpretation':'Callback timestamps occur inside render calls; red/green pixels alternate as observed in stationary images. Diagnostic marker measures rendering association only, not world-state or native-physics sample times; pixel threshold selected after inspection.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))
if __name__=='__main__':main()
