"""Compare adjacent captured pixel changes and native speed; no exposure timing inference."""
import base64,bisect,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
ROOT=Path('D:/CodexResearchRuns/flybrain-stateclock-motion-20261102-01');PLAN=Path('D:/CodexResearchRuns/flybrain-pixel-native-timing-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-pixel-native-timing-20261102-01')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 plan=json.loads(PLAN.read_text(encoding='utf8'));r=json.loads((ROOT/'result.json').read_text(encoding='utf8'))
 if sha(ROOT/'result.json')!=plan['source_result_sha256'] or r['status']!='completed':raise ValueError('completed pinned source required')
 for f,key in [('native-copy-windows.json','native_copy_sha256'),('clock-after.json','clock_after_sha256'),('diagnostic-frames.json','metadata_sha256'),('live-frame-hashes.jsonl','host_observed_hashes_sha256')]:
  if sha(ROOT/f)!=r[key]:raise ValueError('source changed '+f)
 clock=json.loads((ROOT/'clock-after.json').read_text());offset=(clock['offset_lower_seconds']+clock['offset_upper_seconds'])/2
 observations=[json.loads(line) for line in (ROOT/'live-frame-hashes.jsonl').read_text(encoding='utf8').splitlines()];frames=json.loads((ROOT/'diagnostic-frames.json').read_text(encoding='utf8'));native=json.loads((ROOT/'native-copy-windows.json').read_text(encoding='utf8'))
 if len(observations)!=len(frames) or len(frames)!=r['frames'] or len(native)!=r['native_copy_samples']:raise ValueError('incomplete evidence')
 times=[];speeds=[]
 for row in native:
  a=row['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=a['prefix_sha256']:raise ValueError('native digest')
  t=row['copy']['second_copy_end'];s=decode_teacher_physics(raw,t)
  if s.physics.packet_id!=a['packet_id']:raise ValueError('native id')
  times.append(t);speeds.append(float(s.physics.fields['speedKmh']))
 if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('native nonmonotonic')
 views=('front','left','right','rear');previous={};signals={v:[] for v in views}
 for row,obs in zip(frames,observations,strict=True):
  p=Path(row['source_file']);i=row['index']
  if obs['index']!=i or row['sha256']!=obs['metadata_sha256'] or sha(p)!=row['sha256']:raise ValueError('metadata identity')
  frame=json.loads(p.read_text(encoding='utf8'))
  if frame['index']!=i or frame['request_id']!=r['request_id'] or frame['generation']!=clock['generation']:raise ValueError('frame identity')
  for v in frame['views']:
   name=v['view_id'];expected=obs['images'][name];imgpath=p.parent/expected['file']
   if v['file']!=expected['file'] or sha(imgpath)!=expected['sha256']:raise ValueError('image identity')
   with Image.open(imgpath) as image:im=np.asarray(image.convert('L'),dtype=np.float32)/255
   if im.shape!=(180,320):raise ValueError('unexpected image')
   t=float(v['render_started_clock'])+offset
   if name in previous:
    last_i,last_t,last_im=previous[name];dt=t-last_t
    if i!=last_i+1 or not .02<=dt<=.3:raise ValueError('missing frame or interval outside declared range')
    signals[name].append(((last_t+t)/2,float(np.mean(np.abs(im-last_im))/dt)))
   previous[name]=(i,t,im)
 lags=np.arange(-5,6)*.08;results={}
 for name in views:
  results[name]={}
  for lag in lags:
   xs=[];ys=[]
   for t,visual in signals[name]:
    target=t+float(lag);ix=bisect.bisect_left(times,target);candidates=[j for j in (ix-1,ix) if 0<=j<len(times)]
    if not candidates:continue
    j=min(candidates,key=lambda k:abs(times[k]-target))
    if abs(times[j]-target)>.01:continue
    xs.append(visual);ys.append(speeds[j])
   if len(xs)<10 or min(np.std(xs),np.std(ys))<1e-12:raise ValueError('insufficient varying pairs')
   results[name][f'{lag:+.2f}']={'frames':len(xs),'pearson_r':float(np.corrcoef(xs,ys)[0,1])}
 OUT.mkdir(parents=True,exist_ok=False)
 report={'status':'completed_exploratory_pixel_change_comparison','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'source_result_sha256':sha(ROOT/'result.json'),'results':results,'physical_exposure_sync_verified':False,'training_admissible':False,'actor_qualified':False,'interpretation':plan['limitations']}
 (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 for name in views:
  v=results[name];best=max(v,key=lambda k:v[k]['pearson_r']);print(name,'zero',round(v['+0.00']['pearson_r'],3),'maximum',best,round(v[best]['pearson_r'],3),'n',v[best]['frames'])
if __name__=='__main__':main()
