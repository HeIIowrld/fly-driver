"""Exploratory static landmark reprojection on original diagnostic frames."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
ROOT=Path('D:/CodexResearchRuns/flybrain-marker-motion-20261102-01');PLAN=Path('D:/CodexResearchRuns/flybrain-landmark-geometry-plan-20261102-01/plan.json');OUT=Path('D:/CodexResearchRuns/flybrain-landmark-geometry-20261102-01')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def unit(a):
 v=np.asarray(a,dtype=float);n=np.linalg.norm(v)
 if not np.isfinite(v).all() or n<1e-9:raise ValueError('invalid camera basis')
 return v/n
def camera(v):
 forward=unit(v['renderer_only_world_look']);up=unit(v['renderer_only_world_up']);right=unit(np.cross(forward,up));up=unit(np.cross(right,forward));p=ast.literal_eval(v['projection_matrix_diagnostic']);px,py=float(p[0][0]),float(p[1][1]);c=np.array(v['renderer_only_world_position'],dtype=float)
 if min(px,py)<=0 or not np.isfinite(c).all():raise ValueError('bad projection')
 return c,forward,right,up,px,py
def ray(cam,x,y):
 c,f,r,u,px,py=cam
 return unit(f+r*((2*(x+.5)/320-1)/px)+u*((1-2*(y+.5)/180)/py))
def project(cam,xyz):
 c,f,r,u,px,py=cam;v=xyz-c;z=np.dot(v,f)
 if z<=0:return None
 return np.array([((px*np.dot(v,r)/z+1)*320/2)-.5,((1-py*np.dot(v,u)/z)*180/2)-.5])
def match(a,b,x,y):
 rad=4;search=16;patch=a[y-rad:y+rad+1,x-rad:x+rad+1].astype(float);templ=patch-patch.mean();norm=np.linalg.norm(templ)
 if norm/9<7:return None
 region=b[y-search-rad:y+search+rad+1,x-search-rad:x+search+rad+1];windows=sliding_window_view(region,(9,9)).astype(float);windows-=windows.mean(axis=(-2,-1),keepdims=True);den=np.sqrt(np.sum(windows*windows,axis=(-2,-1)))*norm
 score=np.divide(np.einsum('ijkl,kl->ij',windows,templ),den,out=np.full(den.shape,-1.),where=den>1e-8);iy,ix=np.unravel_index(np.argmax(score),score.shape);best=float(score[iy,ix]);masked=score.copy();masked[max(0,iy-2):iy+3,max(0,ix-2):ix+3]=-1
 if best<.88 or best-float(masked.max())<.025:return None
 return x+ix-search,y+iy-search,best
def main():
 plan=json.loads(PLAN.read_text(encoding='utf8'));result=json.loads((ROOT/'result.json').read_text(encoding='utf8'))
 if sha(ROOT/'result.json')!=plan['source_result_sha256'] or result['status']!='completed':raise ValueError('source mismatch')
 for f,key in [('diagnostic-frames.json','metadata_sha256'),('live-frame-hashes.jsonl','host_observed_hashes_sha256')]:
  if sha(ROOT/f)!=result[key]:raise ValueError('source changed')
 rows=json.loads((ROOT/'diagnostic-frames.json').read_text(encoding='utf8'));obs=[json.loads(x) for x in (ROOT/'live-frame-hashes.jsonl').read_text(encoding='utf8').splitlines()];data={}
 for i in range(105,139):
  row=rows[i];record=obs[i];p=Path(row['source_file'])
  if i!=row['index'] or i!=record['index'] or sha(p)!=row['sha256'] or row['sha256']!=record['metadata_sha256']:raise ValueError('metadata provenance')
  f=json.loads(p.read_text(encoding='utf8'));v=next(x for x in f['views'] if x['view_id']=='front');event=v['render_marker_diagnostic'];image=p.parent/v['file']
  if f.get('aid_probe_only') is not True or f.get('training_admissible') is not False or event['index']!=i or sha(image)!=record['images']['front']['sha256']:raise ValueError('image provenance')
  with Image.open(image) as im:gray=np.asarray(im.convert('L'),dtype=np.float32)
  data[i]=(gray,camera(v),float(event['speed_kmh']))
 repro=[];windows=[];counts={'candidate_patches':0,'tracked_triplets':0,'positive_depth':0}
 for i in range(105,133,3):
  triples=[data[i+j] for j in (0,3,6)]
  if min(x[2] for x in triples)<=3:continue
  I0,I1,I2=[x[0] for x in triples];C0,C1,C2=[x[1] for x in triples];n=0
  for y in range(100,160,15):
   for x in range(40,281,20):
    counts['candidate_patches']+=1;one=match(I0,I1,x,y)
    if one is None:continue
    two=match(I1,I2,int(one[0]),int(one[1]))
    if two is None:continue
    counts['tracked_triplets']+=1;a,b=ray(C0,x,y),ray(C1,*one[:2]);D=np.column_stack([a,-b]);depth=np.linalg.lstsq(D,C1[0]-C0[0],rcond=None)[0]
    if min(depth)<=0 or max(depth)>500:continue
    xyz=(C0[0]+a*depth[0]+C1[0]+b*depth[1])/2;pred=project(C2,xyz)
    if pred is None:continue
    counts['positive_depth']+=1;err=float(np.linalg.norm(pred-np.array(two[:2])));repro.append(err);n+=1
  windows.append({'first_frame':i,'third_frame':i+6,'usable_points':n})
 report={'status':'completed_exploratory_landmark_geometry' if repro else 'insufficient_landmarks','plan_sha256':sha(PLAN),'program_sha256':sha(Path(__file__)),'source_result_sha256':sha(ROOT/'result.json'),'counts':counts,'windows':windows,'reprojection_error_px':None if not repro else {'median':float(np.median(repro)),'p95':float(np.percentile(repro,95)),'max':float(max(repro))},'physical_exposure_sync_verified':False,'training_admissible':False,'actor_qualified':False,'limits':plan['caveat']+'; matched patches and multiple windows are correlated, error thresholds not preregistered'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
