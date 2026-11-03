from pathlib import Path
import numpy as np,json,hashlib,datetime
ref=Path('D:/CodexResearchRuns/flybrain-corner-wind-reference-20261102-01/zero-wind-trajectory.npz')
les=Path('D:/CodexResearchRuns/flybrain-corner-spatial-lesion-20261102-01/ablated-trajectory.npz')
out=Path('D:/CodexResearchRuns/flybrain-corner-anticipation-trajectory-audit-20261102-01');out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
with np.load(ref,allow_pickle=False) as a,np.load(les,allow_pickle=False) as b:
 for speed in (3,6):
  for label,z,suffix in [('intact',a,'missing'),('spatial_ablated',b,'spatial_ablated')]:
   minus=z[f'{speed}_-1_0_{suffix}'];plus=z[f'{speed}_1_0_{suffix}']
   for x in (9.,10.,11.,12.):
    if not all(np.all(np.diff(t[:,1])>0) and t[0,1]<=x<=t[-1,1] for t in (minus,plus)):raise ValueError('nonmonotonic or out of range')
    m=float(np.interp(x,minus[:,1],minus[:,7]));p=float(np.interp(x,plus[:,1],plus[:,7]))
    rows.append(dict(initial_speed=speed,mode=label,x_m=x,minus_steer_rad=m,plus_steer_rad=p,left_minus_right_rad=p-m))
result=dict(status='completed_posthoc_trajectory_description',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={'zero_wind_trajectory':sha(ref),'ablated_trajectory':sha(les)},method='Linear interpolation of saved command (column 7) by pre-update world x (column 1); bend begins at x=12 m. Paired episodes are closed loop and their vehicle states may differ.',conditions=rows,interpretation='Posthoc descriptive comparison only; no causal proof of visual anticipation, no physiological validation, and no new simulation, training, or vehicle control.')
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for r in rows:
 if r['x_m'] in (10.,11.,12.):print(r)
print('OUTPUT',out/'result.json')
