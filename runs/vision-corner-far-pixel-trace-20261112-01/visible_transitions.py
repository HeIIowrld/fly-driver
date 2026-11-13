
"""Quantify distance to visible pixel transitions independently of map-side identities."""

from pathlib import Path
import hashlib,json
import numpy as np

out=Path('runs/vision-corner-far-pixel-trace-20261112-01')
old=Path('runs/vision-silverstone-visible-edges-20261112-01')
def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def errors(points,mark):
 mark=np.asarray(mark)
 d=np.diff(mark,axis=0)
 output=[]
 for row in points:
  if row['station_m']<26:continue
  for kind in ('selected_px','reference_px'):
   pt=row[kind]
   if pt is None or not mark[0,0]<=pt[0]<=mark[-1,0]:continue
   offset=np.asarray(pt)-mark[:-1]
   fraction=np.clip(np.sum(offset*d,axis=1)/np.sum(d*d,axis=1),0.,1.)
   distance=float(np.min(np.linalg.norm(offset-fraction[:,None]*d,axis=1)))
   output.append(dict(station_m=row['station_m'],kind=kind,error_px=distance))
 return output

left={'620':[99,97,96,94,93,93,93],'621':[96,95,94,93,93,92,92],
      '622':[94,94,93,93,93,92,92],'623':[94,94,94,94,93,93,93]}
anchors=[120,124,128,132,136,140,144]
trace=out/'pixel-trace.json'
right=old/'manual-right-boundary.json'
images={f:old/f'front-{f}.png' for f in left}
data={}
for f,a in load(trace)['frames'].items():
 label={'left':list(zip(anchors,left[f])),'right':load(right)['frames'][f]['right_px']}
 data[f]=dict(labels=label,errors={s:errors([r for r in a['rows'] if r['side']==s],label[s]) for s in label})
 for s,points in data[f]['errors'].items():
  print(f,s,{mode:(len(x),round(float(np.median(x)),2) if x else None)
             for mode in ('selected_px','reference_px')
             for x in [[r['error_px'] for r in points if r['kind']==mode]]})
files=[trace,right,Path(__file__),*images.values()]
(out/'visible-transitions.json').write_text(json.dumps(dict(status='diagnostic_only',frames=data,
   method='Left grass transition located from archived pixels by chroma peaks at visually checked columns. Right labels trace the previously visible dark strip and do not establish legal road extent.',
   source_sha256={str(f):sha(f) for f in files}),indent=2)+'\n',encoding='utf8')
