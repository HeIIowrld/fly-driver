
import hashlib,json,subprocess
from pathlib import Path
from collections import Counter
p=Path('runs/vision-corner-complementary-history-20261112-01')
def load(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
result=load(p/'result.json')
for source,expected in result['source_sha256'].items():
 assert sha(Path(source))==expected,source
development={}
findings={}
paired={}
examples=[]
for track,proof in result['tracks'].items():
 path=p/f'{track}-frames.jsonl'
 assert sha(path)==proof['rows_sha256']
 rows=[json.loads(line) for line in path.open()]
 assert len({(r['run'],r['frame']) for r in rows})==len(rows)
 assert all(m['frames']==len(rows) and m['no_output']==0 for m in proof['metrics'].values())
 subset=[r for r in rows if r['truth']=='approach' and r['label']=='2']
 findings[track]=dict(right_approaches=len(subset),paired_stations={
     g:sum(r['sections'].get(g,0) for r in subset) for g in ('current','history_1','history_2')})
 paired[track]=Counter()
 for r in rows:
  final=r['final']
  if final['edge_old'] is not None or final['t4t5_old'] is not None:continue
  for geometry in ('current','history_1','history_2'):
   e,t=(final[f'{m}_{geometry}'] for m in ('edge','t4t5'))
   if e==t:continue
   offer=e or t
   category=r['truth']
   if category=='approach':
    if str(offer[0])!=r['label']:category='wrong_direction'
    elif max(abs(offer[1]-d) for d in r['truth_endpoints_m'])<=5.:category='joint'
    else:category='direction_only'
   paired[track][f"{geometry}/{category}/{'t4t5' if t else 'edge'}"]+=1
   if category=='joint' and track in ('ks_silverstone1967','spa') and geometry=='history_2':
    examples.append(dict(track=track,frame=r['frame'],direction=r['label'],offer=offer,truth=r['truth_endpoints_m']))
 if track in ('ks_silverstone1967','spa'):continue
 assert track not in proof['training']
 for mode,stats in proof['metrics'].items():
  development.setdefault(mode,Counter()).update(stats)
assert development['t4t5_old']['frames']==6087
expected=load(Path('runs/vision-corner-reacquisition-heldout-20261112-01/verification.json'))['totals']
for mode in ('edge','t4t5'):
 for k in ('left','right','joint','outside','stable'):
  assert development[mode+'_old'][k]==expected[mode][k],(mode,k)
safe=Path.cwd().resolve().as_posix()
revision=subprocess.check_output(['git','-c',f'safe.directory={safe}','rev-parse','HEAD'],text=True).strip()
(p/'verification.json').write_text(json.dumps(dict(status='verified',seed=None,source_revision=revision,
  result_sha256=sha(p/'result.json'),development=development,coverage=findings,paired=paired,examples=examples,
  termination='all_inputs_and_timelines_verified'),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(development=development,coverage=findings,paired={k:v for k,v in paired.items() if k=='ks_silverstone1967'},examples=[e for e in examples if e['track']=='ks_silverstone1967'])))
