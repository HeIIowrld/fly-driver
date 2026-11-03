"""Compare three predeclared author FlyVis component analyses, no selection."""
from pathlib import Path
import json,hashlib,collections

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 planpath=Path('D:/CodexResearchRuns/flybrain-t4d-prespecified-models-plan-20261102-01/plan.json');plan=read(planpath);assert [x['id'] for x in plan['models']]==['000','001','002']
 paths={'000':Path('runs/flyvis-author-grid-analysis-20261029-t1021'),'001':Path('D:/CodexResearchRuns/flybrain-t4d-model-001-analysis-20261102-01'),'002':Path('D:/CodexResearchRuns/flybrain-t4d-model-002-analysis-20261102-01')};assert sha(paths['000']/'result.json')==plan['original_000_result_sha256'];reports={}
 for m in plan['models']:
  mid=m['id'];base=Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000')/mid
  if sha(base/'best_chkpt')!=m['checkpoint_sha256'] or sha(base/'_meta.yaml')!=m['config_sha256']:raise ValueError('registered checkpoint changed')
  folder=paths[mid];r=read(folder/'result.json');assert r['status']=='completed'
  if mid!='000':
   for kind in ('flash','edge'):
    src=Path(f'D:/CodexResearchRuns/flybrain-t4d-model-{mid}-{kind}-20261102-01');manifest=read(src/'manifest.json')['config'];result=read(src/'result.json')
    if manifest['checkpoint_sha256']!=m['checkpoint_sha256'] or result['status']!='completed':raise ValueError('source model mismatch')
    analysis=read(folder/'manifest.json')['config']['inputs'][kind]
    for file in ('result.json','stimulus.json','responses.npz'):
     if sha(src/file)!=analysis[file]:raise ValueError('analysis input mismatch')
  stats={}
  for cell in ('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'):
   rows=[x for x in r['moving_edge'] if x['cell_type']==cell]
   if len(rows)!=6 or len({x['speed'] for x in rows})!=6:raise ValueError('six speeds required')
   stats[cell]={'flash_full':r['flash_index']['-1'][cell],'flash_radius6':r['flash_index']['6'][cell],'contrast_matches':sum(x['expected_contrast_sign_matches'] for x in rows),'maximum_direction_error_deg':max(x['circular_error_degrees'] for x in rows),'minimum_expected_contrast_dsi':min(x['dsi_expected_contrast'] for x in rows)}
  reports[mid]={'analysis_result_sha256':sha(folder/'result.json'),'checkpoint_sha256':m['checkpoint_sha256'],'cells':stats}
 out=Path('D:/CodexResearchRuns/flybrain-t4d-prespecified-models-comparison-20261102-01');out.mkdir(exist_ok=False);result={'status':'completed_three_prespecified_model_comparison','plan_sha256':sha(planpath),'reports':reports,'actor_deployed':False,'optimizer_updates':0,'interpretation':'Three predeclared public checkpoints on same artificial stimuli; no biological, population, driving or model-selection claim.'};(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('Comparison written:',out/'result.json')
 for mid in reports:
  print(mid,{cell:(reports[mid]['cells'][cell]['contrast_matches'],round(reports[mid]['cells'][cell]['maximum_direction_error_deg'],2)) for cell in reports[mid]['cells']})
if __name__=='__main__':main()
