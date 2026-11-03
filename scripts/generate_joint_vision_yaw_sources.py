"""Generate separately declared synthetic scenes; sequential owned subprocesses."""
import argparse,json,subprocess,sys
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();planpath=B/'20261101-joint-vision-yaw-new-plan/plan.json';plan=read(planpath)
 if sha256(planpath)!='c771c163269730cdadbdd8204d851553e03c696685015120e70dbea3888aa650' or sha256('scripts/replay_flyvis_synthetic_road_varied.py')!=plan['generator_sha256'] or sha256('src/flybrain_drive/synthetic_road_visual.py')!=plan['road_source_sha256']:raise ValueError('Exact pre-generation plan and source required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),actor=False,optimizer_updates=0));result={'status':'failed','items':[],'optimizer_updates':0,'vehicle_control_sent':False}
 try:
  items=[(role,x) for role in ('fit','validation') for x in plan[role]]
  if len({x['seed'] for _,x in items})!=len(items):raise ValueError('Separate seeds required')
  for role,item in items:
   target=run/(role+'-'+str(item['seed']))
   command=[sys.executable,'scripts/replay_flyvis_synthetic_road_varied.py','--output',str(target),'--texture-seed',str(item['seed']),'--curvature',str(item['curvature']),'--bend-start',str(item['bend_start_m'])]
   with (run/(target.name+'-stdout.txt')).open('w',encoding='utf8') as log:
    child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
   row={'role':role,'seed':item['seed'],'exit_code':child.returncode,'path':str(target)};result['items'].append(row);write_json(run/'progress.json',result)
   if child.returncode!=0:raise RuntimeError('Generator subprocess failed: '+target.name)
   r=read(target/'result.json');m=read(target/'manifest.json')['config']
   if r['status']!='completed_causal_four_view_synthetic_road' or sha256(target/'responses.npz')!=r['responses_sha256'] or m['texture_seed']!=item['seed'] or m['curvature']!=item['curvature'] or m['bend_start_m']!=item['bend_start_m'] or m['program_sha256']!=plan['generator_sha256'] or m['road_generator_sha256']!=plan['road_source_sha256']:raise ValueError('Generated source lineage mismatch')
   row['responses_sha256']=r['responses_sha256'];write_json(run/'progress.json',result)
  result['status']='completed_disjoint_synthetic_sources';result['plan_sha256']=sha256(planpath)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_disjoint_synthetic_sources':raise SystemExit(1)
if __name__=='__main__':main()
