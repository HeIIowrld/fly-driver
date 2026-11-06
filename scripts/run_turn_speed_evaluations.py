"""Owned turn/speed evaluations, with speed-route ablation on development replay."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path

def main():
 root=Path('runs/visual-turn-speed-20261104');tasks=[]
 for arm in ('turn_only','turn_speed'):
  cp=str(root/f'{arm}-fit-01/turn-speed.pt');name=f'reserved-{arm}-01';tasks.append((name,['-m','scripts.evaluate_turn_speed_factorial','--stimuli',str(root/'reserved-stimuli-01'),'--route-checkpoint',cp,'--output',str(root/name)]))
  for episode in range(3):
   name=f'actual-{arm}-{episode}-01';tasks.append((name,['-m','scripts.evaluate_turn_speed_recording','--checkpoint',cp,'--episode',str(episode),'--output',str(root/name)]))
 name='reserved-parent-01';tasks.append((name,['-m','scripts.evaluate_turn_speed_factorial','--stimuli',str(root/'reserved-stimuli-01'),'--output',str(root/name)]));name='actual-turn_speed-off-2-01';tasks.append((name,['-m','scripts.evaluate_turn_speed_recording','--checkpoint',str(root/'turn_speed-fit-01/turn-speed.pt'),'--speed-off','--episode','2','--output',str(root/name)]))
 def run(task):
  name,args=task;start=time.time()
  with (root/f'{name}.log').open('w',encoding='utf-8') as log:
   proc=subprocess.Popen([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy());pid=proc.pid;code=proc.wait()
  return dict(name=name,pid=pid,exit_code=code,started_unix=start,finished_unix=time.time(),args=args)
 receipts=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for future in concurrent.futures.as_completed([pool.submit(run,t) for t in tasks]):
   receipt=future.result();receipts.append(receipt);(root/'evaluation-process-receipts-01.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
 return int(any(r['exit_code'] for r in receipts))
if __name__=='__main__':raise SystemExit(main())
