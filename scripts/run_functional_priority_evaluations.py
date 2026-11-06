"""Owned research evaluations; no simulator or actor authority."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path

def main():
    root=Path('runs/functional-priority-learning-20261104');tasks=[]
    for arm in ('supervised','functional_priority'):
        cp=str(root/f'{arm}-fit-01/temporal-gain.pt')
        for role,stimuli in [('reserved',str(root/'reserved-stimuli-01')),('train0','runs/controlled-visual-factorial-20261104/stimuli-01'),('train1','runs/controlled-visual-factorial-20261104/stimuli-02'),('train2','runs/visual-route-protection-20261104/reserved-stimuli-01')]:
            name=f'{role}-{arm}-01';tasks.append((name,['-m','scripts.evaluate_temporal_gain_factorial','--stimuli',stimuli,'--route-checkpoint',cp,'--output',str(root/name)]))
        for episode in range(3):
            name=f'actual-{arm}-{episode}-01';tasks.append((name,['-m','scripts.evaluate_temporal_gain_recording','--checkpoint',cp,'--episode',str(episode),'--output',str(root/name)]))
    def run(task):
        name,args=task;start=time.time()
        with (root/f'{name}.log').open('w',encoding='utf8') as log:
            proc=subprocess.Popen([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy());pid=proc.pid;code=proc.wait()
        return dict(name=name,pid=pid,exit_code=code,started_unix=start,finished_unix=time.time(),args=args)
    receipts=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,t) for t in tasks]):
            receipt=future.result();receipts.append(receipt);(root/'evaluation-process-receipts-01.json').write_text(json.dumps(receipts,indent=2),encoding='utf8');print(json.dumps(receipt),flush=True)
    return int(any(r['exit_code'] for r in receipts))

if __name__=='__main__':raise SystemExit(main())
