"""CPU-only reproduction of spawned Torch object/pipe exit; no GPU use."""
import multiprocessing as mp
import os,time
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def child(pipe,pool):
    torch.set_num_threads(1)
    value=float(pool(torch.ones((1,4))).sum())
    pipe.send(dict(status='ready',pid=os.getpid(),value=value))
    msg=pipe.recv()
    if msg!='stop': os._exit(1)
    pipe.send(dict(status='stopped'))
    pipe.close()
    os._exit(0)

def main():
    run=begin_run('D:/CodexResearchRuns/flybrain-cpu-spawn-exit-20261102-01',dict(program_sha256=sha256(__file__),gpu_initialized=False))
    context=mp.get_context('spawn');parent,other=context.Pipe();pool=torch.nn.Linear(4,4).cpu().eval();process=context.Process(target=child,args=(other,pool));start=time.perf_counter();process.start();other.close()
    result=dict(pid=process.pid,gpu_initialized=False,status='failed')
    try:
        if not parent.poll(20):raise TimeoutError('ready outcome unknown')
        result['ready']=parent.recv();parent.send('stop')
        if not parent.poll(20):raise TimeoutError('stop outcome unknown')
        result['terminal']=parent.recv();process.join(20)
        result.update(alive=process.is_alive(),exitcode=process.exitcode,elapsed=time.perf_counter()-start)
        if result['ready']['status']=='ready' and result['ready']['pid']==process.pid and result['terminal']['status']=='stopped' and not result['alive'] and result['exitcode']==0:result['status']='completed'
    except Exception as e:result.update(error=type(e).__name__,reason=str(e),alive=process.is_alive())
    write_json(run/'result.json',result);print(result,flush=True)
if __name__=='__main__':main()
