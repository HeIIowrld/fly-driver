"""Bounded CPU-only subprocess termination contrast; never initializes the GPU."""
import json, subprocess, sys, time
from pathlib import Path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

def main():
    run=begin_run('D:/CodexResearchRuns/flybrain-cpu-process-exit-20261102-01',dict(program_sha256=sha256(__file__),gpu_initialized=False))
    result={'cases':[],'gpu_initialized':False}
    for name,code in [('python','import os;print("ready",flush=True);os._exit(0)'),('torch_cpu','import torch,os;print("ready",flush=True);os._exit(0)')]:
        start=time.perf_counter()
        proc=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            out,err=proc.communicate(timeout=20)
            result['cases'].append(dict(name=name,pid=proc.pid,returncode=proc.returncode,elapsed=time.perf_counter()-start,stdout=out,stderr=err))
        except subprocess.TimeoutExpired:
            result['cases'].append(dict(name=name,pid=proc.pid,status='timeout_process_left_untouched',elapsed=time.perf_counter()-start))
            break
    result['status']='completed' if len(result['cases'])==2 and all(x.get('returncode')==0 and x.get('stdout')=='ready\n' for x in result['cases']) else 'incomplete'
    write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__': main()
