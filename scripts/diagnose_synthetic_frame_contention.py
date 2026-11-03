"""Bounded synthetic CPU/GIL and disk competition diagnostic, no AC control."""
import argparse,hashlib,json,sys,tempfile,threading,time,subprocess
from pathlib import Path
import numpy as np
from flybrain_drive.ac_shared_texture import current_thread_cycles, unpack_rgba_rows
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
 sys.setswitchinterval(.001);run=begin_run(args.output,dict(program_sha256=sha256(__file__),shared_copy_sha256=sha256('src/flybrain_drive/ac_shared_texture.py'),sample_count=150,switch_interval=sys.getswitchinterval(),synthetic=True,actor=False))
 rng=np.random.default_rng(8191);raw=rng.integers(0,256,size=180*1280,dtype=np.uint8).tobytes();expected=np.frombuffer(raw,dtype=np.uint8).reshape(180,320,4)[:,:,:3].copy();payload=raw*3;results={}
 with tempfile.TemporaryDirectory(prefix='flybrain-contention-') as tmp:
  for mode in ('baseline','cpu_thread','cpu_process','disk_thread'):
   stop=threading.Event();started=threading.Event();errors=[]
   def worker():
    try:
     started.set()
     if mode=='cpu_thread':
      while not stop.is_set():
       v=0
       for x in range(5000):v=(v+x*x)&0xffffffff
     else:
      path=Path(tmp)/'probe.bin'
      while not stop.is_set():
       path.write_bytes(payload); data=path.read_bytes()
       if hashlib.sha256(data).digest()!=hashlib.sha256(payload).digest():raise ValueError('disk readback mismatch')
    except Exception as e:errors.append(repr(e))
   thread=None;process=None
   if mode=='cpu_process':
    process=subprocess.Popen([sys.executable,'-c','x=0\nwhile True: x=(x+1)&0xffffffff'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   if mode not in ('baseline','cpu_process'):
    thread=threading.Thread(target=worker);thread.start()
    if not started.wait(2):raise TimeoutError('worker not started')
   rows=[]
   try:
    for i in range(150):
     time.sleep(.005); c0=current_thread_cycles();t0=time.perf_counter();out=unpack_rgba_rows(raw,1280,28);t1=time.perf_counter();c1=current_thread_cycles()
     if not np.array_equal(out,expected):raise ValueError('pixel mismatch')
     rows.append(dict(wall_ms=(t1-t0)*1000,cycles=c1-c0))
   finally:
    stop.set()
    if process is not None:
     process.terminate();process.wait(timeout=5)
    if thread is not None:
     thread.join(5)
     if thread.is_alive():raise RuntimeError('worker failed to stop')
   if errors:raise RuntimeError(str(errors))
   values=np.array([r['wall_ms'] for r in rows]);results[mode]=dict(samples=rows,median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),maximum_ms=float(values.max()),over_5ms=int((values>5).sum()),over_30ms=int((values>30).sum()))
 write_json(run/'measurements.json',results);write_json(run/'result.json',dict(status='completed_synthetic_contention_diagnostic',measurements_sha256=sha256(run/'measurements.json'),actor_qualified=False,real_camera=False,physical_time_verified=False));print(json.dumps({k:{j:v for j,v in value.items() if j!='samples'} for k,value in results.items()}))
if __name__=='__main__':main()
