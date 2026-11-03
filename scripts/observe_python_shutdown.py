"""Observe Python shutdown after a named owned research script completes."""
import argparse,atexit,faulthandler,json,runpy,sys,threading,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--output',required=True);a=p.parse_args()
if a.target not in ('scripts/replay_archived_visual_live_body_unified.py','scripts/smoke_unified_ticker_native_full_circuit.py'):
 raise ValueError('Explicit diagnostic targets only')
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
trace=(out/'shutdown-trace.txt').open('w',encoding='utf8',buffering=1)
faulthandler.enable(file=trace,all_threads=True)
@atexit.register
def on_exit():
 trace.write('atexit entered '+str(time.time())+'\n');trace.flush()
sys.argv=[a.target,'--output',str(out/'run')]
try:
 runpy.run_path(a.target,run_name='__main__')
finally:
 threads=[{'name':t.name,'ident':t.ident,'daemon':t.daemon,'alive':t.is_alive()} for t in threading.enumerate()]
 (out/'post-main.json').write_text(json.dumps({'time':time.time(),'threads':threads},indent=2),encoding='utf8')
 trace.write('target returned; scheduling traceback in 5 seconds\n');trace.flush()
 faulthandler.dump_traceback_later(5,repeat=False,file=trace)
