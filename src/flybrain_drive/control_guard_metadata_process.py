"""File-backed guard metadata in its own process/GIL; no vehicle controls.

Original read brackets, file mtimes and publisher QPC health clocks survive IPC.
Only500ms guard-metadata observation age is permitted; body/control50ms and
camera150ms budgets are separate and unchanged in their consumers.
"""
import json,math,multiprocessing as mp,threading,time
from pathlib import Path
from .ac_published_state import read_published_json
from .graph import sha256
from .owned_file_change_watch import OwnedFileChangeWatch

SCHEMA='flybrain.control_guard.metadata.v1'

def _observe(sender,stop,heartbeat,profile,stop_file,camera_health):
 count=0;maximum_read=0.;result=dict(status='failed');profile_watch=stop_watch=None
 try:
  profile_watch=OwnedFileChangeWatch(profile);stop_watch=OwnedFileChangeWatch(stop_file)
  profile_hash=sha256(profile);initial_stop=Path(stop_file).exists()
  if profile_watch.poll():raise RuntimeError('Controls profile changed during original baseline hash')
  while not stop.is_set() and count<12000:
   started=time.perf_counter()
   if profile_watch.poll():raise RuntimeError('Controls profile changed; original watched baseline invalid')
   profile_finished=time.perf_counter();hb,mtime=read_published_json(heartbeat);session_finished=time.perf_counter()
   health,health_mtime=read_published_json(camera_health) if camera_health else (None,None)
   health_finished=time.perf_counter();requested=initial_stop or stop_watch.poll();finished=time.perf_counter()
   maximum_read=max(maximum_read,finished-started)
   row=dict(schema=SCHEMA,sequence=count,actor_input=False,read_started=started,read_finished=finished,
    profile_sha256=profile_hash,session=hb,session_mtime=mtime,camera_health=health,
    camera_health_mtime=health_mtime,stop_requested=requested,
    profile_watch_unchanged=True,stop_watch_latched=requested,
    phases_ms=dict(profile_notification=(profile_finished-started)*1000,
     session_read=(session_finished-profile_finished)*1000,camera_health_read=(health_finished-session_finished)*1000,
     stop_notification=(finished-health_finished)*1000))
   payload=json.dumps(row,ensure_ascii=False,allow_nan=False).encode('utf8')
   if len(payload)>65536:raise ValueError('Guard metadata exceeded64KiB IPC bound')
   sender.send_bytes(payload);count+=1
   if stop.wait(.05):break
  if not stop.is_set():raise RuntimeError('Finite guard metadata sample budget exhausted')
  result['status']='completed'
 except BaseException as error:result.update(error=type(error).__name__,reason=str(error))
 finally:
  for watcher in (profile_watch,stop_watch):
   if watcher is not None:watcher.close()
  result.update(schema='flybrain.control_guard.metadata.terminal.v1',samples=count,maximum_read_ms=maximum_read*1000)
  try:sender.send_bytes(json.dumps(result).encode('utf8'))
  finally:sender.close()

class ControlGuardMetadataProcess:
 def __init__(self,*,heartbeat_path,profile_path,stop_path,camera_health_path=None,maximum_age=.5):
  if not .02<=maximum_age<=.5:raise ValueError('Guard metadata age at most500ms required')
  self.maximum_age=maximum_age;self.latest=None;self.error=None;self.terminal=None;self.records=[]
  self.ready=threading.Event();self.reader_stop=threading.Event()
  ctx=mp.get_context('spawn');self.stop=ctx.Event();self.reader,sender=ctx.Pipe(duplex=False)
  self.process=ctx.Process(target=_observe,args=(sender,self.stop,str(Path(heartbeat_path).resolve()),
   str(Path(profile_path).resolve()),str(Path(stop_path).resolve()),str(Path(camera_health_path).resolve()) if camera_health_path else None))
  self.process.start();sender.close()
  self.thread=threading.Thread(target=self._drain,name='control-metadata-ipc',daemon=True);self.thread.start()
  if not self.ready.wait(20):
   self.close();raise TimeoutError('Guard metadata process did not become ready before camera/control')
  try:self.snapshot()
  except BaseException:self.close();raise
 def _drain(self):
  try:
   while not self.reader_stop.is_set():
    if not self.reader.poll(.05):continue
    payload=self.reader.recv_bytes(65536);row=json.loads(payload)
    if row.get('schema')==SCHEMA:
     if (row.get('sequence')!=len(self.records) or row.get('actor_input') is not False
         or not all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) for k in ('read_started','read_finished'))
         or not 0<=row['read_started']<=row['read_finished']<=time.perf_counter()):raise ValueError('Original guard metadata order/clock differs')
     self.records.append(payload);self.latest=row;self.ready.set()
    elif row.get('schema')=='flybrain.control_guard.metadata.terminal.v1':
     self.terminal=row;self.ready.set();break
    else:raise ValueError('Unexpected owned guard metadata message')
  except EOFError:
   if self.terminal is None:self.error='Guard metadata process disconnected without terminal'
   self.ready.set()
  except BaseException as error:self.error=str(error);self.ready.set()
 def snapshot(self):
  if self.error:raise RuntimeError('Guard metadata source failed: '+self.error)
  if self.terminal is not None:raise RuntimeError('Guard metadata source ended: '+str(self.terminal))
  row=self.latest
  if row is None or not 0<=time.perf_counter()-row['read_started']<=self.maximum_age:
   raise TimeoutError('Original guard metadata read age exceeded500ms')
  if row['stop_requested']:raise RuntimeError('Owned run stop requested')
  return row
 def wait_for_live(self,generation,epoch,camera_id=None):
  deadline=time.perf_counter()+1.
  while True:
   row=self.snapshot();hb=row['session']
   if (hb['generation']==generation and hb['epoch']==epoch and not hb['paused']
       and not hb['in_menu'] and (camera_id is None or hb.get('camera_request_id')==camera_id)):
    return row
   if time.perf_counter()>deadline:raise TimeoutError('Original live guard metadata not observed before control acquisition')
   time.sleep(.01)
 def close(self):
  self.stop.set();self.process.join(2)
  forced=self.process.is_alive()
  if forced:self.process.terminate();self.process.join(2)
  self.reader_stop.set();self.thread.join(.5);self.reader.close()
  return dict(self.terminal or dict(status='failed',reason='No original metadata terminal receipt'),
   process_pid=self.process.pid,process_exitcode=self.process.exitcode,forced_termination=forced,
   process_alive=self.process.is_alive(),receiver_alive=self.thread.is_alive(),packets=len(self.records),ipc_error=self.error)
