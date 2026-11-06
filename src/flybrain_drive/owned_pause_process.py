"""Preloaded verified CSP pause, with artifact persistence after API/state verification."""
import copy,contextlib,io,multiprocessing as mp,multiprocessing.spawn as spawn,os,sys,time
from pathlib import Path
from .evidence import write_json
from .graph import sha256

class DeferredPauseEvidence:
    def __init__(self,output,parent_manifest):self.output=Path(output).resolve();self.parent=parent_manifest;self.pending={}
    def begin(self,path,config):
        if Path(path).resolve()!=self.output:raise ValueError('Only owned pause output may use frozen evidence')
        self.write(self.output/'manifest.json',dict(config=config,source_sha256=self.parent['source_sha256'],source_root=self.parent['source_root'],git_revision=self.parent['git_revision'],working_directory=str(Path.cwd()),evidence_mode='Parent source snapshot before actuation; critical pause source verified before readiness; no source scan after publication'))
        return self.output
    def write(self,path,value):
        path=Path(path)
        if path.resolve().is_relative_to(self.output):self.pending[path]=copy.deepcopy(value)
        else:write_json(path,value) # Local immutable intent journal/request stay immediate.
    def flush(self):
        for path,value in self.pending.items():write_json(path,value)

def await_owned_camera_idle(read_heartbeat, validate, *, generation, epoch, request_id,
                            clock=time.monotonic, wall=time.time, sleep=time.sleep):
    """Wait for read-only session metadata to catch up with the camera stop receipt."""
    deadline = clock() + 1.
    observations = []
    while clock() < deadline:
        state, modified, _ = read_heartbeat(maximum_wait=min(.5, max(.001, deadline-clock())))
        validate(state, wall()-modified, mutating=True)
        active, owner = state.get('camera_active'), state.get('camera_request_id')
        if state['generation'] != generation or state['epoch'] != epoch:
            raise ValueError('Session changed before owned pause')
        observations.append(dict(active=active, request_id=owner, script_clock=state['script_clock']))
        if active is False and not owner and clock() < deadline:
            return observations
        if active is not True or owner != request_id:
            raise ValueError('Unowned camera state before pause')
        sleep(.01)
    raise TimeoutError('Owned camera idle heartbeat not observed before pause publication')


def _pause_worker(pipe,output,manifest,pin,heldout_magione):
    from scripts import control_ac_session as core
    output=Path(output);output.mkdir(exist_ok=False)
    expected=manifest['source_sha256']['scripts/control_ac_session.py']
    if sha256(core.__file__)!=expected:raise ValueError('Verified lifecycle source changed')
    evidence=DeferredPauseEvidence(output,manifest);saved_begin,saved_write=core.begin_run,core.write_json
    saved_read=core.read_session_heartbeat
    stream=io.StringIO();code=1
    try:
        pipe.send(dict(status='ready',pid=mp.current_process().pid,source_sha256=expected,python=sys.executable,torch_loaded='torch' in sys.modules))
        command=pipe.recv()
        if command.get('kind')=='cancel':code=0;return
        if command.get('kind')!='pause':raise ValueError('One explicit pause only')
        if sha256(core.__file__)!=expected:raise ValueError('Lifecycle source changed after readiness')
        original_argv=sys.argv;sys.argv=['control_ac_session','--command','pause','--expected-lua-sha256',pin,'--execute','--output',str(output)]
        if heldout_magione:
            sys.argv.append('--heldout-magione')
        core.begin_run,core.write_json=evidence.begin,evidence.write
        started=time.perf_counter()
        try:
            if command.get('camera') is not None:
                expected_camera=command['camera']
                status_path=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
                observations=await_owned_camera_idle(
                    lambda **kwargs: saved_read(status_path, **kwargs),
                    core.validate_heartbeat, **expected_camera)
                evidence.write(output/'camera-idle-admission.json', observations)
                def read_owned_heartbeat(*args,**kwargs):
                    heartbeat=saved_read(*args,**kwargs)
                    state=heartbeat[0]
                    if (state['generation']!=expected_camera['generation']
                            or state['epoch']!=expected_camera['epoch']):
                        raise ValueError('Owned pause session identity changed')
                    return heartbeat
                core.read_session_heartbeat=read_owned_heartbeat
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):core.main()
            code=0
        except BaseException as error:
            evidence.write(output/'worker-error.json',dict(type=type(error).__name__,reason=str(error)))
            if not any(p.name=='result.json' for p in evidence.pending):
                evidence.write(output/'result.json',dict(status='failed_before_pause_publication',
                    reason=str(error),error=type(error).__name__,lifecycle_requests_sent=0,ui_actions_sent=0))
        finally:core.begin_run,core.write_json=saved_begin,saved_write;core.read_session_heartbeat=saved_read;sys.argv=original_argv
        result=evidence.pending.get(output.resolve()/'result.json',evidence.pending.get(output/'result.json',{}))
        pipe.send(dict(status='action_complete',result=result,exitcode=code,started=started,verified=time.perf_counter(),artifact_flush_started=False))
        evidence.flush();write_json(output/'worker-flush.json',dict(completed=time.perf_counter(),source_sha256=expected,stdout=stream.getvalue(),api_reissued=False))
    finally:
        core.begin_run,core.write_json=saved_begin,saved_write;core.read_session_heartbeat=saved_read
        pipe.close()
    raise SystemExit(code)

class OwnedPauseProcess:
    def __init__(self,output,parent_manifest,pin,python_executable,library_directory,*,heldout_magione=False):
        from .current_flight_actuator_process import start_with_lean_main
        self.output=Path(output);self.events=[];self.called=False;self.closed=False
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_pause_worker,args=(child,str(output),parent_manifest,pin,heldout_magione));exe=spawn.get_executable();paths=list(sys.path)
        try:mp.set_executable(str(python_executable));sys.path.insert(0,str(library_directory));start_with_lean_main(self.process)
        finally:mp.set_executable(exe);sys.path[:]=paths
        child.close()
        if not self.pipe.poll(30):raise TimeoutError('Pause worker startup unknown before actuation')
        self.ready=self.pipe.recv()
        if self.ready['status']!='ready' or self.ready['pid']!=self.process.pid:raise ValueError('Owned pause worker attestation differs')
    def pause(self, *, camera=None):
        if self.called:raise RuntimeError('Pause request may not be reissued')
        self.called=True;self.pipe.send(dict(kind='pause',camera=camera))
        if not self.pipe.poll(10):raise TimeoutError('Published pause outcome unknown; no retry')
        result=self.pipe.recv();self.events.append(result)
        if result['status']!='action_complete':raise ValueError('Typed pause completion required')
        return result
    def close(self):
        if self.closed:return self.terminal
        if not self.called and self.process.is_alive():self.pipe.send(dict(kind='cancel'))
        self.process.join(60)
        if self.process.is_alive():raise RuntimeError('Pause artifact worker still owns flush; no forced termination')
        self.pipe.close();self.closed=True;self.terminal=dict(pid=self.process.pid,exitcode=self.process.exitcode,original_child_handle=True,forced_termination=False,worker_command_sent=self.called,
            request_issued=(self.events[-1]['result'].get('lifecycle_requests_sent',0)>0 if self.events else None if self.called else False))
        return self.terminal
