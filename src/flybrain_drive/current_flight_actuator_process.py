"""CPU vJoy owner/watchdog independent of GPU/caller liveness; finite stop audit."""
import multiprocessing as mp
import multiprocessing.spawn as spawn
import sys, time
from pathlib import Path
from .evidence import write_json



def _serve_actuator(pipe, directory, axis, identity, limits, guard_config, mailbox_spec=None):
    from .ac_vjoy import VJoySession, AxisBinding, load_vjoy_sdk
    from .ac_teacher_physics import NativeTeacherPhysicsReader, repeated_teacher_snapshot
    from .ac_native_physics import PHYSICS_PREFIX, PHYSICS_LAYOUT
    from .current_flight_actuation import FlightActuationLease
    from .ac_published_state import read_published_json
    from .graph import sha256
    from .ac_runtime_gc import defer_automatic_cyclic_gc
    from .ac_native_position import NativeDiagnosticPositionReader
    import struct, ctypes, math
    user=ctypes.WinDLL("user32",use_last_error=True)
    user.GetForegroundWindow.restype=ctypes.c_void_p
    session = lease = reader = position_reader = mailbox = None
    result = dict(status='failed',stationary_feedback_verified=False,student_requests=0)
    feedback = []; last_packet=None; first_packet=None; last_receipt=None; last_prefix=None; last_status=0.; request_id=0;neutral_steer=guard_config['native_feedback']['neutral_tolerance_native'];travel=0.;last_guard=None;initial_damage=None;origin=None
    def snapshot():
        nonlocal last_packet, first_packet, last_receipt, last_prefix
        coupled=repeated_teacher_snapshot(reader.mapping);sample=coupled.physics;now=time.perf_counter()
        if first_packet is None:first_packet=sample.packet_id
        if last_packet is not None and sample.packet_id < last_packet:raise ValueError('Native generation reset')
        if sample.packet_id==last_packet and sample.prefix_bytes!=last_prefix:raise ValueError('Repeated native packet changed bytes')
        if sample.packet_id != last_packet:last_packet=sample.packet_id;last_receipt=now;last_prefix=sample.prefix_bytes
        if coupled.builtin_ai_controlled:raise ValueError('AI authority became active')
        fields=sample.fields
        raw=PHYSICS_PREFIX.unpack(sample.prefix_bytes);offset=0;extra={}
        for name,_,width in PHYSICS_LAYOUT:
            if name in ('gear','numberOfTyresOut','carDamage'):extra[name]=raw[offset] if width==1 else raw[offset:offset+width]
            offset+=width
        from .current_flight_launch_diagnostics import decode_launch_diagnostics
        extra.update(decode_launch_diagnostics(coupled.authority_prefix_bytes,last_receipt))
        record=dict(when=now,first_receipt=last_receipt,packet=last_packet,fields=dict(fields),guard=extra,authority=coupled.authority_record())
        feedback.append(record)
        return record
    def guard(now):
        nonlocal last_status,travel,last_guard,initial_damage,origin
        if user.GetForegroundWindow()==guard_config["window_handle"]:raise ValueError("AC became foreground; no focus action permitted")
        if now-last_status >= .05:
            status,mtime=read_published_json(guard_config['status_path'])
            if (not 0<=time.time()-mtime<=2 or status['generation']!=identity['generation']
                    or status['epoch']!=guard_config['epoch'] or status['ai_controlled']
                    or status.get('expert_probe_active') or status.get('in_menu') or status.get('is_online')
                    or status.get('error') or status.get('pending_receipt_id')):
                raise ValueError('Session authority/heartbeat changed')
            if sha256(axis['profile_path'])!=axis['profile_sha256']:raise ValueError('Control profile changed')
            last_status=now
        if (Path(directory)/'stop').exists():raise RuntimeError('Explicit stop file')
        sample=snapshot();f=sample['fields'];d=sample['guard']
        if initial_damage is None:initial_damage=d['carDamage']
        if any(a>b+.01 for a,b in zip(d['carDamage'],initial_damage)):raise ValueError('New native damage')
        if last_guard is not None:travel+=abs(f['speedKmh'])/3.6*(now-last_guard)
        last_guard=now
        if travel>limits['maximum_travel_m']:raise ValueError('Integrated travel limit')
        if abs(f['speedKmh'])>limits['maximum_speed_kmh']:raise ValueError('Native speed limit: '+str(f['speedKmh']))
        if d['numberOfTyresOut']>0:raise ValueError('Native tyres-out limit: '+str(d['numberOfTyresOut']))
        if d['gear']<2:raise ValueError('Native forward-gear limit: '+str(d['gear']))
        if lease.state=='armed':
            if now-last_receipt>.05 or last_packet==first_packet:raise ValueError('Native feedback stopped advancing')
            position=position_reader.snapshot();coordinates=position.coordinates_native_xyz_m
            if origin is None:origin=coordinates
            displacement=math.sqrt(sum((a-b)**2 for a,b in zip(coordinates,origin)))
            sample['diagnostic_position']=position.metadata();sample['displacement_m']=displacement
            if displacement>limits['maximum_travel_m']:raise ValueError('Native diagnostic displacement limit')
        return sample
    with defer_automatic_cyclic_gc():
        try:
            sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
            reader=NativeTeacherPhysicsReader(timeout=.05)
            position_reader=NativeDiagnosticPositionReader()
            first=snapshot()
            if abs(first['fields']['speedKmh'])>.1 or first['fields']['gas']>.02 or first['fields']['brake']<.98:
                raise ValueError('Stationary brake-held native preflight required')
            session=VJoySession(sdk,axis['device_id'],[AxisBinding(**v) for v in axis['bindings']])
            lease=FlightActuationLease(session,identity=identity,clock=time.perf_counter,
                maximum_seconds=limits['maximum_seconds'],maximum_decisions=limits['maximum_decisions'])
            opened=session.open();pipe.send(dict(type='response',request_id=0,status='ready',open_receipt=opened,pid=mp.current_process().pid,python_executable=sys.executable,server_source_sha256=sha256(__file__),torch_loaded='torch' in sys.modules))
            deadline=time.perf_counter()+10.;planned_end=None
            if mailbox_spec is not None:
                from .flight_decision_mailbox import FlightDecisionMailbox,consume_mailbox_step
                mailbox=FlightDecisionMailbox(mailbox_spec)
            while lease.state in ('preparing','armed'):
                now=time.perf_counter()
                if now>=deadline:lease.stop('finite_owner_lifetime');break
                try:
                    if mailbox is not None and lease.state=='armed':
                        consume_mailbox_step(lease,mailbox,guard,planned_end)
                    else:
                        lease.poll()
                        if mailbox is not None and mailbox.stop_requested:lease.stop('explicit_stop')
                        if lease.state in ('armed','preparing'):guard(now)
                    if lease.state not in ('armed','preparing'):break
                    if pipe.poll(0):
                        message=pipe.recv()
                        if message.get('request_id')!=request_id+1:raise ValueError('Actuator RPC sequence differs')
                        request_id+=1;kind=message['kind']
                        if kind=='arm':
                            if first_packet==last_packet or time.perf_counter()-last_receipt>.05:raise ValueError('Live advancing native packets required')
                            held=snapshot()['fields']
                            if held['gas']>.02 or held['brake']<.98 or abs(held['steerAngle'])>neutral_steer:raise ValueError('Neutral brake hold not acknowledged before arm')
                            lease.arm(prepared_since=message['prepared_since'],valid_camera_groups=message['valid_camera_groups'])
                            if mailbox is not None:
                                seconds=message['planned_seconds']
                                if not .1<=seconds<=.8:raise ValueError('Original planned duration bound required')
                                planned_end=time.perf_counter()+seconds
                            value=dict(status='armed')
                        elif kind=='apply':
                            if mailbox is not None:raise ValueError('RPC actuation disabled in mailbox mode')
                            value=dict(status='applied',receipt=lease.apply(message['decision']))
                        elif kind=='stop':lease.stop('explicit_stop');value=dict(status='stopping')
                        else:raise ValueError('Unknown actuator request')
                        pipe.send(dict(type='response',request_id=request_id,**value))
                    time.sleep(.001)
                except BaseException as error:
                    lease.stop(type(error).__name__+': '+str(error));result['error']=str(error);break
        except BaseException as error:
            result['error']=type(error).__name__+': '+str(error)
        finally:
            if lease is not None and session is not None and session.owned:
                lease.stop(lease.stop_reason or 'owner_failure')
                # Keep the brake held while proving stop on three different live packets.
                until=time.perf_counter()+3.;seen=set();terminal=[]
                while time.perf_counter()<until:
                    try:
                        value=snapshot();f=value['fields']
                        if abs(f['speedKmh'])<=.1 and f['gas']<=.02 and f['brake']>=.98 and abs(f['steerAngle'])<=neutral_steer:
                            seen.add(value['packet']);terminal.append(value)
                        else:seen.clear();terminal=[]
                        if len(seen)>=3:result['stationary_feedback_verified']=True;break
                    except BaseException as error:result['stop_feedback_error']=str(error);break
                    time.sleep(.005)
                result['stop_feedback']=terminal
                try:lease.release(stationary_feedback_verified=result['stationary_feedback_verified'])
                except BaseException as error:result['release_error']=str(error)
            if mailbox is not None:mailbox.close()
            if reader is not None:reader.close()
            if position_reader is not None:position_reader.close()
            if lease is not None:
                result.update(student_requests=sum(e['kind']=='student_request' and e.get('receipt',{}).get('driver_accepted') is True for e in lease.events),
                    student_request_attempts=sum(e['kind']=='student_request' for e in lease.events),
                    stop_reason=lease.stop_reason,stop_request_failed=lease.stop_request_failed,events=lease.events,
                    device_released=not session.owned and sdk.GetVJDStatus(axis['device_id'])==1,
                    driver_status_after_release=sdk.GetVJDStatus(axis['device_id']))
                if result['stationary_feedback_verified'] and result.get('device_released') and not lease.stop_request_failed and not result.get('release_error'):
                    result['status']='stopped'
            write_json(Path(directory)/'actuator-feedback.json',feedback)
            write_json(Path(directory)/'actuator-terminal.json',result)
            pipe.close()


class CurrentFlightActuatorProcess:
    def __init__(self, *, directory, axis, identity, limits, guard_config, python_executable, library_directory, mailbox_spec=None):
        import threading
        self.mailbox_client=None
        if mailbox_spec is not None:
            from .flight_decision_mailbox import FlightDecisionMailbox
            self.mailbox_client=FlightDecisionMailbox(mailbox_spec)
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.lock=threading.Lock();self.poisoned=False;self.pending_unknown=False;self.stopped=False
        self.sequence=0;self.events=[];self.stages=[];self.stop_requested=False
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe()
        self.process=ctx.Process(target=_serve_actuator,args=(child,str(directory),axis,identity,limits,guard_config,mailbox_spec))
        original_executable=spawn.get_executable();original_path=list(sys.path)
        try:mp.set_executable(str(python_executable));sys.path.insert(0,str(library_directory));start_with_lean_main(self.process)
        finally:mp.set_executable(original_executable);sys.path[:]=original_path
        child.close()
        try:
            self.ready=self._receive(0,'ready',time.perf_counter()+10)
            from .graph import sha256
            if (self.ready['status']!='ready' or self.ready['pid']!=self.process.pid or self.ready['torch_loaded'] is not False
                    or self.ready['server_source_sha256']!=sha256(__file__) or Path(self.ready['python_executable'])!=Path(python_executable)):raise RuntimeError('Exact lean control-only worker not ready')
        except BaseException:
            self.poisoned=True;self.pending_unknown=True;self.stop();raise

    def _receive(self,request_id,kind,deadline):
        while True:
            remaining=deadline-time.perf_counter()
            if remaining<=0 or not self.pipe.poll(remaining):
                self.poisoned=True;self.pending_unknown=True;raise TimeoutError('Current owner response unknown; no retry: '+kind)
            try:value=self.pipe.recv()
            except (EOFError,OSError) as error:
                self.poisoned=True;self.pending_unknown=True;raise RuntimeError('Current owner connection lost; response unknown; no retry') from error
            if time.perf_counter()>deadline:
                self.poisoned=True;self.pending_unknown=True;raise TimeoutError('Current owner response exceeded deadline; no retry: '+kind)
            if value.get('request_id')!=request_id:self.poisoned=True;self.pending_unknown=True;raise RuntimeError('Current owner response sequence differs')
            if value.get('type')=='stage':self.stages.append(value);continue
            if value.get('type')!='response':self.poisoned=True;raise RuntimeError('Typed current owner response required')
            if value['status'] in ['failed','rejected']:self.poisoned=True;raise RuntimeError(value.get('reason',value['status']))
            return value
    def _request(self,kind,*,timeout=.05,**payload):
        if self.poisoned or self.stopped:raise RuntimeError('Owner unavailable until explicit new run')
        started=time.perf_counter();deadline=started+timeout
        if not self.lock.acquire(timeout=max(0.,deadline-time.perf_counter())):self.poisoned=True;self.pending_unknown=True;raise TimeoutError('Owner request queue exceeded deadline')
        event=dict(kind=kind,started=started,timeout_seconds=timeout,status='failed')
        try:
            self.sequence+=1;event['request_id']=self.sequence;self.pipe.send(dict(kind=kind,request_id=self.sequence,**payload));event['send_finished']=time.perf_counter();value=self._receive(self.sequence,kind,deadline);event.update(status=value['status'],finished=time.perf_counter());return value
        except Exception as error:
            if isinstance(error,(EOFError,OSError)):self.poisoned=True;self.pending_unknown=True
            event.update(finished=time.perf_counter(),error=type(error).__name__,reason=str(error));raise
        finally:self.events.append(event);self.lock.release()
    def request(self,kind,**payload):
        if kind=='stop':self.stop_requested=True
        return self._request(kind,**payload)

    def arm(self,*,prepared_since,valid_camera_groups,planned_seconds=None):
        payload=dict(prepared_since=prepared_since,valid_camera_groups=valid_camera_groups)
        if planned_seconds is not None:payload['planned_seconds']=planned_seconds
        return self.request('arm',**payload)

    def apply(self,decision):return self.request('apply',decision=decision)

    def begin_stop(self):
        if self.mailbox_client is not None:
            self.mailbox_client.request_stop();return dict(status='stopping',transport='independent_mailbox_stop_flag')
        if not self.poisoned and not self.stop_requested and self.process.is_alive():
            return self.request('stop')
        return None

    def stop(self):
        if self.stopped:return self.terminal
        try:self.begin_stop()
        except Exception:pass
        self.pipe.close()
        # Caller failure never kills a braking owner. Its own finite lifetime bounds exit.
        self.process.join(15)
        if self.process.is_alive():raise RuntimeError('Actuator still owns its bounded stop path; do not terminate')
        self.stopped=True
        if self.mailbox_client is not None:self.mailbox_client.close();self.mailbox_client=None
        self.terminal=dict(pid=self.process.pid,exitcode=self.process.exitcode,forced_termination=False,
                           poisoned=self.poisoned,pending_unknown=self.pending_unknown)
        write_json(self.directory/'client-terminal.json',self.terminal)
        write_json(self.directory/'client-request-events.json',self.events)
        return self.terminal


def start_with_lean_main(process):
    """Spawn the control-only worker without reimporting the GPU pilot main module."""
    import types,importlib.util
    previous=sys.modules['__main__'];bootstrap=types.ModuleType('__main__')
    bootstrap.__spec__=importlib.util.find_spec('flybrain_drive.current_flight_actuator_process')
    bootstrap.__file__=__file__;bootstrap.__package__='flybrain_drive'
    try:
        sys.modules['__main__']=bootstrap
        process.start()
    finally:sys.modules['__main__']=previous


def _probe_cpu_bootstrap(pipe):
    pipe.send(dict(torch_loaded='torch' in sys.modules,pid=mp.current_process().pid))
    pipe.close()
