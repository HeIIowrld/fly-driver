"""Versioned source-bound owner. Derived from current_flight_owner; sealed source unchanged."""
import multiprocessing as mp
import multiprocessing.spawn as spawn
from bisect import bisect_left
from pathlib import Path
import sys,time,threading,copy
from .evidence import write_json
from .sac_unified_student import read
from .graph import sha256

def receive_with_history(pipe,selector,mode):
    """Maintain causal body history while idle; no flight inference or sensor re-stamping."""
    if mode!='live_readonly':return pipe.recv()
    while not pipe.poll(.005):selector._drain_native()
    selector._drain_native()
    return pipe.recv()


def _serve_control_owner(pipe,checkpoint,checkpoint_sha256,directory,flush_denormal,actor_kind,run_id,mailbox_spec=None,native_sample_budget=20000,body_threads=1):
    runtime=sampler=ticker=selector=mailbox=None;trace=[];mode=None;generation=None;request_id=0;last_request=0;outputs=0;poisoned=False;publication_sequence=0;directory=Path(directory)
    pending_context=None
    summary=dict(actor_kind=actor_kind,run_id=run_id,status='failed',vehicle_control_sent=False,optimizer_updates=0,checkpoint_sha256=checkpoint_sha256)
    def stage(name,**fields):
        value=dict(type='stage',request_id=request_id,stage=name,host_time=time.perf_counter(),**fields);trace.append(value);pipe.send(value)
    def reply(status,**fields):pipe.send(dict(type='response',request_id=request_id,status=status,**fields))
    def finish():
        nonlocal summary
        errors=[]
        if ticker is not None:
            try:ticker.stop()
            except Exception as error:errors.append(dict(stage='ticker_stop',reason=str(error)))
        if sampler is not None:
            try:sampler.stop()
            except Exception as error:errors.append(dict(stage='sampler_stop',reason=str(error)))
            for filename,rows in [('owner-native-copy.json',sampler.rows),('owner-rejected-native-copy.json',sampler.source.rejected_rows),('owner-retired-native-copy.json',sampler.source.retired_rows)]:write_json(directory/filename,rows)
            summary.update(native_producer_pid=sampler.source.process.pid,native_producer_exitcode=sampler.source.process.exitcode,native_rows=len(sampler.rows),sampler_error=sampler.error)
        if runtime is not None:runtime.close()
        events=[] if ticker is None else ticker.events;wait=[] if ticker is None else ticker.wait_events
        write_json(directory/'owner-ticker-events.json',events);write_json(directory/'owner-wait-events.json',wait);write_json(directory/'owner-stage-trace.json',trace)
        summary.update(status='stopped' if not errors and not summary.get('owner_error') and not summary.get('request_error') and (ticker is None or ticker.error is None) else 'stopped_failed_observation',mode=mode,outputs=outputs if ticker is None else sum(v['status']=='shadow_output' for v in events),ticker_error=None if ticker is None else ticker.error,closing_errors=errors,stage_trace_sha256=sha256(directory/'owner-stage-trace.json'),ticker_events_sha256=sha256(directory/'owner-ticker-events.json'),vehicle_control_sent=False)
        write_json(directory/'owner-terminal.json',summary);return summary
    try:
        if native_sample_budget==45000:
            sys.setswitchinterval(.001)
        import torch
        if mailbox_spec is not None:
            from .flight_decision_mailbox import FlightDecisionMailbox
            mailbox=FlightDecisionMailbox(mailbox_spec)
        from .current_flight_runtime import load_current_flight_runtime
        from .dual_reference_candidate import load_dual_reference_runtime
        if actor_kind not in ("parent", "dual_guard", "launch_parent", "launch_dual_guard", "balanced_parent", "balanced_dual_guard","compact_parent","compact_dual_guard"):raise ValueError("Explicit actor loader required")
        from .launch_pedal_candidate import load_control_actor_runtime
        loader=lambda checkpoint,digest:load_control_actor_runtime(actor_kind,checkpoint,digest)
        torch.set_num_threads(body_threads);torch.set_flush_denormal(flush_denormal);stage('load_actor_start');runtime,receipt=loader(checkpoint,checkpoint_sha256);receipt.update(actor_kind=actor_kind,motion_features_required=True,body_threads=torch.get_num_threads(),python_switch_interval_seconds=sys.getswitchinterval(),torch_version=torch.__version__,torch_file=torch.__file__);stage('load_actor_finished');summary['actor_receipt']=receipt
        reply('ready',pid=mp.current_process().pid,python_executable=sys.executable,receipt=receipt)
        while True:
            message=receive_with_history(pipe,selector,mode);request_id=message.get('request_id');kind=message.get('kind')
            if type(request_id) is not int or request_id!=last_request+1:raise ValueError('Strictly increasing request ID required')
            last_request=request_id;stage('request_received',kind=kind)
            if kind=='stop':reply('stopped',summary=finish());break
            if poisoned:reply('rejected',reason='Owner poisoned; explicit new run required');continue
            try:
                if kind in ['begin_replay','begin_live']:
                    if mode is not None:raise ValueError('No owner episode reset inside a run')
                    mode='recorded_replay' if kind=='begin_replay' else 'live_readonly';generation=message['generation'];runtime.begin_episode(message['episode']);stage('runtime_episode_begun',mode=mode)
                    if kind=='begin_live':
                        from .native_process_rows import NativeProcessRowsSampler
                        from .current_flight_control_ticker import CurrentFlightControlTicker
                        from .online_body_shadow import OnlineBodyShadow
                        sampler=NativeProcessRowsSampler(maximum_samples=native_sample_budget,drain_on_stop=True);selector=OnlineBodyShadow(directory='.',request_id=runtime.episode,generation=generation,offset_lower=0.,native_rows=sampler.rows,native_identity=message['native_identity']);ticker=CurrentFlightControlTicker(runtime,run_id=run_id,generation=generation,checkpoint_sha256=checkpoint_sha256,native_rows=sampler.rows,native_identity=message['native_identity'],episode_id=runtime.episode,source_error=lambda:sampler.error,flush_denormal=flush_denormal)
                        ticker.decision_mailbox=mailbox
                        stage('sampler_start');sampler.start();stage('sampler_ready',native_producer_pid=sampler.source.process.pid);ticker.start();stage('ticker_ready')
                    reply('begun',episode=runtime.episode,mode=mode);continue
                if kind=='replay_decide':
                    if mode!='recorded_replay':raise ValueError('Recorded replay requires its explicit episode mode')
                    stage('replay_inference_start');decision=runtime.decide(**message['arguments']);stage('replay_inference_finished');outputs+=1;reply('replay_output',decision=decision);continue
                if kind=='status':
                    reply('status',mode=mode,error=None if ticker is None else ticker.error,outputs=outputs if ticker is None else sum(v['status']=='shadow_output' for v in ticker.events));continue
                if mode!='live_readonly':raise ValueError('Live publication requires native-owned episode')
                if kind=='latest':
                    decision=ticker.latest_after(message['after_decision']);reply('decision' if decision is not None else 'no_new_decision',decision=decision);continue
                if kind=='context':
                    if message['generation']!=generation:raise ValueError('Original camera generation differs')
                    stage('native_drain_start',native_rows=len(sampler.rows),position=selector.position);selector._drain_native();stage('native_drain_finished',position=selector.position);bounds=tuple(message['bounds']);gaps=[];starts=[];pending_context=None
                    try:
                        source_lookup_started=time.perf_counter()
                        for lower,upper in zip(bounds,message['uppers'],strict=True):
                            j=bisect_left(selector.history.receipts,lower)-1
                            if j<1:raise ValueError('No noninitial prior body receipt')
                            packet=selector.history.samples[j].legacy.packet_id
                            # The sequential native drain has already verified monotonic IDs.
                            # Keep the first original copy when a packet has duplicates.
                            index=bisect_left(sampler.rows,packet,hi=selector.position,
                                              key=lambda row:row['authority']['packet_id'])
                            if index==selector.position or sampler.rows[index]['authority']['packet_id']!=packet:
                                raise ValueError('Original prior body copy missing from the consumed prefix')
                            row=sampler.rows[index]
                            gap=upper-row['copy']['first_copy_start']
                            if not 0<=gap<=.025:raise ValueError('Original source-to-render bound exceeds25ms')
                            gaps.append(gap);starts.append(row['copy']['first_copy_start'])
                        source_lookup_finished=time.perf_counter()
                        stage('context_selection_start');event=dict(source_lookup_started=source_lookup_started,source_lookup_finished=source_lookup_finished);context=selector._select_context(event,camera_receipt=message['camera_received'],bounds=bounds);stage('context_selection_finished')
                        if context is None or any(v is None for v in context.visual_contexts):raise ValueError('Original prior body context unavailable')
                        pending_context=dict(context=context,context_id=request_id,bounds=bounds,uppers=tuple(message['uppers']),starts=tuple(starts),camera_received=message['camera_received'])
                        ticker.announce_next_visual(capture_id=message['capture_id'],bounds=bounds,camera_received=message['camera_received'],context_id=request_id)
                        reply('context',context_id=request_id,context=context,evidence=event,gaps=gaps)
                    except ValueError as error:reply('context_unavailable',reason=str(error))
                    continue
                if kind=='publish':
                    if message['sequence']!=publication_sequence+1 or message['visual'].generation!=generation:raise ValueError('Original publication sequence/generation differs')
                    if (pending_context is None or message['context_id']!=pending_context['context_id'] or message['visual'].render_start_lowers!=pending_context['bounds'] or message['visual'].camera_received!=pending_context['camera_received']):raise ValueError('Original issued context required')
                    stage('ticker_publish_start');publication=ticker.publish_source(message['visual'],pending_context['context'],uppers=pending_context['uppers'],first_copy_starts=pending_context['starts']);pending_context=None;stage('ticker_publish_finished');publication_sequence+=1;reply('published',sequence=publication_sequence,publication=publication);continue
                raise ValueError('Unknown current-owner request kind')
            except Exception as error:
                poisoned=True;summary['request_error']=dict(request_id=request_id,kind=kind,type=type(error).__name__,reason=str(error));stage('request_rejected',reason=str(error));reply('rejected',reason=type(error).__name__+': '+str(error))
    except BaseException as error:
        summary['owner_error']=type(error).__name__+': '+str(error)
        try:finish()
        except Exception as closing:summary['finish_error']=str(closing);write_json(directory/'owner-terminal.json',summary)
        try:reply('failed',reason=summary['owner_error'])
        except Exception:pass
    finally:
        if mailbox is not None:mailbox.close()
        pipe.close()

class CurrentFlightControlOwnerProcess:
    def __init__(self,*,checkpoint,checkpoint_sha256,directory,python_executable,library_directory,actor_kind,run_id,flush_denormal=True,mailbox_spec=None,native_sample_budget=20000,body_threads=1):
        if native_sample_budget not in (20000,45000) or native_sample_budget==45000 and mailbox_spec is None:
            raise ValueError("Long native copy budget requires an owned decision mailbox")
        if type(body_threads) is not int or body_threads not in (1,2,4) or body_threads!=1 and native_sample_budget!=45000:
            raise ValueError("Parallel body inference requires a bounded moving trial")
        self.directory=Path(directory)
        if actor_kind not in ("parent","dual_guard","launch_parent","launch_dual_guard","balanced_parent","balanced_dual_guard","compact_parent","compact_dual_guard") or not isinstance(run_id,str) or not run_id:raise ValueError("Explicit actor kind and owned run required")
        self.run_id=run_id;self.actor_kind=actor_kind;self.checkpoint_sha256=checkpoint_sha256;self.generation=None;self.last_decision=0
        if not self.directory.is_dir() or not Path(python_executable).is_file() or not Path(library_directory,'torch').is_dir() or sha256(checkpoint)!=checkpoint_sha256:raise ValueError('Owned directory and exact current checkpoint/CPU runtime required')
        self.lock=threading.Lock();self.poisoned=False;self.pending_unknown=False;self.stopped=False;self.sequence=0;self.publication_sequence=0;self.events=[];self.stages=[];self.episode=None;self.mode=None
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_serve_control_gc_owner,args=(child,str(checkpoint),checkpoint_sha256,str(self.directory),flush_denormal,actor_kind,run_id,mailbox_spec,native_sample_budget,body_threads));original_executable=spawn.get_executable();original_path=list(sys.path)
        try:mp.set_executable(str(python_executable));sys.path.insert(0,str(library_directory));self.process.start()
        finally:mp.set_executable(original_executable);sys.path[:]=original_path
        child.close()
        try:
            self.ready=self._receive(0,'ready',time.perf_counter()+30)
            if self.ready['status']!='ready' or self.ready['pid']!=self.process.pid or '+cpu' not in self.ready['receipt']['torch_version'] or Path(self.ready['python_executable'])!=Path(python_executable) or self.ready['receipt']['checkpoint_sha256']!=checkpoint_sha256:raise ValueError('Exact current CPU owner readiness differs')
        except Exception:self.poisoned=True;self.pending_unknown=True;self.stop();raise
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
    def request(self,kind,*,timeout=.05,**payload):
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
    def begin_replay(self,episode):
        value=self.request('begin_replay',timeout=5,episode=episode,generation='recorded-replay');self.episode=episode;self.mode=value['mode'];return value
    def begin_live(self,episode,generation,native_identity):
        value=self.request('begin_live',timeout=30,episode=episode,generation=generation,native_identity=native_identity);self.episode=episode;self.generation=generation;self.mode=value['mode'];return value
    def replay_decide(self,**arguments):return self.request('replay_decide',arguments=arguments)['decision']
    def status(self):return self.request('status')
    @property
    def error(self):return self.status()['error']
    @property
    def output_count(self):return self.status()['outputs']
    def context_for_camera(self,*,bounds,uppers,camera_received,generation,capture_id):return self.request('context',bounds=bounds,uppers=uppers,camera_received=camera_received,generation=generation,capture_id=capture_id)
    def publish(self,visual,context_id):
        value=self.request('publish',sequence=self.publication_sequence+1,visual=visual,context_id=context_id)
        if value['status']!='published' or value['sequence']!=self.publication_sequence+1:self.poisoned=True;raise RuntimeError('Publication response differs')
        self.publication_sequence+=1;return value['publication']
    def latest_after(self,decision_id):
        value=self.request('latest',after_decision=decision_id)
        if value['status']=='no_new_decision':return None
        if value['status']!='decision':self.poisoned=True;raise RuntimeError('Latest decision response differs')
        decision=value['decision']
        try:
            decision.validate(now=time.perf_counter(),run_id=self.run_id,episode_id=self.episode,generation=self.generation,checkpoint_sha256=self.checkpoint_sha256,after_decision=decision_id)
        except Exception:
            self.poisoned=True
            raise
        return decision
    def stop(self):
        if self.stopped:return self.terminal
        response=None;forced=False
        if not self.pending_unknown:
            try:
                self.sequence+=1;self.pipe.send(dict(kind='stop',request_id=self.sequence));response=self._receive(self.sequence,'stop',time.perf_counter()+30)
            except Exception:self.poisoned=True
        self.pipe.close();self.process.join(30)
        if self.process.is_alive():self.process.terminate();self.process.join(5);forced=True
        if self.process.is_alive():raise RuntimeError('Owned current actor process remains live')
        self.stopped=True;write_json(self.directory/'client-request-events.json',self.events);write_json(self.directory/'client-stage-events.json',self.stages);self.terminal=dict(pid=self.process.pid,exitcode=self.process.exitcode,forced_termination=forced,poisoned=self.poisoned,pending_unknown=self.pending_unknown,response=response);write_json(self.directory/'client-terminal.json',self.terminal);return self.terminal


def _serve_control_gc_owner(*args):
    from .ac_runtime_gc import defer_automatic_cyclic_gc
    try:
        with defer_automatic_cyclic_gc() as receipt:
            _serve_control_owner(*args)
    finally:write_json(Path(args[3])/'owner-gc-scope.json',receipt)
