"""Research-only process owning native body receipts and one unified circuit.

No actuator or simulator lifecycle authority. Visual contexts retain their original
host timestamps; a separate native stream supplies the current decision body.
"""
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import torch

from .ac_image_buffer import ExistingNamedMapping
from .ac_teacher_status import AI_STATUS_PREFIX
from .evidence import write_json
from .graph import sha256
from .temporal_body_context import per_view_temporal_body_context
from .unified_flight_sensorimotor import UnifiedFlightSensorimotor
from .unified_host_guard import UnifiedHostGuard
from .unified_live_ticker import UnifiedLiveBodyTicker
from .unified_multirate_shadow import UnifiedMultirateShadow
from .virtual_navigation_body import NavigationBodyObservation
from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler


def _warm(model, goals):
    # Existing graph, synthetic data only. Reset numerics and episode metadata.
    values=np.zeros(21);valid=np.ones(21,dtype=bool)
    for i in range(2):
        now=10.+.04*i
        body=NavigationBodyObservation('body-owner-warmup',i,now-.01,now,values.copy(),valid.copy(),'warmup-only')
        ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,
            camera_receipt_time=now-.02,decision_time=now)
        with torch.no_grad():
            model.sensory_decision(decision_time=now,racing_goals=goals,
                scene_features=torch.zeros((1,576)),scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=ctx)
    model.reset_episode()
    if model.previous_time is not None: raise RuntimeError('Explicit warmup reset failed')


def _command_loop(conn, ticker):
    sequence=0
    ticker.owner_sequence=0
    while True:
        msg=conn.recv()
        kind=msg.get('kind')
        if kind=='status':
            conn.send(dict(status='status',sequence=sequence,error=ticker.error,alive=ticker.thread.is_alive()))
        elif kind=='stop':
            return sequence
        elif kind=='publish':
            if msg.get('sequence')!=sequence+1: raise ValueError('Ordered scene publication required')
            if ticker.error or not ticker.thread.is_alive(): raise RuntimeError('Body ticker unavailable: '+str(ticker.error))
            scene=msg.get('scene')
            if not isinstance(scene,np.ndarray) or scene.dtype!=np.float32 or scene.shape!=(1,576) or not np.isfinite(scene).all():
                raise ValueError('Finite original CPU scene required')
            publication=ticker.publish(torch.from_numpy(scene.copy()),msg.get('context'),msg.get('capture'))
            sequence+=1
            ticker.owner_sequence=sequence
            conn.send(dict(status='published',sequence=sequence,publication=publication))
        else:
            raise ValueError('Unknown body-owner command')


def _serve(conn, model, goals, identity, episode, evidence_dir):
    sampler=ticker=None;sequence=0;error=None
    try:
        torch.set_num_threads(1)
        if type(model) is not UnifiedFlightSensorimotor or model.previous_time is not None:
            raise ValueError('Fresh exact unified circuit required')
        model.eval().requires_grad_(False);_warm(model,goals)
        conn.send(dict(status='ready',pid=mp.current_process().pid,warmup_reset=True))
        msg=conn.recv()
        if msg.get('kind')=='stop':return
        if msg.get('kind')!='start': raise ValueError('Explicit start required')
        episode=msg.get('episode',episode)
        if not isinstance(episode,str) or not episode:raise ValueError('Explicit episode required at start')
        sampler=NativeCopySampler(lambda:ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size))
        sampler.start()
        ticker=UnifiedLiveBodyTicker(UnifiedMultirateShadow(UnifiedHostGuard(model),goals=goals),
            native_rows=sampler.rows,native_identity=identity,episode_id=episode)
        ticker.start();conn.send(dict(status='started'))
        sequence=_command_loop(conn,ticker)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        try:conn.send(dict(status='failed',error=error))
        except Exception:pass
    finally:
        failures=[]
        if ticker is not None:
            try:ticker.stop()
            except Exception as exc:failures.append('ticker: '+str(exc))
        if sampler is not None:
            try:sampler.stop()
            except Exception as exc:failures.append('sampler: '+str(exc))
        try:
            dest=Path(evidence_dir);dest.mkdir(parents=True,exist_ok=False)
            events=ticker.events if ticker else [];rows=sampler.rows if sampler else []
            write_json(dest/'ticker-events.json',events);write_json(dest/'native-copy.json',rows)
            sequence=getattr(ticker,'owner_sequence',sequence) if ticker else sequence
            receipt=dict(status='stopped' if error is None and not failures else 'failed',sequence=sequence,
                ticker_error=ticker.error if ticker else None,sampler_error=sampler.error if sampler else None,
                error=error,cleanup_errors=failures,events_sha256=sha256(dest/'ticker-events.json'),
                native_sha256=sha256(dest/'native-copy.json'),events=len(events),copies=len(rows),
                actor_deployed=False,vehicle_control_sent=False)
            write_json(dest/'result.json',receipt)
            try:conn.send(receipt)
            except Exception:pass
        except Exception as exc:
            try:conn.send(dict(status='failed',error='evidence: '+str(exc)))
            except Exception:pass
        conn.close()


class BodyCircuitProcess:
    def __init__(self, model, goals, identity, episode, evidence_dir, *, timeout=60.):
        if type(model) is not UnifiedFlightSensorimotor or model.previous_time is not None:
            raise ValueError('Fresh unified model required')
        if (episode is not None and (not isinstance(episode,str) or not episode)) or not 0<timeout<=120:raise ValueError('Valid episode and timeout required')
        self.episode=episode
        context=mp.get_context('spawn');self.pipe,child=context.Pipe()
        self.process=context.Process(target=_serve,args=(child,model,goals,identity,episode,str(evidence_dir)))
        self.timeout=timeout;self.sequence=0;self.poisoned=False;self.started=False;self.evidence_dir=Path(evidence_dir)
        self.process.start();child.close()
        try:
            ready=self._receive()
            if ready.get('status')!='ready' or ready.get('pid')!=self.process.pid or ready.get('warmup_reset') is not True:
                raise RuntimeError('Body owner startup identity mismatch')
        except BaseException as exc:
            self.pipe.close();self.process.join(5)
            raise RuntimeError(f'Body owner startup failed; pid={self.process.pid}; alive={self.process.is_alive()}: {exc}') from exc
    def _receive(self):
        if not self.pipe.poll(self.timeout):
            self.poisoned=True;raise TimeoutError('Body owner outcome unknown; no retry')
        try: response=self.pipe.recv()
        except (OSError,EOFError) as exc:
            self.poisoned=True;raise RuntimeError('Body owner connection ended') from exc
        if response.get('status')=='failed':
            self.poisoned=True;raise RuntimeError(response.get('error','Body owner failed'))
        return response
    def start(self, episode=None):
        if self.started or self.poisoned:raise RuntimeError('Owner unavailable')
        episode=self.episode if episode is None else episode
        if not isinstance(episode,str) or not episode:raise ValueError('Explicit episode required')
        self.pipe.send(dict(kind='start',episode=episode));response=self._receive()
        if response.get('status')!='started':self.poisoned=True;raise RuntimeError('Start receipt mismatch')
        self.started=True
    def status(self):
        if not self.started or self.poisoned:raise RuntimeError('Owner unavailable')
        self.pipe.send(dict(kind='status'));response=self._receive()
        if response.get('status')!='status' or response.get('sequence')!=self.sequence:
            self.poisoned=True;raise RuntimeError('Status receipt mismatch')
        return response
    def publish(self,scene,context,capture):
        if not self.started or self.poisoned or not self.process.is_alive():raise RuntimeError('Owner unavailable')
        if not isinstance(scene,torch.Tensor) or scene.device.type!='cpu' or scene.dtype!=torch.float32:
            raise ValueError('CPU scene required')
        self.pipe.send(dict(kind='publish',sequence=self.sequence+1,scene=scene.detach().numpy().copy(),context=context,capture=capture))
        response=self._receive()
        if response.get('status')!='published' or response.get('sequence')!=self.sequence+1:
            self.poisoned=True;raise RuntimeError('Publication receipt mismatch')
        self.sequence+=1;return response['publication']
    def collect_failure(self):
        """Read the owner's immutable failed result after process exit; no retry."""
        if not self.poisoned:raise RuntimeError('Only an explicitly failed owner can be collected')
        self.process.join(5.)
        if self.process.is_alive():raise RuntimeError('Failed owner remains live; outcome unknown')
        path=self.evidence_dir/'result.json'
        receipt=json.loads(path.read_text(encoding='utf8'))
        if (receipt.get('status')!='failed' or receipt.get('events_sha256')!=sha256(self.evidence_dir/'ticker-events.json')
                or receipt.get('native_sha256')!=sha256(self.evidence_dir/'native-copy.json')):
            raise ValueError('Matching failed owner evidence required')
        self.pipe.close()
        return receipt
    def close(self):
        if self.poisoned:raise RuntimeError('Owner outcome unknown; inspect pid '+str(self.process.pid))
        self.pipe.send(dict(kind='stop'));response=self._receive()
        if response.get('status')!='stopped' or response.get('sequence')!=self.sequence:
            self.poisoned=True;raise RuntimeError('Terminal receipt mismatch')
        self.process.join(max(5.,self.timeout))
        if self.process.is_alive() or self.process.exitcode!=0:raise RuntimeError('Owner process remains live or failed')
        self.pipe.close();return response
