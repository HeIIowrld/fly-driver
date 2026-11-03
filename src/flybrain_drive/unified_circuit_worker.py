"""One sequential, state-owning CPU process for the existing unified circuit.

Research replay only: no simulator handles, actuator, or physical time claims.
"""
import multiprocessing as mp
import time
import numpy as np
import torch
from .unified_flight_sensorimotor import UnifiedFlightSensorimotor


def _serve(connection, model, goals):
    try:
        torch.set_num_threads(1)
        if type(model) is not UnifiedFlightSensorimotor or model.previous_time is not None:
            raise ValueError('Fresh exact unified circuit required')
        model.eval().requires_grad_(False)
        connection.send({'status':'ready','pid':mp.current_process().pid})
        sequence=0
        while True:
            message=connection.recv()
            if message.get('kind')=='reset':
                model.reset_episode();connection.send({'status':'reset','sequence':sequence});continue
            if message.get('kind')=='stop':
                connection.send({'status':'stopped','sequence':sequence});return
            if message.get('kind')!='step' or message.get('sequence')!=sequence+1:
                raise ValueError('Exactly ordered single-owner decisions required')
            scene=message['scene']
            if not isinstance(scene,np.ndarray) or scene.shape!=(1,576) or scene.dtype!=np.float32 or not np.isfinite(scene).all():
                raise ValueError('Exact finite CPU scene required')
            started=time.perf_counter()
            with torch.no_grad():
                intent,muscles=model.sensory_decision(decision_time=message['when'],racing_goals=goals,
                    scene_features=torch.from_numpy(scene.copy()),scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=message['context'])
            sequence+=1
            connection.send({'status':'ok','sequence':sequence,'started':started,'finished':time.perf_counter(),
                'intent':intent.numpy().copy(),'muscles':muscles.numpy().copy(),
                'state':model.state.numpy().copy(),'drive':model.previous_drive.numpy().copy(),
                'traces':tuple(x.numpy().copy() for x in model.traces)})
    except BaseException as exc:
        try:connection.send({'status':'failed','error':type(exc).__name__+': '+str(exc)})
        except Exception:pass
    finally:connection.close()


class UnifiedCircuitWorker:
    def __init__(self, model, goals, *, timeout=30.):
        if type(model) is not UnifiedFlightSensorimotor or model.previous_time is not None or next(model.parameters()).device.type!='cpu':
            raise ValueError('Fresh exact CPU circuit required')
        if not isinstance(goals,torch.Tensor) or goals.device.type!='cpu' or goals.shape!=(1,2) or not torch.isfinite(goals).all():
            raise ValueError('Finite original CPU goal required')
        if not 0 < timeout <= 60:raise ValueError('Bounded worker timeout required')
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_serve,args=(child,model,goals.detach().clone()));self.timeout=timeout;self.poisoned=False;self.sequence=0
        self.process.start();child.close();ready=self._receive()
        if ready.get('status')!='ready' or ready.get('pid')!=self.process.pid:
            self.poisoned=True;raise RuntimeError('Worker startup outcome invalid')
    def _receive(self):
        if not self.pipe.poll(self.timeout):
            self.poisoned=True;raise TimeoutError('Worker outcome unknown; no retry or reset allowed')
        try:response=self.pipe.recv()
        except (EOFError,OSError):
            self.poisoned=True;raise RuntimeError('Worker connection ended; state outcome unknown')
        if response.get('status')=='failed':
            self.poisoned=True;raise RuntimeError(response.get('error','Worker failed'))
        return response
    def step(self, *, when, scene, context):
        if self.poisoned or not self.process.is_alive():raise RuntimeError('Worker unavailable; no retry')
        self.pipe.send({'kind':'step','sequence':self.sequence+1,'when':when,'scene':scene,'context':context})
        response=self._receive()
        if response.get('status')!='ok' or response.get('sequence')!=self.sequence+1:
            self.poisoned=True;raise RuntimeError('Worker response identity mismatch')
        self.sequence+=1;return response
    def reset(self):
        if self.poisoned or not self.process.is_alive():raise RuntimeError('Worker unavailable; no reset')
        self.pipe.send({'kind':'reset'});response=self._receive()
        if response.get('status')!='reset' or response.get('sequence')!=self.sequence:
            self.poisoned=True;raise RuntimeError('Worker reset receipt mismatch')
        return response
    def close(self):
        if self.poisoned:raise RuntimeError('Unknown worker outcome; inspect process before cleanup')
        self.pipe.send({'kind':'stop'});response=self._receive()
        if response.get('status')!='stopped' or response.get('sequence')!=self.sequence:
            self.poisoned=True;raise RuntimeError('Worker terminal receipt mismatch')
        self.process.join(max(5., self.timeout))
        if self.process.is_alive():raise RuntimeError('Worker remains live; do not restart')
        self.pipe.close()
        if self.process.exitcode!=0:raise RuntimeError('Worker exited unsuccessfully')
        return response
