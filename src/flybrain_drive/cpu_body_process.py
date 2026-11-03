"""Owned CPU flight worker for research replay; no simulator or actuator access."""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from .flight_embodied_navigation import EmbodiedVisualFlightCircuit
from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from .flight_visual_body_temporal_context import PerViewTemporalContextOpponentFlightStream, PerViewVisualBodyContextBoundary
from .graph import sha256
from .multirate_flight_stream import MultirateFlightStream
from .temporal_body_context import per_view_temporal_body_context
from .virtual_navigation_body import NavigationBodyObservation


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _construct(training, opponent):
    training, opponent = Path(training), Path(opponent)
    tr, plan, op = _read(training/'result.json'), _read(training/'manifest.json')['config']['plan'], _read(opponent/'result.json')
    source = _read(Path(plan['source'])/'result.json')
    if (sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']):
        raise ValueError('Exact original checkpoints required')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(source['checkpoint'], map_location='cpu', weights_only=True))
    model.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
    model.backbone.circuit.sparse_backend = 'scipy_csr_cpu'
    model.eval().requires_grad_(False)
    temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
        model.backbone.scene_command.scale, len(model.backbone.vision_ports))
    temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt', map_location='cpu', weights_only=True)['model'], strict=True)
    temporal.eval().requires_grad_(False)
    joint = ExperimentalVisualBodyAirflow(model).eval().requires_grad_(False)
    boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device='cpu')
    return MultirateFlightStream(PerViewTemporalContextOpponentFlightStream(joint, temporal, boundary, optimize_held=True))


def _serve(connection, training, opponent):
    try:
        with torch.no_grad(), torch.device('cpu'):
            stream = _construct(training, opponent)
            kwargs = dict(goals=torch.tensor([[0., .5]]), visual=torch.zeros(1, 6),
                          visual_valid=torch.zeros(1, 6, dtype=torch.bool),
                          scene_valid=torch.ones(1, 4, dtype=torch.bool))
            # Warm caches with disposable state before an actual episode exists.
            contract = sha256('configs/ac-virtual-navigation-body-v2.json')
            values, valid = np.zeros(21, np.float32), np.ones(21, bool)
            for tick in range(2):
                stamp = 10. + tick*.08
                body = NavigationBodyObservation('worker-warmup', tick+2, stamp-.005,
                                                 stamp, values, valid, contract)
                if tick == 0:
                    old = NavigationBodyObservation('worker-warmup', tick, stamp-.04,
                                                     stamp-.01, values, valid, contract)
                    frame = per_view_temporal_body_context((old,)*4, body,
                        render_start_lowers=(stamp-.02,)*4, camera_receipt_time=stamp-.01,
                        decision_time=stamp)
                    scene = torch.zeros(1, 576)
                else:
                    frame, scene = None, None
                stream.update(when=stamp, body=body, new_scene=scene,
                              new_visual_context=frame, **kwargs)
            previous = stream.stream
            stream = MultirateFlightStream(PerViewTemporalContextOpponentFlightStream(
                previous.joint, previous.temporal, previous.context_boundary, optimize_held=True))
            connection.send({'status':'ready', 'pid':mp.current_process().pid})
            sequence = 0
            while True:
                command = connection.recv()
                if command['kind'] == 'stop':
                    connection.send({'status':'stopped', 'sequence':sequence})
                    return
                if command['kind'] == 'snapshot':
                    current = stream.stream
                    connection.send({'status':'snapshot', 'sequence':sequence,
                        'state':current.state.numpy().copy(), 'drive':current.previous_drive.numpy().copy(),
                        'traces':[v.numpy().copy() for v in current.traces]})
                    continue
                if command['kind'] != 'update' or command['sequence'] != sequence + 1:
                    raise ValueError('Original ordered, exactly-once body events required')
                scene = command['scene']
                if scene is not None:
                    if not isinstance(scene, np.ndarray) or scene.shape != (1, 576) or scene.dtype != np.float32 or not np.isfinite(scene).all():
                        raise ValueError('Exact finite original CPU scene required')
                    scene = torch.from_numpy(scene.copy())
                start = time.perf_counter()
                intent, _, info = stream.update(when=command['when'], body=command['body'],
                    new_scene=scene, new_visual_context=command['context'], **kwargs)
                finish = time.perf_counter()
                sequence += 1
                connection.send({'status':'updated', 'sequence':sequence, 'intent':intent.numpy().copy(),
                                 'info':info, 'worker_started':start, 'worker_finished':finish})
    except BaseException as error:
        try:
            connection.send({'status':'failed', 'error':type(error).__name__, 'reason':str(error)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class CpuBodyProcess:
    """One owned process and at most one synchronous, ordered update at a time."""

    def __init__(self, training, opponent, *, timeout=60):
        if not 0 < timeout <= 120:
            raise ValueError('Explicit bounded worker timeout required')
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, str(training), str(opponent)), daemon=True)
        self.process.start()
        child.close()
        self.timeout, self.sequence, self.failed = timeout, 0, False
        self.closed, self.forced_termination = False, False
        try:
            response = self._receive('ready')
        except Exception:
            self.close()
            raise
        self.pid = response['pid']

    def _receive(self, expected):
        try:
            ready = self.connection.poll(self.timeout)
        except (EOFError, OSError):
            self.failed = True
            raise RuntimeError('CPU worker connection closed; no automatic replay') from None
        if not ready:
            self.failed = True
            raise TimeoutError('CPU worker response not observed; no automatic replay')
        try:
            response = self.connection.recv()
        except (EOFError, OSError):
            self.failed = True
            raise RuntimeError('CPU worker connection closed; no automatic replay') from None
        if response.get('status') != expected:
            self.failed = True
            raise RuntimeError(f'CPU worker stopped: {response}')
        return response

    def _send(self, payload):
        try:
            self.connection.send(payload)
        except (BrokenPipeError, EOFError, OSError):
            self.failed = True
            raise RuntimeError('CPU worker connection closed; no automatic replay') from None

    def update(self, *, when, body, scene, context):
        if self.failed or not self.process.is_alive():
            raise RuntimeError('CPU worker unavailable; no automatic reset or retry')
        self._send({'kind':'update', 'sequence':self.sequence+1, 'when':when,
                    'body':body, 'scene':scene, 'context':context})
        response = self._receive('updated')
        if response['sequence'] != self.sequence+1:
            self.failed = True
            raise RuntimeError('CPU worker sequence mismatch')
        self.sequence += 1
        return response

    def snapshot(self):
        if self.failed or self.sequence == 0:
            raise RuntimeError('A valid completed event is required')
        self._send({'kind':'snapshot'})
        response = self._receive('snapshot')
        if response['sequence'] != self.sequence:
            self.failed = True
            raise RuntimeError('CPU worker state sequence mismatch')
        return response

    def close(self):
        if self.closed:
            return self.process.exitcode
        try:
            if not self.failed and self.process.is_alive():
                self._send({'kind':'stop'})
                self._receive('stopped')
        finally:
            self.connection.close()
            self.process.join(timeout=1 if self.failed else 10)
            if self.process.is_alive():
                # This handle belongs exclusively to this diagnostic; never restart it.
                self.forced_termination = True
                self.process.terminate()
                self.process.join(timeout=5)
            self.closed = True
        if self.process.is_alive():
            raise TimeoutError('Owned CPU worker did not terminate')
        return self.process.exitcode
