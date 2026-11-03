"""Sequential FlyVis owner for a four-view diagnostic; no control or clock substitution."""
import multiprocessing as mp
import gc
import os
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from .flyvis_camera import FlyvisCameraProjector
from .flyvis_compat import install_datamate_cache_compat
from .flight_stream import held_input_batches
from .retinotopic_readout import column_indices


class FlyvisEngine:
    def __init__(self, config, modeldir, cache, pool, device='cuda'):
        if device not in ('cpu', 'cuda'): raise ValueError('Explicit CPU or CUDA device required')
        self.device = device
        torch.set_num_threads(1)
        os.environ['FLYVIS_ROOT_DIR'] = str(Path(cache).resolve())
        os.environ['MPLBACKEND'] = 'Agg'
        install_datamate_cache_compat(Path(cache))
        import flyvis
        import importlib
        initialization = importlib.import_module('flyvis.network.initialization')
        # FlyVis caches its default RNG device at import, which can differ from
        # an explicitly selected CPU device on a machine with a GPU.
        original_rng_device = initialization.device
        initialization.device = torch.device(self.device)
        try:
            self._initialize_network(flyvis, config, modeldir, pool)
        finally:
            initialization.device = original_rng_device
        self.held = None
        self.last_visual = None
        self.last_capture = None

    def _initialize_network(self, flyvis, config, modeldir, pool):
        with torch.device(self.device):
            cfg = yaml.safe_load((Path(modeldir) / '_meta.yaml').read_text(encoding='utf8'))['config']['network']
            self.network = flyvis.Network(**cfg).to(self.device)
            self.network.load_state_dict(torch.load(Path(modeldir) / 'best_chkpt', map_location='cpu', weights_only=True)['network'], strict=True)
            self.network.eval().requires_grad_(False)
            nodes = self.network.connectome.nodes
            receptors = self.network.stimulus.input_index[0]
            u, v = nodes.u[:][receptors], nodes.v[:][receptors]
            self.indices = column_indices(nodes.type[:].astype(str), nodes.u[:], nodes.v[:], u, v,
                [f'{kind}{d}' for kind in ('T4','T5') for d in 'abcd'])
            self.projector = FlyvisCameraProjector(config, u, v)
            self.pool = pool.to(self.device).eval().requires_grad_(False)
            self.state = self.network.steady_state(1., .005, batch_size=4, value=.5)
            gray = np.zeros((180,320,3), dtype=np.uint8)
            warm = np.stack([self.projector.sample(gray) for _ in range(4)])
            x = torch.as_tensor(warm, device=self.device)
            with torch.no_grad():
                temp = self.network.simulate(x[:,None,None,:].expand(-1,8,-1,-1), dt=.005, initial_state=self.state, as_states=True)[-1]
                motion = temp.nodes.activity[:,self.indices]
                self.pool(torch.cat((x[:,None,:], motion), dim=1)[None]).to('cpu')
                self._synchronize()
                self.state = self.network.steady_state(1., .005, batch_size=4, value=.5)
                self._synchronize()

    def _synchronize(self):
        if self.device == 'cuda': torch.cuda.synchronize()

    def close(self):
        # Release device tensors before acknowledging the owned terminal.
        self._synchronize()
        self.held = None
        self.state = None
        self.network = None
        self.pool = None
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            self._synchronize()

    def step(self, capture, pixels):
        if type(capture) is not int or capture < 0 or (self.last_capture is not None and capture <= self.last_capture):
            raise ValueError('Strictly increasing original capture required')
        if not isinstance(pixels, tuple) or len(pixels) != 4 or any(not isinstance(p,np.ndarray) or p.dtype != np.uint8 or p.ndim != 3 or p.shape[2] != 3 for p in pixels):
            raise ValueError('Four original RGB views required')
        visual = time.perf_counter()
        sampled = np.stack([self.projector.sample(p) for p in pixels])
        sampled_at = time.perf_counter()
        with torch.no_grad(), torch.device(self.device):
            current = torch.as_tensor(sampled, device=self.device)
            if self.last_visual is not None:
                elapsed = visual - self.last_visual
                if not 0 < elapsed <= .15:
                    raise ValueError('visual processing gap too large')
                for dt, count in held_input_batches(elapsed):
                    while count:
                        n = min(count, 16)
                        self.state = self.network.simulate(self.held[:,None,None,:].expand(-1,n,-1,-1), dt=dt, initial_state=self.state, as_states=True)[-1]
                        count -= n
            self._synchronize()
            motion_at = time.perf_counter()
            motion = self.state.nodes.activity[:,self.indices]
            scene = self.pool(torch.cat((current[:,None,:],motion),dim=1)[None]).to('cpu')
            self._synchronize()
        finished = time.perf_counter()
        self.held = current.detach().clone()
        self.last_visual, self.last_capture = visual, capture
        return dict(scene=scene.numpy().copy(), visual_snapshot=visual, sampling_finished=sampled_at,
                    motion_finished=motion_at, visual_finished=finished)


def _serve(conn, config, modeldir, cache, pool, device):
    sequence = 0
    exit_code = 1
    try:
        engine = FlyvisEngine(config, modeldir, cache, pool, device)
        conn.send(dict(status='ready', pid=os.getpid()))
        while True:
            message = conn.recv()
            if message.get('kind') == 'stop':
                engine.close()
                conn.send(dict(status='stopped', sequence=sequence, device_released=True))
                exit_code = 0
                return
            if message.get('kind') != 'step' or message.get('sequence') != sequence + 1:
                raise ValueError('Exactly ordered visual requests required')
            result = engine.step(message['capture'], message['pixels'])
            sequence += 1
            conn.send(dict(status='ok', sequence=sequence, **result))
    except BaseException as exc:
        try: conn.send(dict(status='failed', error=type(exc).__name__+': '+str(exc)))
        except Exception: pass
    finally:
        conn.close()
        # CUDA/FlyVis may leave non-daemon library threads alive after the
        # acknowledged terminal. Exit only this owned worker after flushing IPC.
        os._exit(exit_code)


class VisualWorkerStartupError(RuntimeError):
    def __init__(self, pid, alive, reason):
        super().__init__(f'Visual worker {pid} startup failed; alive={alive}: {reason}')
        self.pid, self.alive = pid, alive


class FlyvisProcess:
    def __init__(self, config, modeldir, cache, pool, *, timeout=60., device='cuda'):
        if device not in ('cpu', 'cuda'): raise ValueError('Explicit CPU or CUDA device required')
        if not 0 < timeout <= 120: raise ValueError('Bounded timeout required')
        ctx = mp.get_context('spawn')
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=_serve, args=(child, config, str(modeldir), str(cache), pool, device))
        self.timeout, self.poisoned, self.sequence = timeout, False, 0
        self.process.start(); child.close()
        try:
            ready = self._receive()
            if ready.get('status') != 'ready' or ready.get('pid') != self.process.pid:
                raise RuntimeError('Visual worker startup identity mismatch')
        except BaseException as exc:
            self.pipe.close()
            self.process.join(5)
            raise VisualWorkerStartupError(self.process.pid, self.process.is_alive(), str(exc)) from exc
    def _receive(self):
        if not self.pipe.poll(self.timeout):
            self.poisoned = True
            raise TimeoutError('Visual worker outcome unknown; no retry')
        try: response = self.pipe.recv()
        except (EOFError, OSError) as exc:
            self.poisoned = True
            raise RuntimeError('Visual worker connection ended') from exc
        if response.get('status') == 'failed':
            self.poisoned = True
            raise RuntimeError(response.get('error', 'Visual worker failed'))
        return response
    def step(self, capture, pixels):
        if self.poisoned or not self.process.is_alive(): raise RuntimeError('Visual worker unavailable')
        self.pipe.send(dict(kind='step', sequence=self.sequence+1, capture=capture, pixels=pixels))
        result = self._receive()
        if result.get('status') != 'ok' or result.get('sequence') != self.sequence+1:
            self.poisoned = True
            raise RuntimeError('Visual response identity mismatch')
        self.sequence += 1
        return result
    def close(self):
        if self.poisoned:
            self.process.join(5)
            raise RuntimeError('Visual outcome unknown or failed; process alive='+str(self.process.is_alive()))
        self.pipe.send(dict(kind='stop'))
        result = self._receive()
        if result.get('status') != 'stopped' or result.get('sequence') != self.sequence or result.get('device_released') is not True:
            self.poisoned = True
            raise RuntimeError('Visual terminal receipt mismatch')
        self.process.join(max(60., self.timeout))
        if self.process.is_alive() or self.process.exitcode != 0: raise RuntimeError('Visual worker did not exit cleanly')
        self.pipe.close()
        return result
