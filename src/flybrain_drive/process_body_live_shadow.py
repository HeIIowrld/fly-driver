"""Research-only GPU visual shadow with an independently owned CPU process."""

import threading
from contextlib import nullcontext

import numpy as np
import torch

from .cpu_body_process import CpuBodyProcess
from .multirate_live_shadow import SplitCudaMultirateLiveShadow
from .virtual_navigation_body import NavigationBodyEncoder


class ProcessBodyAdapter:
    """Adapt ordered synchronous worker events to the ticker's research interface."""

    def __init__(self, worker):
        self.worker = worker
        self.previous_time = None

    def update(self, *, when, body, goals, visual, visual_valid, scene_valid,
               new_scene=None, new_visual_context=None):
        if (not isinstance(goals, torch.Tensor) or goals.device.type != 'cpu'
                or goals.shape != (1, 2) or not torch.equal(goals, torch.tensor([[0., .5]], device='cpu'))
                or not isinstance(visual, torch.Tensor) or visual.device.type != 'cpu'
                or visual.shape != (1, 6) or torch.any(visual != 0)
                or not isinstance(visual_valid, torch.Tensor) or visual_valid.device.type != 'cpu'
                or visual_valid.dtype != torch.bool or visual_valid.shape != (1, 6) or visual_valid.any()
                or not isinstance(scene_valid, torch.Tensor) or scene_valid.device.type != 'cpu'
                or scene_valid.dtype != torch.bool or scene_valid.shape != (1, 4) or not scene_valid.all()):
            raise ValueError('Original declared task and sensor masks required')
        if new_scene is not None and (not isinstance(new_scene, np.ndarray)
                or new_scene.shape != (1, 576) or new_scene.dtype != np.float32):
            raise ValueError('Exact copied original CPU scene required')
        response = self.worker.update(when=when, body=body, scene=new_scene, context=new_visual_context)
        intent = response['intent']
        if (not isinstance(intent, np.ndarray) or intent.shape != (1, 2)
                or intent.dtype != np.float32 or not np.isfinite(intent).all()):
            raise ValueError('Finite original worker intent required')
        self.previous_time = float(when)
        return torch.from_numpy(intent.copy()), None, response['info']


class ProcessBodyLiveShadow(SplitCudaMultirateLiveShadow):
    """One GPU visual producer and one owned CPU process; never sends actions."""

    def __init__(self, *, config_path, training, opponent):
        super().__init__(config_path=config_path, training=training, opponent=opponent)
        self.worker = CpuBodyProcess(training, opponent)
        try:
            # Initialization is bounded separately; body decisions must not wait indefinitely.
            self.worker.timeout = .05
            self.visual = torch.zeros(1, 6, device='cpu')
            self.visual_valid = torch.zeros(1, 6, dtype=torch.bool, device='cpu')
            self.scene_valid = torch.ones(1, 4, dtype=torch.bool, device='cpu')
            self.goals = torch.tensor([[0., .5]], device='cpu')
            self.provenance['body_device'] = 'owned_cpu_process_scipy_csr_inference_only'
            self.provenance['body_worker_pid'] = self.worker.pid
        except BaseException:
            self.worker.close()
            raise

    def enable_diagnostic_cuda_timing(self):
        raise ValueError('Body CUDA timing is invalid for a CPU process')

    def _body_device_scope(self):
        return torch.device('cpu')

    def _body_scope(self):
        return nullcontext()

    def _prepare_body_scene(self, scene):
        if not isinstance(scene, torch.Tensor) or scene.device.type != 'cuda' or scene.shape != (1, 576):
            raise ValueError('Original completed GPU scene required')
        return scene.detach().to(device='cpu', copy=True).numpy().copy()

    def _sync_body(self):
        pass

    def attach(self, *, native_rows, native_identity, episode_id, body_period=.02):
        if not .01 <= body_period <= .04 or hasattr(self, 'ticker'):
            raise ValueError('One bounded original body ticker required')
        self.native_rows, self.period = native_rows, body_period
        self.body_encoder = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                                  native_identity=native_identity)
        self.episode_id = episode_id
        self.multirate = ProcessBodyAdapter(self.worker)
        self.mail_lock, self.mail = threading.Lock(), None
        self.mail_condition = threading.Condition(self.mail_lock)
        self.mail_version, self.consumed_version = 0, 0
        self.ticker_events, self.ticker_error, self.ticker_stop_reason = [], None, None
        self.ticker_stop_details = None
        self.stop_event = threading.Event()
        self.ticker = threading.Thread(target=self._body_loop, name='process-flight-body-ticker', daemon=True)
        self.ticker.start()

    def stop_ticker(self):
        super().stop_ticker()
        self.worker_exit_code = self.worker.close()
        if self.worker_exit_code != 0:
            raise RuntimeError('Owned body worker did not stop successfully')
