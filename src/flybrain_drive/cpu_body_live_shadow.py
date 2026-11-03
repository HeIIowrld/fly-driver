"""Research shadow: GPU FlyVis producer and separately owned CPU flight state."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .flight_embodied_navigation import EmbodiedVisualFlightCircuit
from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from .flight_visual_body_temporal_context import (
    PerViewTemporalContextOpponentFlightStream,
    PerViewVisualBodyContextBoundary,
)
from .graph import sha256
from .multirate_live_shadow import SplitCudaMultirateLiveShadow
from .relative_airflow_sensor import missing_airflow
from .temporal_body_context import per_view_temporal_body_context
from .virtual_navigation_body import NavigationBodyObservation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class CpuBodyLiveShadow(SplitCudaMultirateLiveShadow):
    """Pinned GPU visual state and CPU inference-only flight state, never an actor."""

    def __init__(self, *, config_path, training, opponent):
        super().__init__(config_path=config_path, training=training, opponent=opponent)
        training, opponent = Path(training), Path(opponent)
        tr, plan, op = read(training/'result.json'), read(training/'manifest.json')['config']['plan'], read(opponent/'result.json')
        source = read(Path(plan['source'])/'result.json')
        if (sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
                or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']):
            raise ValueError('Original flight and temporal checkpoints required')
        with torch.device('cpu'), torch.no_grad():
            model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
                source['checkpoint'], map_location='cpu', weights_only=True))
            model.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
            model.backbone.circuit.sparse_backend = 'scipy_csr_cpu'
            model.eval().requires_grad_(False)
            temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
                model.backbone.scene_command.scale, len(model.backbone.vision_ports))
            temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',
                map_location='cpu', weights_only=True)['model'], strict=True)
            joint = ExperimentalVisualBodyAirflow(model).eval().requires_grad_(False)
            boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device='cpu')
            self.visual = torch.zeros(1, 6)
            self.visual_valid = torch.zeros(1, 6, dtype=torch.bool)
            self.scene_valid = torch.ones(1, 4, dtype=torch.bool)
            self.goals = torch.tensor([[0., .5]])
            contract = sha256('configs/ac-virtual-navigation-body-v2.json')
            values, valid = np.zeros(21, np.float32), np.ones(21, bool)
            warm = PerViewTemporalContextOpponentFlightStream(joint, temporal, boundary, optimize_held=True)
            for tick in range(2):
                stamp = 10. + tick*.08
                prior = NavigationBodyObservation('cpu-warmup', tick, stamp-.04, stamp-.01,
                                                  values, valid, contract)
                current = NavigationBodyObservation('cpu-warmup', tick+2, stamp-.005, stamp,
                                                    values, valid, contract)
                context = per_view_temporal_body_context((prior,)*4, current,
                    render_start_lowers=(stamp-.02,)*4, camera_receipt_time=stamp-.01, decision_time=stamp)
                warm.decision(decision_time=stamp, visual=self.visual, visual_valid=self.visual_valid,
                    body_context=context, goals=self.goals, scene_features=torch.zeros(1, 576),
                    scene_valid=self.scene_valid, airflow_observation=missing_airflow(stamp))
            del warm
            self.stream = PerViewTemporalContextOpponentFlightStream(joint, temporal, boundary, optimize_held=True)
        self.provenance['body_device'] = 'cpu_scipy_csr_fixed_inference_only'
        self.provenance['scene_transfer'] = 'one GPU-to-CPU copy on new published scene; timed by body ticker'

    def enable_diagnostic_cuda_timing(self):
        raise ValueError('Body CUDA event timing is invalid for CPU body shadow')

    def _body_device_scope(self):
        return torch.device('cpu')

    def _body_scope(self):
        return nullcontext()

    def _body_thread_time(self):
        return time.thread_time()

    def _prepare_body_scene(self, scene):
        if not isinstance(scene, torch.Tensor) or scene.device.type != 'cuda' or scene.shape != (1, 576):
            raise ValueError('Original completed four-view GPU scene required')
        return scene.detach().to(device='cpu', copy=True)

    def _sync_body(self):
        pass
