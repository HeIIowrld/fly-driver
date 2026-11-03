"""Prepared four-view FlyVis and flight shadow for diagnostic recording; no controls."""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .episode_storage import load_array
from .flight_embodied_navigation import EmbodiedVisualFlightCircuit
from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from .flight_stream import held_input_batches
from .flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from .flight_visual_body_temporal_context import (
    PerViewTemporalContextOpponentFlightStream, PerViewVisualBodyContextBoundary,
)
from .flyvis_camera import VIEW_ORDER, FlyvisCameraProjector
from .flyvis_compat import install_datamate_cache_compat
from .graph import sha256
from .relative_airflow_sensor import missing_airflow
from .temporal_body_context import per_view_temporal_body_context
from .virtual_navigation_body import NavigationBodyObservation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class OnlineNeuralShadow:
    """Own visual/flight state for one capture. Fail closed; never reset mid-run."""

    def __init__(self, *, config_path, training, opponent):
        started = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError('Verified GPU backend required')
        torch.set_num_threads(1)
        config, training, opponent = read(config_path), Path(training), Path(opponent)
        result, plan = read(training/'result.json'), read(training/'manifest.json')['config']['plan']
        op = read(opponent/'result.json')
        source = read(Path(plan['source'])/'result.json')
        for key in ('camera_config', 'retina_registration'):
            if sha256(config[key]) != config[key+'_sha256']:
                raise ValueError('Pinned camera and retinal registration required')
        author = Path(config['author_model_dir'])
        if (sha256(author/'best_chkpt') != config['author_weights_sha256']
                or sha256(author/'_meta.yaml') != config['author_meta_sha256']
                or sha256(Path(result['checkpoint'])) != op['source_checkpoint_sha256']
                or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']):
            raise ValueError('Pinned author and exact flight checkpoints required')
        os.environ['FLYVIS_ROOT_DIR'] = str(Path(config['cache_root']).resolve())
        os.environ['MPLBACKEND'] = 'Agg'
        install_datamate_cache_compat(config['cache_root'])
        import flyvis
        from .flyvis_fractional_tail import FractionalVisualTail

        meta = yaml.safe_load((author/'_meta.yaml').read_text(encoding='utf8'))
        with torch.device('cuda'), torch.no_grad():
            self.network = flyvis.Network(**meta['config']['network']).to('cuda')
            weight = torch.load(author/'best_chkpt', map_location='cpu', weights_only=True)['network']
            self.network.load_state_dict(weight, strict=True)
            self.network.eval().requires_grad_(False)
            nodes = self.network.connectome.nodes
            receptors = self.network.stimulus.input_index[0]
            u, v = nodes.u[:][receptors], nodes.v[:][receptors]
            self.projector = FlyvisCameraProjector(read(config['camera_config']), u, v)
            with load_array(config['retina_registration'], allow_pickle=False) as reg:
                np.testing.assert_array_equal(self.projector.uv, reg['image_uv'])
                np.testing.assert_array_equal(self.projector.directions_body, reg['body_directions'])
                self.motion_indices = reg['motion_indices'].copy()
                names = reg['motion_types'].astype(str).copy()
            all_u, all_v, types = nodes.u[:], nodes.v[:], nodes.type[:].astype(str)
            maps = []
            for name in names:
                mapping = {(int(all_u[i]), int(all_v[i])): i for i in np.flatnonzero(types == name)}
                maps.append([mapping[(int(q), int(r))] for q, r in zip(u, v, strict=True)])
            np.testing.assert_array_equal(maps, self.motion_indices)
            model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
                source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
            model.load_state_dict(torch.load(result['checkpoint'], map_location='cpu',
                                             weights_only=True)['model'], strict=True)
            model.eval().requires_grad_(False)
            temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
                model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to('cuda')
            temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt', map_location='cpu',
                                                weights_only=True)['model'], strict=True)
            joint = ExperimentalVisualBodyAirflow(model).to('cuda').eval().requires_grad_(False)
            boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device='cuda')
            self.visual = torch.zeros(1, 6, device='cuda')
            self.visual_valid = torch.zeros(1, 6, dtype=torch.bool, device='cuda')
            self.scene_valid = torch.ones(1, 4, dtype=torch.bool, device='cuda')
            self.goals = torch.tensor([[0., .5]], device='cuda')
            self.vis_state = self.network.steady_state(1., .005, batch_size=4, value=.5)
            self.network.simulate = FractionalVisualTail(self.network, self.vis_state).simulate
            grey = torch.full((4, 1, 1, 721), .5, device='cuda').expand(-1, 8, -1, -1)
            self.network.simulate(grey, dt=.005, initial_state=self.vis_state, as_states=True)
            warm = PerViewTemporalContextOpponentFlightStream(joint, temporal, boundary,
                                                               optimize_held=True)
            contract = sha256('configs/ac-virtual-navigation-body-v2.json')
            values, valid = np.zeros(21, np.float32), np.ones(21, bool)
            for tick in range(2):
                stamp = 10.+tick*.08
                old = NavigationBodyObservation('disposable-warmup', tick, stamp-.04,
                                                stamp-.01, values, valid, contract)
                now = NavigationBodyObservation('disposable-warmup', tick+2, stamp-.005,
                                                stamp, values, valid, contract)
                context = per_view_temporal_body_context((old,)*4, now,
                    render_start_lowers=(stamp-.02,)*4, camera_receipt_time=stamp-.01,
                    decision_time=stamp)
                intent, _ = warm.decision(decision_time=stamp, visual=self.visual,
                    visual_valid=self.visual_valid, body_context=context, goals=self.goals,
                    scene_features=torch.zeros(1, 576, device='cuda'), scene_valid=self.scene_valid,
                    airflow_observation=missing_airflow(stamp))
                intent.cpu()
            torch.cuda.synchronize()
            del warm
            self.stream = PerViewTemporalContextOpponentFlightStream(joint, temporal, boundary,
                                                                     optimize_held=True)
        self.model, self.held, self.previous_time = model, None, None
        self.flight_active, self.flight_stopped, self.failed = False, False, False
        self.initialization_seconds = time.perf_counter()-started
        self.provenance = {'config_sha256': sha256(config_path),
            'author_weights_sha256': config['author_weights_sha256'],
            'flight_checkpoint_sha256': op['source_checkpoint_sha256'],
            'opponent_checkpoint_sha256': op['checkpoint_sha256'],
            'warmup': 'disposable visual and flight states; new actual recurrent states',
            'initialization_seconds': self.initialization_seconds}

    @torch.no_grad()
    def process(self, frame, event, context, directory):
        if self.failed:
            raise RuntimeError('Failed neural shadow cannot be silently resumed')
        started, decision = time.perf_counter(), event['decision']
        try:
            if self.previous_time is not None and not 0 < decision-self.previous_time <= .15:
                raise ValueError('Original camera decision gap exceeds 150ms; no reset or filling')
            views = {view['view_id']: view for view in frame['views']}
            retinal, hashes = [], []
            for name in VIEW_ORDER:
                expected = f"{event['index']:04d}-{name}.png"
                if views[name].get('file') != expected:
                    raise ValueError('Exact same capture image identity required')
                data = (Path(directory)/expected).read_bytes()
                hashes.append(hashlib.sha256(data).hexdigest())
                with Image.open(io.BytesIO(data)) as picture:
                    retinal.append(self.projector.sample(np.asarray(picture.convert('RGB')).copy()))
            with torch.device('cuda'):
                current = torch.as_tensor(np.stack(retinal), dtype=torch.float32, device='cuda')
                if self.previous_time is not None:
                    for dt, count in held_input_batches(decision-self.previous_time):
                        while count:
                            n = min(count, 16)
                            source = self.held[:, None, None, :].expand(-1, n, -1, -1)
                            self.vis_state = self.network.simulate(source, dt=dt,
                                initial_state=self.vis_state, as_states=True)[-1]
                            count -= n
                motion = self.vis_state.nodes.activity[:, self.motion_indices]
                scene = self.model.backbone.scene_pool(torch.cat((current[:, None], motion), dim=1)[None])
                required = [0, 1, 4, 16, 17, 18, 19, 20]
                valid = context is not None and bool(context.decision_body.valid[required].all())
                intent = None
                if not valid and self.flight_active:
                    self.flight_stopped = True
                if valid and not self.flight_stopped:
                    self.flight_active = True
                    intent, _ = self.stream.decision(decision_time=decision, visual=self.visual,
                        visual_valid=self.visual_valid, body_context=context, goals=self.goals,
                        scene_features=scene, scene_valid=self.scene_valid,
                        airflow_observation=missing_airflow(decision))
                    intent = intent.cpu().numpy()[0].tolist()
                torch.cuda.synchronize()
                self.held, self.previous_time = current.detach().clone(), decision
            finished = time.perf_counter()
            body_receipt = None if context is None else context.decision_body.original_received_monotonic_seconds
            return {'status': 'intent' if intent is not None else (
                'flight_stopped_after_body_gap' if self.flight_stopped else 'missing_required_body'),
                'intent': intent, 'started': started, 'finished': finished,
                'processing_seconds': finished-started,
                'camera_receipt_to_output_seconds': finished-event['camera_receipt'],
                'body_receipt_age_at_output_seconds': None if body_receipt is None else finished-body_receipt,
                'view_render_age_upper_at_output_seconds': [finished-x for x in event['render_start_lower']],
                'source_images_sha256': hashes, 'source_images_hashed_at_neural_read': True,
                'physical_sample_or_exposure_time_verified': False,
                'flight_segment_started': self.flight_active, 'vehicle_control': False}
        except Exception:
            self.failed = True
            raise
