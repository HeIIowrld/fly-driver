"""Replay one sensory/flight stream, timing real image and receipt work; no vehicle control."""

import argparse
import base64
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

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.flight_visual_body_temporal_context import (
    PerViewTemporalContextOpponentFlightStream, PerViewVisualBodyContextBoundary,
)
from flybrain_drive.flyvis_camera import VIEW_ORDER, FlyvisCameraProjector
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.per_view_body_history import PerViewBodyHistory
from flybrain_drive.relative_airflow_sensor import missing_airflow
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'projection', 'flyvis', 'flight', 'training', 'opponent', 'config', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--warm-flight', action='store_true')
    a = p.parse_args()
    root, projection, replay, flight, training, opponent = map(Path,
        (a.recording, a.projection, a.flyvis, a.flight, a.training, a.opponent))
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    sr, pr, rr, fr, tr, op = (read(x/'result.json') for x in
                               (root, projection, replay, flight, training, opponent))
    checks = ((root/'online-body-shadow.json', sr['shadow_sha256']),
              (root/'native-copy-windows.json', sr['native_copy_sha256']),
              (root/'diagnostic-frames.json', sr['metadata_sha256']),
              (projection/'source-images.json', pr['source_images_sha256']),
              (projection/'retina-body-diagnostic.npz', pr['arrays_sha256']),
              (replay/'responses.npz', rr['responses_sha256']),
              (flight/'outputs.npz', fr['arrays_sha256']),
              (opponent/'temporal-boundary.pt', op['checkpoint_sha256']))
    fm = read(replay/'manifest.json')['config']
    if (any(sha256(path) != expected for path, expected in checks)
            or any(x['training_admissible'] is not False for x in (sr, rr, fr))
            or fm['online_shadow_sha256'] != sr['shadow_sha256']
            or fm['recording_result_sha256'] != sha256(root/'result.json')
            or op['source_checkpoint_sha256'] != sha256(Path(tr['checkpoint']))):
        raise ValueError('Exact diagnostic images, online receipts and checkpoints required')
    cfg = read(Path(a.config))
    for key in ('retina_registration', 'camera_config'):
        if sha256(cfg[key]) != cfg[key+'_sha256']:
            raise ValueError('Pinned retina and camera configuration required')
    author_dir = Path(cfg['author_model_dir'])
    if (sha256(author_dir/'best_chkpt') != cfg['author_weights_sha256']
            or sha256(author_dir/'_meta.yaml') != cfg['author_meta_sha256']):
        raise ValueError('Pinned FlyVis author model required')
    events, native, frames, images = (read(path) for path in
        (checks[0][0], checks[1][0], checks[2][0], checks[3][0]))
    with np.load(checks[4][0]) as f, np.load(checks[5][0]) as v, np.load(checks[6][0]) as g:
        expected_retina, expected_motion = f['retinal'].copy(), v['motion_activity'].copy()
        expected_intent, clocks = g['intent'].copy(), v['group_source_available_clock'].copy()
    np.testing.assert_array_equal(clocks, [x['decision'] for x in events])
    if len(events) != len(frames) or len(images) != len(events)*4:
        raise ValueError('Complete original frame/image identity required')
    plan = read(training/'manifest.json')['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'recording_sha256': sha256(root/'result.json'), 'projection_sha256': pr['arrays_sha256'],
        'flyvis_sha256': rr['responses_sha256'], 'flight_sha256': fr['arrays_sha256'],
        'joint_checkpoint_sha256': op['source_checkpoint_sha256'],
        'opponent_checkpoint_sha256': op['checkpoint_sha256'], 'author_sha256': cfg['author_weights_sha256'],
        'gpu_warmup': ('disposable visual and flight states before fresh episode' if a.warm_flight
                       else 'disposable visual state before fresh episode; flight retains cold first step'),
        'scope': 'offline single-path timing; recorded body selection clocks, no live scheduling',
        'training_admissible': False, 'actor_input': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'training_admissible': False, 'actor_deployed': False}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('Original GPU required')
        torch.set_num_threads(1)
        os.environ['FLYVIS_ROOT_DIR'] = str(Path(cfg['cache_root']).resolve())
        os.environ['MPLBACKEND'] = 'Agg'
        install_datamate_cache_compat(cfg['cache_root'])
        import flyvis
        from flybrain_drive.flyvis_fractional_tail import FractionalVisualTail

        previous_device = torch.get_default_device()
        torch.set_default_device('cuda')
        model_config = yaml.safe_load((author_dir/'_meta.yaml').read_text(encoding='utf8'))
        network = flyvis.Network(**model_config['config']['network']).to('cuda')
        weights = torch.load(author_dir/'best_chkpt', map_location='cpu', weights_only=True)['network']
        network.load_state_dict(weights, strict=True)
        network.eval().requires_grad_(False)
        nodes = network.connectome.nodes
        receptors = network.stimulus.input_index[0]
        u, v = nodes.u[:][receptors], nodes.v[:][receptors]
        projector = FlyvisCameraProjector(read(Path(cfg['camera_config'])), u, v)
        with load_array(cfg['retina_registration'], allow_pickle=False) as reg:
            np.testing.assert_array_equal(projector.uv, reg['image_uv'])
            np.testing.assert_array_equal(projector.directions_body, reg['body_directions'])
            motion_indices = reg['motion_indices'].copy()
            names = reg['motion_types'].astype(str).copy()
        all_u, all_v, types = nodes.u[:], nodes.v[:], nodes.type[:].astype(str)
        recalculated = []
        for name in names:
            lookup = {(int(all_u[i]), int(all_v[i])): i for i in np.flatnonzero(types == name)}
            recalculated.append([lookup[(int(q), int(r))] for q, r in zip(u, v, strict=True)])
        np.testing.assert_array_equal(recalculated, motion_indices)
        model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
            source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
        model.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
        model.eval().requires_grad_(False)
        temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
            model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to('cuda')
        temporal.load_state_dict(torch.load(checks[7][0], map_location='cpu', weights_only=True)['model'], strict=True)
        joint = ExperimentalVisualBodyAirflow(model).to('cuda').eval().requires_grad_(False)
        boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device='cuda')
        identity = read(Path('configs/ac-virtual-body-v1.json'))['native_identity']
        history = PerViewBodyHistory(episode_id=sr['request_id'], native_identity=identity)
        vis_state = network.steady_state(1., .005, batch_size=4, value=.5)
        network.simulate = FractionalVisualTail(network, vis_state).simulate
        # Execute original held path using disposable state before the fresh episode.
        grey = torch.full((4, 8, 1, 721), .5, device='cuda')[:, :1].expand(-1, 8, -1, -1)
        network.simulate(grey, dt=.005, initial_state=vis_state, as_states=True)
        torch.cuda.synchronize()
        visual = torch.zeros(1, 6, device='cuda')
        visual_valid = torch.zeros(1, 6, dtype=torch.bool, device='cuda')
        scene_valid = torch.ones(1, 4, dtype=torch.bool, device='cuda')
        goals = torch.tensor([[0., .5]], device='cuda')
        if a.warm_flight:
            warm = PerViewTemporalContextOpponentFlightStream(joint, temporal, boundary,
                                                               optimize_held=True)
            contract = sha256('configs/ac-virtual-navigation-body-v2.json')
            values = np.zeros(21, np.float32)
            valid = np.ones(21, bool)
            for tick in range(2):
                stamp = 10.+tick*.08
                historical = NavigationBodyObservation('disposable-warmup', tick, stamp-.04,
                    stamp-.01, values, valid, contract)
                current = NavigationBodyObservation('disposable-warmup', tick+2, stamp-.005,
                    stamp, values, valid, contract)
                context = per_view_temporal_body_context((historical,)*4, current,
                    render_start_lowers=(stamp-.02,)*4, camera_receipt_time=stamp-.01,
                    decision_time=stamp)
                intent, _ = warm.decision(decision_time=stamp, visual=visual,
                    visual_valid=visual_valid, body_context=context, goals=goals,
                    scene_features=torch.zeros(1, 576, device='cuda'), scene_valid=scene_valid,
                    airflow_observation=missing_airflow(stamp))
                intent.cpu()
            torch.cuda.synchronize()
            del warm
        stream = PerViewTemporalContextOpponentFlightStream(joint, temporal, boundary,
                                                             optimize_held=True)
        durations, outputs, retinal_errors, motion_errors, intent_errors = [], [], [], [], []
        stages = []
        position, held, active = 0, None, False
        required = [0, 1, 4, 16, 17, 18, 19, 20]
        with torch.no_grad():
            for i, event in enumerate(events):
                t0 = time.perf_counter()
                source_images = []
                for image in images[4*i:4*i+4]:
                    data = Path(image['path']).read_bytes()
                    if hashlib.sha256(data).hexdigest() != image['sha256']:
                        raise ValueError('Previously hashed diagnostic image changed')
                    with Image.open(io.BytesIO(data)) as pic:
                        source_images.append(np.asarray(pic.convert('RGB')).copy())
                image_done = time.perf_counter()
                while position < event['native_rows_consumed']:
                    row = native[position]
                    raw = base64.b64decode(row['authority']['prefix_base64'], validate=True)
                    if hashlib.sha256(raw).hexdigest() != row['authority']['prefix_sha256']:
                        raise ValueError('Original native bytes changed')
                    history.ingest_native_physics(decode_physics_prefix(
                        raw[:PHYSICS_PREFIX.size], row['copy']['second_copy_end']))
                    position += 1
                context = history.build(render_start_lowers=tuple(event['render_start_lower']),
                    camera_receipt_time=event['camera_receipt'], decision_time=event['decision'])
                history_done = time.perf_counter()
                retinal_np = np.stack([projector.sample(rgb) for rgb in source_images])
                retinal = torch.as_tensor(retinal_np, dtype=torch.float32, device='cuda')
                projection_done = time.perf_counter()
                if i:
                    for dt, count in held_input_batches(float(clocks[i]-clocks[i-1])):
                        while count:
                            size = min(16, count)
                            stimulus = held[:, None, None, :].expand(-1, size, -1, -1)
                            vis_state = network.simulate(stimulus, dt=dt, initial_state=vis_state,
                                                         as_states=True)[-1]
                            count -= size
                motion = vis_state.nodes.activity[:, motion_indices]
                scene = model.backbone.scene_pool(torch.cat((retinal[:, None], motion), dim=1)[None])
                torch.cuda.synchronize()
                visual_done = time.perf_counter()
                good = context is not None and bool(context.decision_body.valid[required].all())
                prediction = None
                if good:
                    if active and i != fr['start_group']+len(outputs):
                        raise ValueError('Missing body requires an explicit new flight segment')
                    active = True
                    prediction, _ = stream.decision(decision_time=event['decision'], visual=visual,
                        visual_valid=visual_valid, body_context=context, goals=goals,
                        scene_features=scene, scene_valid=scene_valid,
                        airflow_observation=missing_airflow(event['decision']))
                    prediction = prediction.cpu().numpy()[0]
                    outputs.append(prediction.copy())
                elif active and i < fr['end_group_exclusive']:
                    raise ValueError('Unexpected missing required body in active segment')
                torch.cuda.synchronize()
                finish = time.perf_counter()
                durations.append(1000*(finish-t0))
                stages.append([1000*(image_done-t0), 1000*(history_done-image_done),
                    1000*(projection_done-history_done), 1000*(visual_done-projection_done),
                    1000*(finish-visual_done)])
                retinal_errors.append(float(np.max(np.abs(retinal_np-expected_retina[i]))))
                motion_errors.append(float(torch.max(torch.abs(motion-torch.as_tensor(
                    expected_motion[i], device='cuda'))).item()))
                if prediction is not None:
                    intent_errors.append(float(np.max(np.abs(
                        prediction-expected_intent[len(outputs)-1, 0]))))
                held = retinal
        torch.set_default_device(previous_device)
        if (len(outputs) != fr['groups'] or max(retinal_errors) > 1e-6
                or max(motion_errors) > 2e-5 or max(intent_errors) > 2e-6):
            raise ValueError('Integrated outputs diverged from sealed original replays')
        durations, stages = np.asarray(durations), np.asarray(stages)
        arrivals = np.asarray([x['camera_receipt'] for x in events])
        queued, available = [], arrivals[0]
        for arrival, duration in zip(arrivals, durations/1000, strict=True):
            available = max(available, arrival)+duration
            queued.append(1000*(available-arrival))
        np.savez_compressed(run/'timings.npz', total_ms=durations, stage_ms=stages,
            idealized_queue_sojourn_ms=queued, decision_clock=clocks, intent=np.stack(outputs))
        result.update(status='completed_integrated_offline_single_stream_benchmark', groups=len(events),
            flight_warmup_disposable=a.warm_flight,
            flight_groups=len(outputs), vision_warmup_disposable=True, original_vision_resets=0,
            flight_segment_start=fr['start_group'], flight_state_initial_reset=True,
            total_ms_p50_p95_max=np.percentile(durations, [50, 95, 100]).tolist(),
            stage_names=['png_read_decode_hash', 'body_receipts_and_build', 'retinal_projection',
                         'flyvis_and_scene_pool', 'one_flight_stream'],
            stage_ms_p50_p95_max=np.percentile(stages, [50, 95, 100], axis=0).T.tolist(),
            valid_body_groups_ms_p50_p95_max=np.percentile(durations[fr['start_group']:fr['end_group_exclusive']],
                [50, 95, 100]).tolist(),
            idealized_queue_sojourn_ms_p50_p95_max=np.percentile(queued, [50, 95, 100]).tolist(),
            idealized_queue_uses_offline_measured_durations=True,
            idealized_queue_reuses_original_selection_times=True,
            retina_max_abs=max(retinal_errors), motion_max_abs=max(motion_errors),
            intent_max_abs=max(intent_errors), arrays_sha256=sha256(run/'timings.npz'),
            image_hashes_observed_at_capture=False, live_actor_or_physical_sync_verified=False,
            optimizer_updates=0)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
