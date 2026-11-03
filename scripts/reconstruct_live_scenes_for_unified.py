"""Recompute a completed live shadow from original pixels and body receipts."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.cpu_body_process import CpuBodyProcess
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.multirate_live_shadow import SplitCudaMultirateLiveShadow
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, NavigationBodyObservation
from scripts.audit_causal_copy_body_context import first_observed_packets


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def body_from_saved(row, *, episode, decision, contract):
    if row is None:
        return None
    return NavigationBodyObservation(episode, row['packet'], row['receipt'], decision,
        np.asarray(row['values'], np.float64), np.asarray(row['valid'], bool), contract)


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root, audit = base/'20261031-process-body-owned-pause-live', base/'20261031-process-body-owned-pause-live-audit'
    training, opponent = base/'20261029-t1437-embodied-joint-training', base/'20261031-opponent-temporal-graph-trained'
    result, audited, manifest = read(root/'result.json'), read(audit/'result.json'), read(root/'manifest.json')
    for name, key in [('body-ticker.json','ticker_sha256'), ('native-copy-windows.json','native_sha256'),
                      ('bridge-events.json','bridge_sha256')]:
        if sha256(root/name) != result[key]:
            raise ValueError('Exact recorded source hashes required')
    bundle = read(root/'vision/bundle.json')
    archive = root/'vision'/bundle['file']
    if (result['status'] != 'completed_diagnostic' or audited['status'] != 'completed_mmf_immutable_shadow_audit'
            or sha256(archive) != bundle['sha256'] or result['archive']['bundle_sha256'] != bundle['sha256']
            or manifest['config']['process_shadow_source_sha256'] != sha256('src/flybrain_drive/process_body_live_shadow.py')):
        raise ValueError('Completed exact original camera and model evidence required')
    source_paths = ('src/flybrain_drive/cpu_body_process.py',
        'src/flybrain_drive/multirate_live_shadow.py', 'src/flybrain_drive/flight_temporal_motion_boundary.py',
        'src/flybrain_drive/flight_visual_body_temporal_context.py')
    if any(manifest['source_sha256'].get(path) != sha256(path) for path in source_paths):
        raise ValueError('Exact recorded model implementation required')
    rows, ticks = read(root/'bridge-events.json'), read(root/'body-ticker.json')
    packets = first_observed_packets(read(root/'native-copy-windows.json'))
    by_packet = {p['packet']:p for p in packets}
    processed = [x for x in rows if x.get('neural', {}).get('status') in ('visual_published','missing_current_body')]
    published = {x['capture_id']:x for x in processed if x['neural']['status'] == 'visual_published'}
    if len(ticks) != audited['ticks'] or len(published) != audited['published_visual_groups']:
        raise ValueError('All original visual and body observations required')
    output = begin_run(base/'20261101-real-pixels-flyvis-scene-reconstruction',
        {'program_sha256':sha256(__file__), 'recording_sha256':sha256(root/'result.json'),
         'source_audit_sha256':sha256(audit/'result.json'), 'camera_bundle_sha256':bundle['sha256'],
         'seed':863009, 'original_vehicle_source_sha256':manifest['source_sha256']['src/flybrain_drive/vehicle_circuit.py'], 'current_vehicle_source_sha256':sha256('src/flybrain_drive/vehicle_circuit.py'), 'scope':'current vehicle source changed; numerical original output parity independently checked, offline, no actuator'})
    outcome = {'status':'failed','actor_deployed':False,'training_admissible':False}
    worker = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(863009)
        shadow = SplitCudaMultirateLiveShadow(config_path='configs/flight-stream-batched-v1-20261029.json',
                                               training=str(training), opponent=str(opponent))
        scenes, prior_time, held, state, total_ticks = {}, None, None, shadow.vis_state, 0
        with zipfile.ZipFile(archive) as z, torch.no_grad(), torch.device('cuda'):
            for event in processed:
                visual, neural, capture = event['visual'], event['neural'], event['capture_id']
                clock = visual['visual_snapshot']
                if prior_time is not None and not 0 < clock-prior_time <= .15:
                    raise ValueError('Original visual gaps must remain ordered and bounded')
                retina = []
                for i, view in enumerate(VIEW_ORDER):
                    with Image.open(io.BytesIO(z.read(f'{capture:04d}-{view}.bmp'))) as pic:
                        pixels = np.asarray(pic.convert('RGB')).copy()
                    if hashlib.sha256(pixels.tobytes()).hexdigest() != neural['source_images_sha256'][i]:
                        raise ValueError('Original RGB does not match model input')
                    retina.append(shadow.projector.sample(pixels))
                current = torch.as_tensor(np.stack(retina), dtype=torch.float32, device='cuda')
                if prior_time is not None:
                    for dt, count in held_input_batches(clock-prior_time):
                        while count:
                            n = min(count, 16)
                            state = shadow.network.simulate(held[:, None, None, :].expand(-1,n,-1,-1),
                                dt=dt, initial_state=state, as_states=True)[-1]
                            total_ticks += n
                            count -= n
                motion = state.nodes.activity[:, shadow.motion_indices]
                scene = shadow.model.backbone.scene_pool(torch.cat((current[:, None], motion), dim=1)[None])
                if neural['status'] == 'visual_published':
                    scenes[capture] = scene.detach().cpu().numpy().copy()
                held, prior_time = current.detach().clone(), clock
        if set(scenes) != set(published):
            raise ValueError('Every original published scene must be reconstructed')
        capture_ids=np.array(sorted(scenes),dtype=np.int64)
        np.savez_compressed(output/'published-scenes.npz',capture_ids=capture_ids,scene=np.concatenate([scenes[int(i)] for i in capture_ids],axis=0))
        identity = read('configs/ac-virtual-body-v1.json')['native_identity']
        encoder = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json', native_identity=identity)
        episode, contract = result['camera_request_id'], sha256('configs/ac-virtual-navigation-body-v2.json')
        worker = CpuBodyProcess(training, opponent)
        differences, estimates = [], []
        for index, tick in enumerate(ticks):
            source = by_packet[tick['packet']]
            if source['first_end'] != tick['body_receipt']:
                raise ValueError('Original first body receipt changed')
            body = encoder.encode(source['sample'], episode_id=episode, decision_monotonic_seconds=tick['decision'])
            frame = published[tick['visual_frame_index']]
            context, scene = None, None
            if tick['visual_replaced']:
                v = frame['visual']
                current = body_from_saved(v['decision_body'], episode=episode, decision=v['decision'], contract=contract)
                previous = tuple(body_from_saved(x, episode=episode, decision=v['camera_receipt'], contract=contract)
                                 for x in v['view_body'])
                context = per_view_temporal_body_context(previous, current,
                    render_start_lowers=tuple(v['render_start_lower']), camera_receipt_time=v['camera_receipt'],
                    decision_time=v['decision'])
                scene = scenes[tick['visual_frame_index']]
            reply = worker.update(when=tick['decision'], body=body, scene=scene, context=context)
            prediction = reply['intent'][0]
            delta = np.abs(prediction-np.asarray(tick['intent']))
            differences.append(delta.tolist())
            estimates.append({'index':index, 'packet':tick['packet'], 'decision':tick['decision'],
                              'capture':tick['visual_frame_index'], 'intent':prediction.tolist(),
                              'abs_error':delta.tolist()})
        worker_pid, worker_exit = worker.pid, worker.close()
        worker = None
        if worker_exit != 0:
            raise RuntimeError('Replay worker did not exit normally')
        write_json(output/'comparisons.json', estimates)
        errors = np.asarray(differences)
        outcome.update(status='completed' if np.max(errors) <= 2e-6 else 'replay_mismatch',
            processed_visual_groups=len(processed), published_groups=len(scenes), internal_visual_ticks=total_ticks,
            body_decisions=len(ticks), published_scenes_sha256=sha256(output/'published-scenes.npz'), source_vehicle_changed=manifest['source_sha256']['src/flybrain_drive/vehicle_circuit.py']!=sha256('src/flybrain_drive/vehicle_circuit.py'), max_abs_intent_error=errors.max(axis=0).tolist(),
            mean_abs_intent_error=errors.mean(axis=0).tolist(), tolerance=2e-6,
            over_tolerance=int(np.any(errors > 2e-6, axis=1).sum()), comparisons_sha256=sha256(output/'comparisons.json'),
            worker_pid=worker_pid, worker_exit_code=worker_exit, episode_resets=0, optimizer_updates=0,
            vehicle_control_sent=False, physical_source_or_exposure_verified=False)
    except Exception as error:
        outcome.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        if worker is not None:
            try:
                outcome['worker_exit_code'] = worker.close()
            except Exception as error:
                outcome['worker_cleanup_error'] = str(error)
        write_json(output/'result.json', outcome)
        print(json.dumps(outcome), flush=True)


if __name__ == '__main__':
    main()
