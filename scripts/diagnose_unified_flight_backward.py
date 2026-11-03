"""Exercise original flight graph autograd with synthetic sensory inputs; no fitting."""
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--recorded-train-probe', action='store_true',
                   help='Use first 16 original fit-forbidden TRAIN observations, never fit')
    a = p.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    training = base/'20261029-t1437-embodied-joint-training'
    opponent = base/'20261031-opponent-temporal-graph-trained'
    context = base/'20261031-historical-train00-temporal-context-onepass'
    motor_plan = base/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json'
    tr, op, cr = read(training/'result.json'),read(opponent/'result.json'),read(context/'result.json')
    motor = read(motor_plan)['model']
    if (sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']
            or sha256(context/'context-boundary.pt') != cr['checkpoint_sha256']):
        raise ValueError('Original checkpoint lineage mismatch')
    run = begin_run(a.output, dict(program_sha256=sha256(__file__),
        source_motor_sha256=sha256(motor['checkpoint']), source_visual_sha256=sha256(tr['checkpoint']),
        temporal_sha256=op['checkpoint_sha256'],context_sha256=cr['checkpoint_sha256'],
        synthetic_inputs=not a.recorded_train_probe, seed=863009, optimizer_updates=0))
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False,
        synthetic_inputs=not a.recorded_train_probe, actual_driving_or_physiology_verified=False)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(863009)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required')
        plan = read(training/'manifest.json')['config']['plan']
        source = read(Path(plan['source'])/'result.json')
        visual = EmbodiedVisualFlightCircuit(plan['source_spec'],
            torch.load(source['checkpoint'],map_location='cpu',weights_only=True),
            sparse_backend='cached_csr_cpu')
        visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
        physical = PhysicalManeuverCircuit(motor['spec'],torch.load(motor['checkpoint'],
            map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
        temporal = OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,
            visual.backbone.scene_command.scale,len(visual.backbone.vision_ports))
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',
            map_location='cpu',weights_only=True)['model'])
        boundary = PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports))
        boundary.load_state_dict(torch.load(context/'context-boundary.pt',
            map_location='cpu',weights_only=True)['model'])
        model = UnifiedFlightSensorimotor(physical,visual.backbone,temporal,boundary)
        model.motor.backbone.circuit.sparse_backend = 'cached_csr_gpu'
        model = model.to('cuda').eval().requires_grad_(False)
        model.body_feedback.requires_grad_(True)
        model.temporal.opponent_projection.requires_grad_(True)
        scene = torch.zeros(1,576,device='cuda')
        valid = torch.ones(1,4,dtype=torch.bool,device='cuda')
        goals = torch.tensor(racing_goal(100,1),device='cuda')
        command = torch.tensor([[0.,10.]],device='cuda')
        values = np.zeros(21,dtype=np.float64)
        values[0],values[1],values[16] = .02,-.03,10.
        body_valid = np.ones(21,dtype=bool)
        torch.cuda.reset_peak_memory_stats()
        events = views = retinal = motion = clocks = captures = None
        if a.recorded_train_probe:
            candidate = base/'20261031-pace-train-native-copy-candidate-00'
            visual_run = base/'20261031-pace-train-flyvis-replay'
            body_run = base/'20261031-pace-train-high-rate-per-view-body'
            candidate_result, visual_result, body_result = (read(x/'result.json') for x in
                (candidate,visual_run,body_run))
            if (candidate_result['role'] != 'train' or candidate_result['fit_allowed'] is not False
                    or sha256(visual_run/'responses.npz') != visual_result['responses_sha256']
                    or sha256(body_run/'per-view-body.npz') != body_result['arrays_sha256']):
                raise ValueError('Original fit-forbidden input provenance required')
            event_file = candidate/'recording/events.jsonl'
            event_hash = sha256(event_file)
            for root in (visual_run,body_run):
                if read(root/'manifest.json')['config']['events_sha256'] != event_hash:
                    raise ValueError('Different original event history')
            events = [json.loads(row) for row in event_file.read_text(encoding='utf8').splitlines()]
            with np.load(visual_run/'responses.npz',allow_pickle=False) as f:
                retinal,motion,clocks,captures = [f[k].copy() for k in
                    ('retina_intensity','motion_activity','decision_monotonic_seconds','capture_ids')]
            with np.load(body_run/'per-view-body.npz',allow_pickle=False) as f:
                views = {k:f[k].copy() for k in f.files}
            if not np.array_equal(clocks,views['decisions']):
                raise ValueError('Original sensor clocks differ')
            result.update(events_sha256=event_hash, visual_sha256=visual_result['responses_sha256'],
                body_sha256=body_result['arrays_sha256'],fit_allowed=False)
        count = 16 if a.recorded_train_probe else 8
        for i in range(count):
            if a.recorded_train_probe:
                event = events[i]
                camera,native = event['camera'],event['native']
                now = float(clocks[i])
                if event['index'] != i or camera['capture_id'] != captures[i] or event['decision_monotonic_seconds'] != now:
                    raise ValueError('Original event identity changed')
                contract = sha256('configs/ac-virtual-navigation-body-v2.json')
                body = NavigationBodyObservation('train',int(native['packet_id']),
                    float(native['received_monotonic_seconds']),now,
                    views['current_values'][i].copy(),views['current_valid'][i].copy(),contract)
                prior = tuple(None if views['packets'][i,j] < 0 else NavigationBodyObservation(
                    'train',int(views['packets'][i,j]),float(views['receipts'][i,j]),
                    float(camera['received']),views['values'][i,j].copy(),
                    views['valid'][i,j].copy(),contract) for j in range(4))
                ctx = per_view_temporal_body_context(prior,body,
                    render_start_lowers=tuple(views['render_start_lower'][i]),
                    camera_receipt_time=float(camera['received']),decision_time=now)
                image = np.concatenate((retinal[i,:,None],motion[i]),axis=1)
                scene = model.scene_pool(torch.as_tensor(image[None],dtype=torch.float32,device='cuda'))
            else:
                now = 1.+.04*i
                body = NavigationBodyObservation('synthetic',i,now-.01,now,
                    values.copy(),body_valid.copy(),'synthetic-fixture')
                ctx = per_view_temporal_body_context((None,)*4,body,
                    render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
            intent,muscles = model.training_decision(decision_time=now,command=command,
                racing_goals=goals,scene_features=scene,scene_valid=valid,body_context=ctx)
        loss = intent.square().sum()+muscles.square().sum()
        loss.backward()
        grad = model.body_feedback.projection.weight.grad
        if grad is None or not torch.isfinite(grad).all():
            raise ValueError('Missing or nonfinite original-graph gradient')
        temporal_grad = model.temporal.opponent_projection.weight.grad
        if temporal_grad is None or not torch.isfinite(temporal_grad).all():
            raise ValueError('Missing or nonfinite temporal projection gradient')
        result.update(status='completed', decisions=count, loss=float(loss.detach().cpu()),
            body_gradient_norm=float(grad.norm().cpu()),
            body_gradient_nonzero=int(torch.count_nonzero(grad).cpu()),
            temporal_gradient_norm=float(temporal_grad.norm().cpu()),
            temporal_gradient_nonzero=int(torch.count_nonzero(temporal_grad).cpu()),
            peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
            interpretation='Graph diagnostic with arbitrary output objective; no fitting or performance evidence')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
