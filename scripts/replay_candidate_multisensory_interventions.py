"""Replay whole candidate vision/body histories through fixed flight circuitry."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.flight_visual_body_temporal_context import (PerViewTemporalContextOpponentFlightStream,
                                                               PerViewVisualBodyContextBoundary)
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.relative_airflow_sensor import missing_airflow
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate', 'raw-audit', 'vision', 'vision-audit', 'body', 'per-view',
                'per-view-audit', 'training', 'opponent', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--role', choices=('train', 'development'), required=True)
    parser.add_argument('--historical-context', help='Pinned existing common-yaw projection; no fit')
    a = parser.parse_args()
    candidate, raw, vision, va, body, pv, pa, train, opponent = map(Path, (
        a.candidate, a.raw_audit, a.vision, a.vision_audit, a.body, a.per_view,
        a.per_view_audit, a.training, a.opponent))
    root = candidate/'recording'
    outer, rr, vr, var, br, pr, par, tr, op = [read(x/'result.json') for x in
        (candidate, raw, vision, va, body, pv, pa, train, opponent)]
    event_path = root/'events.jsonl'
    event_hash = sha256(event_path)
    manifests = [read(x/'manifest.json')['config'] for x in (raw, vision, body, pv, pa)]
    if (outer['status'] != 'completed_preassigned_pace_recording' or outer['role'] != a.role
            or outer['fit_allowed'] is not False or rr['status'] != 'completed'
            or vr['status'] != 'completed' or var['status'] != 'completed'
            or var['full_chronological_comparison']['observations_compared'] != outer['decisions']
            or br['status'] != 'completed' or pr['status'] != 'completed_high_rate_per_view_body'
            or par['status'] != 'completed' or not par['raw_values_masks_receipts_match']
            or any(m['events_sha256'] != event_hash for m in manifests)
            or sha256(vision/'responses.npz') != vr['responses_sha256']
            or sha256(body/'virtual-body.npz') != br['virtual_body_sha256']
            or sha256(pv/'per-view-body.npz') != pr['arrays_sha256']
            or manifests[-1]['arrays_sha256'] != pr['arrays_sha256']
            or sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']
            or not torch.cuda.is_available()):
        raise ValueError('Exact roles, original sensor audits and fixed GPU model lineage required')
    events = [json.loads(line) for line in event_path.read_text(encoding='utf8').splitlines()]
    with np.load(vision/'responses.npz',allow_pickle=False) as f:
        retina, motion, clocks, capture = [f[k].copy() for k in
            ('retina_intensity','motion_activity','decision_monotonic_seconds','capture_ids')]
    with np.load(body/'virtual-body.npz',allow_pickle=False) as f:
        current, valid, times, packets, received = [f[k].copy() for k in
            ('values','valid','decision_monotonic_seconds','native_packet_ids','received_monotonic_seconds')]
    with np.load(pv/'per-view-body.npz',allow_pickle=False) as f:
        views = {k:f[k].copy() for k in f.files}
    n = len(events)
    if (n != outer['decisions'] or current.shape != (n,21) or retina.shape != (n,4,721)
            or motion.shape != (n,4,8,721) or views['values'].shape != (n,4,21)
            or not np.array_equal(clocks,times) or not np.array_equal(clocks,views['decisions'])
            or not np.array_equal(current,views['current_values'])
            or not np.array_equal(valid,views['current_valid'])):
        raise ValueError('Whole original body and visual alignment required')
    context_result = read(Path(a.historical_context)/'result.json') if a.historical_context else None
    if context_result is not None and (context_result['status'] != 'completed_temporal_context_train_onepass'
            or context_result['original_train_only'] is not True
            or sha256(Path(a.historical_context)/'context-boundary.pt') != context_result['checkpoint_sha256']):
        raise ValueError('Exact historical TRAIN-only context checkpoint required')
    run = begin_run(a.output, dict(program_sha256=sha256(__file__), role=a.role,
        events_sha256=event_hash, raw_audit_sha256=sha256(raw/'result.json'),
        vision_sha256=vr['responses_sha256'], vision_audit_sha256=sha256(va/'result.json'),
        body_sha256=br['virtual_body_sha256'], per_view_sha256=pr['arrays_sha256'],
        per_view_audit_sha256=sha256(pa/'result.json'), flight_sha256=sha256(tr['checkpoint']),
        opponent_sha256=op['checkpoint_sha256'], seed=863009,
        historical_context_sha256=context_result['checkpoint_sha256'] if context_result else None,
        fixed_task_speed_kmh=15, scope='fixed model interventions; no fit or control'))
    output = dict(status='failed', fit_allowed=False, optimizer_updates=0, vehicle_control_sent=False)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(863009)
        plan = read(train/'manifest.json')['config']['plan']
        source = read(Path(plan['source'])/'result.json')
        model = EmbodiedVisualFlightCircuit(plan['source_spec'],
            torch.load(source['checkpoint'],map_location='cpu',weights_only=True)).to('cuda')
        model.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'],strict=True)
        model.eval().requires_grad_(False)
        temporal = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
            model.backbone.scene_command.scale,len(model.backbone.vision_ports)).to('cuda')
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model'],strict=True)
        temporal.eval().requires_grad_(False)
        joint = ExperimentalVisualBodyAirflow(model).to('cuda').eval().requires_grad_(False)
        boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports),device='cuda')
        zero_boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports),device='cuda')
        if context_result is not None:
            payload = torch.load(Path(a.historical_context)/'context-boundary.pt',
                                 map_location='cpu',weights_only=True)
            if payload['schema'] != 'flybrain.visual_body_context_projection.v1' or payload['actor_qualified'] is not False:
                raise ValueError('Original unqualified historical common projection required')
            boundary.load_state_dict(payload['model'],strict=True)
        boundaries = (boundary,zero_boundary,boundary,boundary) if context_result else (boundary,)*4
        streams = [PerViewTemporalContextOpponentFlightStream(joint,temporal,b,optimize_held=True)
                   for b in boundaries]
        contract, episode = sha256('configs/ac-virtual-navigation-body-v2.json'), a.role+'-candidate'
        visual = torch.zeros(1,6,device='cuda')
        visual_valid = torch.zeros(1,6,dtype=torch.bool,device='cuda')
        goals = torch.tensor([[0.,.5]],device='cuda')
        scene_valid = torch.ones(1,4,dtype=torch.bool,device='cuda')
        records = []
        with torch.no_grad():
            for i,event in enumerate(events):
                camera = event['camera']; view_map = {v['view_id']:v for v in camera['views']}
                lower = tuple(view_map[v]['render_started_lower'] for v in VIEW_ORDER)
                if (event['index'] != i or capture[i] != camera['capture_id']
                        or event['native']['packet_id'] != packets[i]
                        or not np.array_equal(views['render_start_lower'][i],lower)):
                    raise ValueError('Original event and render identity required')
                body_now = NavigationBodyObservation(episode,int(packets[i]),float(received[i]),
                    float(clocks[i]),current[i].copy(),valid[i].copy(),contract)
                prior = tuple(None if views['packets'][i,j] < 0 else NavigationBodyObservation(
                    episode,int(views['packets'][i,j]),float(views['receipts'][i,j]),
                    float(camera['received']),views['values'][i,j].copy(),views['valid'][i,j].copy(),
                    contract) for j in range(4))
                image = np.concatenate((retina[i,:,None],motion[i]),axis=1)
                scene = model.backbone.scene_pool(torch.as_tensor(image[None],dtype=torch.float32,device='cuda'))
                intents = []
                for k,stream in enumerate(streams):
                    changed = body_now.values.copy()
                    if context_result is None and k == 1: changed[[0,1]] = 0
                    if context_result is None and k == 2: changed[[8,9,10,11]] = 0
                    selected_prior = prior
                    if context_result is not None and k == 2:
                        selected_prior = tuple(None if item is None else replace(item,
                            values=np.concatenate((item.values[:4],np.zeros(1),item.values[5:])))
                            for item in prior)
                    if context_result is not None and k == 3:
                        selected_prior = tuple(None for _ in prior)
                    context = per_view_temporal_body_context(selected_prior,replace(body_now,values=changed),
                        render_start_lowers=lower,camera_receipt_time=float(camera['received']),
                        decision_time=float(clocks[i]))
                    intent,_ = stream.decision(decision_time=float(clocks[i]),visual=visual,
                        visual_valid=visual_valid,body_context=context,goals=goals,
                        scene_features=torch.zeros_like(scene) if context_result is None and k == 3 else scene,
                        scene_valid=scene_valid,airflow_observation=missing_airflow(float(clocks[i])))
                    intents.append(intent.cpu().numpy()[0].tolist())
                records.append(dict(index=i,packet=int(packets[i]),time=float(clocks[i]),intents=intents))
        write_json(run/'events.json',records)
        x = np.asarray([r['intents'] for r in records])
        effects = {}
        names = ((1,'zero_projection'),(2,'yaw_zero'),(3,'history_missing')) if context_result else (
            (1,'tilt_zero'),(2,'load_zero'),(3,'scene_features_zero'))
        for k,name in names:
            delta = np.abs(x[:,k]-x[:,0])
            effects[name] = dict(mean_abs=delta.mean(0).tolist(),max_abs=delta.max(0).tolist())
        output.update(status='completed_fixed_multisensory_interventions',role=a.role,
            decisions=n,events_sha256=sha256(run/'events.json'),effects=effects,
            original_episode_resets=0,fixed_speed_goal_not_original_pace_task=True,
            per_view_body_projection_untrained_zero=context_result is None,
            historical_context_weights_reused_without_retraining=context_result is not None,
            zero_interventions_may_be_out_of_distribution=True,
            physical_timing_verified=False,physiology_or_driving_benefit_verified=False)
    except Exception as error:
        output.update(error=type(error).__name__,reason=str(error))
    finally:
        write_json(run/'result.json',output)
    print(json.dumps(output))
    if output['status'] == 'failed':
        raise RuntimeError(output['reason'])


if __name__ == '__main__':
    main()
