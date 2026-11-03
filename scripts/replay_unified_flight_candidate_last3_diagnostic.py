"""Replay retained four-view/body candidate through one fixed physical flight graph."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.flight_geometry_feedback import CausalGeometryVisualPort
from flybrain_drive.visual_self_motion import StationaryMotionReference

class LastThreeDiagnosticPort(CausalGeometryVisualPort):
    """Offline paired checkpoint contract; full original reference admission remains."""
    def reset_episode(self):
        super().reset_episode()
        self.reference = StationaryMotionReference(mean_frames=3)

from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from flybrain_drive.visual_self_motion import VisualSelfMotionPool
from flybrain_drive.temporal_body_context import per_view_temporal_body_context


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class FeedbackValuesIntervention(nn.Module):
    """Diagnostic-only value ablation at adapter input; preserve body validity and other routes."""
    GROUPS = {'tilt_zero': (0,1), 'load_zero': (8,9,10,11),
              'temperature_zero': (12,13,14,15), 'wheel_zero': (17,18,19,20)}

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.condition = 'intact'

    def forward(self, body, valid):
        if self.condition == 'intact':
            return self.adapter(body,valid)
        if self.condition not in self.GROUPS:
            raise ValueError('Unknown diagnostic body intervention')
        modified = body.clone()
        modified[:,self.GROUPS[self.condition]] = 0
        return self.adapter(modified,valid)


def reverse_view_motion(scene, view):
    """Swap opponent pairs within one view; keep retinal and other view features."""
    if scene.shape != (1,576) or view not in (0,1,2,3):
        raise ValueError('One original four-view pooled scene and a valid view required')
    changed = scene.reshape(1,4,9,16).clone()
    changed[:,view,1:9] = changed[:,view,[2,1,4,3,6,5,8,7]].clone()
    return changed.reshape(1,576)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','raw-audit','visual','visual-audit','body-context',
                 'body-audit','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    p.add_argument('--preview-commands',help='Original role-pinned fixed visual/body preview commands')
    p.add_argument('--body-feedback',help='Experimental historical anchored body-adapter run; offline transfer comparison')
    p.add_argument('--body-interventions',action='store_true',help='Compare individual adapter-input value ablations')
    p.add_argument('--motion-direction-interventions',action='store_true',help='Swap opponent motion channels in one view at a time')
    p.add_argument('--causal-geometry',help='Pinned experimental geometry checkpoint directory, offline only')
    p.add_argument('--geometry-pulse-index',type=int,help='Offline single-decision synthetic geometry pulse')
    p.add_argument('--zero-command-diagnostic',action='store_true',help='Offline fixed yaw=0 and speed=0 instead of preview command')
    p.add_argument('--zero-command-sensory-interventions',action='store_true', help='Offline zero-task scene, geometry and learned body-port ablations')
    a = p.parse_args()
    if a.body_feedback and not a.preview_commands:
        raise ValueError('Body transfer comparison requires identical fixed preview commands')
    if a.body_interventions and not a.body_feedback:
        raise ValueError('Body interventions require the historical body adapter')
    if a.motion_direction_interventions and (not a.body_feedback or a.body_interventions):
        raise ValueError('Motion interventions require body fit and cannot mix intervention modes')
    if a.causal_geometry and (not a.body_feedback or a.body_interventions or a.motion_direction_interventions):
        raise ValueError('Causal geometry requires body fit and an exclusive intervention mode')
    if a.geometry_pulse_index is not None and (not a.causal_geometry or a.geometry_pulse_index < 0):
        raise ValueError('Nonnegative pulse index requires causal geometry experiment')
    if a.zero_command_diagnostic and (not a.causal_geometry or a.geometry_pulse_index is not None):
        raise ValueError('Quiet diagnostic requires causal geometry without a synthetic pulse')
    if a.zero_command_sensory_interventions and not a.zero_command_diagnostic:
        raise ValueError('Sensory ablations require the zero-command diagnostic')
    candidate,raw,visual,va,body,ba = map(Path,(a.candidate,a.raw_audit,a.visual,
                                                  a.visual_audit,a.body_context,a.body_audit))
    root = candidate/'recording'
    outer, rr, vr, ar, br, bar = [read(x/'result.json') for x in (candidate,raw,visual,va,body,ba)]
    event_path = root/'events.jsonl'
    event_hash = sha256(event_path)
    vm, bm, am = [read(x/'manifest.json')['config'] for x in (visual,body,ba)]
    if (outer['role'] != a.role or outer['status'] != 'completed_preassigned_pace_recording'
            or outer['fit_allowed'] is not False or rr['status'] != 'completed'
            or read(raw/'manifest.json')['config']['events_sha256'] != event_hash
            or vr['status'] != 'completed' or ar['status'] != 'completed'
            or ar['full_chronological_comparison']['observations_compared'] != outer['decisions']
            or br['status'] != 'completed_high_rate_per_view_body' or bar['status'] != 'completed'
            or not bar['raw_values_masks_receipts_match']
            or any(m['events_sha256'] != event_hash for m in (vm,bm,am))
            or sha256(visual/'responses.npz') != vr['responses_sha256']
            or sha256(body/'per-view-body.npz') != br['arrays_sha256']
            or am['arrays_sha256'] != br['arrays_sha256'] or not torch.cuda.is_available()):
        raise ValueError('Exact audited whole candidate and GPU required')
    events = [json.loads(x) for x in event_path.read_text(encoding='utf8').splitlines()]
    with np.load(visual/'responses.npz',allow_pickle=False) as f:
        retinal,motion,clocks,captures = [f[k].copy() for k in
            ('retina_intensity','motion_activity','decision_monotonic_seconds','capture_ids')]
    with np.load(body/'per-view-body.npz',allow_pickle=False) as f:
        views = {k:f[k].copy() for k in f.files}
    n = len(events)
    if (n != outer['decisions'] or retinal.shape != (n,4,721) or motion.shape != (n,4,8,721)
            or not np.array_equal(clocks,views['decisions'])):
        raise ValueError('Whole original visual and body history required')
    preview = preview_result = None
    if a.preview_commands:
        preview_root = Path(a.preview_commands)
        preview_result = read(preview_root/'result.json')
        preview_config = read(preview_root/'manifest.json')['config']
        if (preview_result['status'] != 'completed_visual_preview_commands_diagnostic'
                or preview_result['role'] != a.role or preview_result['fit_allowed'] is not False
                or preview_config['events_sha256'] != event_hash
                or preview_config['body_sha256'] != br['arrays_sha256']
                or sha256(preview_root/'commands.npz') != preview_result['commands_sha256']):
            raise ValueError('Original fixed visual preview command provenance required')
        with np.load(preview_root/'commands.npz',allow_pickle=False) as f:
            preview=f['commands'].copy()
            if not np.array_equal(f['decision_clock'],clocks) or preview.shape != (2,n,2):
                raise ValueError('Original preview commands and clocks required')
    body_fit = None
    if a.body_feedback:
        fit_root = Path(a.body_feedback)
        fit_result = read(fit_root/'result.json')
        checkpoint = fit_root/'body-feedback-adapter.pt'
        if (fit_result['status'] != 'completed_brake_steering_anchored_body_fit'
                or fit_result['checkpoint_sha256'] != sha256(checkpoint)
                or sha256(checkpoint) != '4030ac7e53c9175a517adb3d588d9c5be4746f818ddc09762c6ee524771e2315'):
            raise ValueError('Exact historical experimental anchored fit required')
        body_fit = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if body_fit['schema'] != 'flybrain.experimental_body_feedback_adapter.v1' or body_fit['actor_qualified'] is not False:
            raise ValueError('Unqualified historical engineering adapter required')
    geometry_fit = None
    if a.causal_geometry:
        geometry_path=Path(a.causal_geometry)/'geometry-visual-boundary.pt'
        if sha256(geometry_path) != '093345b9d78b516a3c55804714e1a446fef057e7d4271d5e331aa5d67aae896c':
            raise ValueError('Exact experimental geometry checkpoint required')
        geometry_fit=torch.load(geometry_path,map_location='cpu',weights_only=True)
        if geometry_fit['actor_qualified'] is not False or geometry_fit.get('reference_window') != 'last3_of_first11':
            raise ValueError('Unqualified offline geometry candidate required')
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
        raise ValueError('Exact prior fixed model/checkpoint lineage required')
    run = begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,
        events_sha256=event_hash,visual_sha256=vr['responses_sha256'],body_sha256=br['arrays_sha256'],
        visual_audit_sha256=sha256(va/'result.json'),body_audit_sha256=sha256(ba/'result.json'),
        source_motor_sha256=sha256(motor['checkpoint']),source_visual_sha256=sha256(tr['checkpoint']),
        temporal_sha256=op['checkpoint_sha256'],context_sha256=cr['checkpoint_sha256'],
        preview_commands_sha256=preview_result['commands_sha256'] if preview_result else None,
        historical_body_fit_sha256=fit_result['checkpoint_sha256'] if body_fit else None,
        body_value_interventions=a.body_interventions,
        motion_direction_interventions=a.motion_direction_interventions,
        geometry_candidate_sha256=sha256(geometry_path) if geometry_fit is not None else None,
        geometry_reference_contract='last3_of_first11',
        geometry_pulse_index=a.geometry_pulse_index,
        geometry_pulse_amplitude=.005 if a.geometry_pulse_index is not None else None,
        fixed_zero_command_diagnostic=a.zero_command_diagnostic,
        zero_command_sensory_interventions=a.zero_command_sensory_interventions,
        seed=863009,scope='one physical flight state, offline only, no actor qualification'))
    result = dict(status='failed',role=a.role,fit_allowed=False,optimizer_updates=0,vehicle_control_sent=False)
    try:
        torch.set_num_threads(1);torch.manual_seed(863009)
        plan = read(training/'manifest.json')['config']['plan']
        source = read(Path(plan['source'])/'result.json')
        visual_model = EmbodiedVisualFlightCircuit(plan['source_spec'],
            torch.load(source['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
        visual_model.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
        physical = PhysicalManeuverCircuit(motor['spec'],torch.load(motor['checkpoint'],
            map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
        temporal = OpponentTemporalMotionBoundary(visual_model.backbone.scene_command.mean,
            visual_model.backbone.scene_command.scale,len(visual_model.backbone.vision_ports))
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model'])
        context = PerViewVisualBodyContextBoundary(len(visual_model.backbone.vision_ports))
        context.load_state_dict(torch.load(base/'20261031-historical-train00-temporal-context-onepass/context-boundary.pt',
            map_location='cpu',weights_only=True)['model'])
        model = UnifiedFlightSensorimotor(physical,visual_model.backbone,temporal,context)
        if body_fit is not None:
            for name in ('indices','scales'):
                if not torch.equal(getattr(model.body_feedback,name),body_fit['model'][name]):
                    raise ValueError('Historical body channel ordering or scales changed')
            if model.body_feedback.projection.weight.shape != body_fit['model']['projection.weight'].shape:
                raise ValueError('Historical sensory port shape changed')
        model.motor.backbone.circuit.sparse_backend='cached_csr_gpu'
        model = model.to('cuda').eval().requires_grad_(False)
        if a.body_interventions or a.motion_direction_interventions or geometry_fit is not None:
            with torch.no_grad():
                model.body_feedback.projection.weight.copy_(body_fit['model']['projection.weight'].to('cuda'))
        if a.body_interventions:
            model.body_feedback = FeedbackValuesIntervention(model.body_feedback)
        if geometry_fit is not None:
            registration=visual/'retina-registration.npz'
            camera_config=read('configs/ac-csp-exterior-front-v3-retina.json')
            with np.load(registration,allow_pickle=False) as f:
                geometry_pool=VisualSelfMotionPool.from_registration(camera_config,{k:f[k].copy() for k in f.files})
            model.view_context=LastThreeDiagnosticPort(model.view_context,geometry_pool,
                len(model.motor.backbone.vision_ports)).to('cuda')
        goals = torch.tensor(racing_goal(100,1),device='cuda')
        scene_valid = torch.ones(1,4,dtype=torch.bool,device='cuda')
        contract = sha256('configs/ac-virtual-navigation-body-v2.json')
        records=[]
        conditions = (('intact','scene_zero','geometry_zero','body_adapter_zero') if a.zero_command_sensory_interventions else
            ('geometry_trained','geometry_zero') if geometry_fit is not None else
            ('intact','front_reverse','left_reverse','right_reverse','rear_reverse')
            if a.motion_direction_interventions else
            ('intact',*FeedbackValuesIntervention.GROUPS) if a.body_interventions else
            ('historical_body_adapter','neutral_body_adapter') if body_fit is not None else
            ('intact','preview_motion_mean') if preview is not None else ('intact','scene_features_zero'))
        geometry_ready=[]
        for condition_index,condition in enumerate(conditions):
            model.reset_episode()
            if geometry_fit is not None:
                with torch.no_grad():
                    weights=geometry_fit['model']['weight'].to('cuda')
                    model.view_context.projection.weight.copy_(weights if (condition != 'geometry_zero' if a.zero_command_sensory_interventions else condition_index == 0) else torch.zeros_like(weights))
                    if a.zero_command_sensory_interventions:
                        body_weights=body_fit['model']['projection.weight'].to('cuda')
                        model.body_feedback.projection.weight.copy_(torch.zeros_like(body_weights) if condition == 'body_adapter_zero' else body_weights)
            if a.body_interventions:
                model.body_feedback.condition = condition
            elif body_fit is not None and not a.motion_direction_interventions and geometry_fit is None:
                with torch.no_grad():
                    model.body_feedback.projection.weight.copy_(body_fit['model']['projection.weight'].to('cuda')
                        if condition_index == 0 else torch.zeros_like(model.body_feedback.projection.weight))
            intents=[]
            with torch.no_grad():
                for i,event in enumerate(events):
                    camera, native = event['camera'],event['native']
                    if event['index'] != i or camera['capture_id'] != captures[i] or event['decision_monotonic_seconds'] != clocks[i]:
                        raise ValueError('Original event identity changed')
                    current = NavigationBodyObservation(a.role,int(native['packet_id']),
                        float(native['received_monotonic_seconds']),float(clocks[i]),
                        views['current_values'][i].copy(),views['current_valid'][i].copy(),contract)
                    prior = tuple(None if views['packets'][i,j] < 0 else NavigationBodyObservation(
                        a.role,int(views['packets'][i,j]),float(views['receipts'][i,j]),
                        float(camera['received']),views['values'][i,j].copy(),views['valid'][i,j].copy(),contract)
                        for j in range(4))
                    ctx = per_view_temporal_body_context(prior,current,
                        render_start_lowers=tuple(views['render_start_lower'][i]),
                        camera_receipt_time=float(camera['received']),decision_time=float(clocks[i]))
                    image = np.concatenate((retinal[i,:,None],motion[i]),axis=1)
                    scene = model.scene_pool(torch.as_tensor(image[None],dtype=torch.float32,device='cuda'))
                    if geometry_fit is not None:
                        raw_native=base64.b64decode(native['prefix_base64'],validate=True)
                        sample=decode_physics_prefix(raw_native,float(native['received_monotonic_seconds']))
                        ready=model.view_context.observe(motion[i],np.ones(4,dtype=bool),sample,
                            capture_id=int(captures[i]),decision_time=float(clocks[i]), camera_receipt_time=float(camera['received']))
                        if a.geometry_pulse_index is not None and ready:
                            model.view_context.current=torch.zeros_like(model.view_context.current)
                            if i == a.geometry_pulse_index:
                                model.view_context.current[0,2]=.005
                        if condition_index == 0:
                            geometry_ready.append(ready)
                    if condition in ('scene_features_zero','scene_zero'): scene = torch.zeros_like(scene)
                    if a.motion_direction_interventions and condition_index:
                        scene = reverse_view_motion(scene,condition_index-1)
                    task = event['external_task_at_receipt']
                    if task.get('held_external_instruction') is not True or task.get('actor_input') is not True:
                        raise ValueError('Original explicit held task required')
                    command = (torch.tensor([[0.,0.]],dtype=torch.float32,device='cuda') if a.zero_command_diagnostic else
                        torch.as_tensor(preview[0 if body_fit is not None else condition_index,i:i+1],dtype=torch.float32,device='cuda')
                        if preview is not None else torch.tensor([[0.,task['desired_speed_m_s']]],dtype=torch.float32,device='cuda'))
                    intent,_ = model.decision(decision_time=float(clocks[i]),command=command,
                        racing_goals=goals,scene_features=scene,scene_valid=scene_valid,body_context=ctx)
                    intents.append(intent.cpu().numpy()[0].tolist())
            records.append(intents)
        values = np.asarray(records)
        if values.shape != (len(conditions),n,2) or not np.isfinite(values).all():
            raise ValueError('Full finite original intent series required')
        np.savez_compressed(run/'intents.npz',intents=values,decision_clock=clocks)
        difference=np.abs(values[0]-values[1])
        speeds=[e['external_task_at_receipt']['desired_speed_m_s']*3.6 for e in events]
        result.update(status='completed_unified_flight_offline_replay',decisions=n,
            distinct_task_speeds_kmh=sorted(set(round(x,3) for x in speeds)),
            intents_sha256=sha256(run/'intents.npz'),same_graph_and_ports_verified=True,
            state_resets_inside_episode=0,external_task_yaw_fixed_zero=preview is None or a.zero_command_diagnostic,
            visual_preview_command_used=preview is not None and not a.zero_command_diagnostic,
            preview_command_is_unqualified_engineering_hypothesis=preview is not None and not a.zero_command_diagnostic,
            fixed_zero_command_diagnostic=a.zero_command_diagnostic,
            motion_intervention_scope=('zero_task_scene_geometry_body_adapter' if a.zero_command_sensory_interventions else
                'causal_geometry_visual_port' if geometry_fit is not None else
                'pooled_scene_one_view_opponent_swap' if a.motion_direction_interventions else
                'none' if body_fit is not None else
                'preview_only' if preview is not None else 'full_scene_features'),
            historical_body_adapter_transfer=body_fit is not None,
            historical_body_training_had_different_pedal_command_parameters=body_fit is not None,
            full_visual_actor_decision_not_implemented=True,body_adapter_is_zero_initialized=body_fit is None,
            zero_scene_may_be_out_of_distribution=True,physical_timing_verified=False,
            physiological_or_closed_loop_driving_verified=False)
        prefix=('zero_task_scene_ablation' if a.zero_command_sensory_interventions else
                'causal_geometry_trained_vs_zero' if geometry_fit is not None else
                'front_motion_reversal' if a.motion_direction_interventions else
                'tilt_values_zero' if a.body_interventions else
                'historical_body_adapter_vs_neutral' if body_fit is not None else
                'preview_motion_mean' if preview is not None else 'visual_zero')
        result[prefix+'_mean_abs_intent_delta']=difference.mean(0).tolist()
        result[prefix+'_max_abs_intent_delta']=difference.max(0).tolist()
        if a.zero_command_sensory_interventions:
            result['zero_task_sensory_interventions'] = {name: dict(
                mean_abs_intent_delta=np.abs(values[0]-values[index]).mean(0).tolist(),
                max_abs_intent_delta=np.abs(values[0]-values[index]).max(0).tolist())
                for index,name in enumerate(conditions) if index}
            result['intervention_scope'] = 'fixed zero task; scene features, geometry correction, or learned body adapter separately zeroed'
            result['ablations_may_be_out_of_distribution'] = True
        if a.body_interventions:
            result['body_value_interventions'] = {name: dict(
                mean_abs_intent_delta=np.abs(values[0]-values[index]).mean(0).tolist(),
                max_abs_intent_delta=np.abs(values[0]-values[index]).max(0).tolist())
                for index,name in enumerate(conditions) if index}
            result['intervention_scope'] = 'only selected current body values at learned adapter; original mask and other routes preserved'
            result['zero_values_may_be_out_of_distribution'] = True
        if a.motion_direction_interventions:
            result['motion_direction_interventions'] = {name: dict(
                mean_abs_intent_delta=np.abs(values[0]-values[index]).mean(0).tolist(),
                max_abs_intent_delta=np.abs(values[0]-values[index]).max(0).tolist())
                for index,name in enumerate(conditions) if index}
            result['intervention_scope'] = 'one view opponent-channel swap at pooled scene; fixed preview command, other views and body unchanged'
            result['swapped_activities_may_be_out_of_distribution'] = True
        if geometry_fit is not None:
            mask=np.asarray(geometry_ready,dtype=bool)
            if mask.shape != (n,): raise ValueError('Complete original reference readiness mask required')
            if a.geometry_pulse_index is not None and (a.geometry_pulse_index >= n or not mask[a.geometry_pulse_index]):
                raise ValueError('Synthetic pulse must target an originally ready decision')
            np.savez_compressed(run/'geometry-readiness.npz',ready=mask,decision_clock=clocks)
            result.update(geometry_ready_count=int(mask.sum()),
                geometry_first_ready_index=int(np.flatnonzero(mask)[0]) if mask.any() else None,
                geometry_readiness_sha256=sha256(run/'geometry-readiness.npz'),
                geometry_is_unqualified_engineering_candidate=True,
                geometry_body_source_is_original_native_receipt=True)
            if a.geometry_pulse_index is not None:
                result.update(synthetic_geometry_pulse_index=a.geometry_pulse_index,
                    synthetic_geometry_pulse_amplitude=.005,
                    synthetic_pulse_scope='geometry feature only; all other original inputs unchanged')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed': raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
