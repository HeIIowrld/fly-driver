"""Hash-bound fine-motion candidate loader for read-only timing experiments only."""
from pathlib import Path
import math
import torch
from .audited_native_visual_candidate import load_readonly_native_candidate
from .motion_spatial_residual_port import MotionSpatialResidualPort
from .graph import sha256
from .sac_unified_student import read


def exact(left,right):
    return set(left)==set(right) and all(v.shape==right[k].shape and v.dtype==right[k].dtype and torch.equal(v,right[k]) for k,v in left.items())


def require_frozen_motion_parent(checkpoint,components):
    for key,index in [('readout',1),('adapter',2),('optical',3),('body_port',4)]:
        if not exact(checkpoint[key],components[index].state_dict()):raise ValueError('Motion candidate changed frozen '+key)


def load_readonly_motion_candidate(plan):
    parent={k:v for k,v in plan.items() if k!='motion'};components,receipt=load_readonly_native_candidate(parent)
    if 'motion' not in plan:return components,None,receipt
    spec=plan['motion'];root=Path(spec['fit']);r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];path=root/'motion-spatial.pt'
    if r['status']!='completed_motion_spatial_port_fit' or r['arm']!='fine_motion' or sha256(path)!=r['checkpoint_sha256'] or sha256(path)!=spec['checkpoint_sha256']:raise ValueError('Completed unchanged fine-motion fit required')
    reference=read(cfg['plan']['reference_plan']);parent_path=Path(reference['candidate_plan'])
    if sha256(parent_path)!=reference['candidate_plan_sha256'] or read(parent_path)!=parent or cfg['candidate']!=receipt:raise ValueError('Motion candidate parent differs')
    audit_root=Path(spec['audit']);audit=read(audit_root/'result.json');am=read(audit_root/'manifest.json')['config']
    if audit['status']!='completed_independent_motion_spatial_port_audit' or am['fit_result_sha256']!=sha256(root/'result.json') or am['fit_checkpoint_sha256']!=sha256(path) or not audit['full_runtime_requested']:raise ValueError('Full independent motion runtime audit required')
    episodes=audit['full_runtime_episodes']
    if len(episodes)!=3 or any(e['episode']!=i or e['events']!=cfg['entries'][i]['model_events'] or e['full_replay_max_difference']>2e-6 for i,e in enumerate(episodes)):raise ValueError('Incomplete motion neural trajectory evidence')
    iv_root=Path(spec['interventions']);iv=read(iv_root/'result.json');ivm=read(iv_root/'manifest.json')['config'];stats_root=Path(spec['intervention_audit']);stats=read(stats_root/'result.json');sm=read(stats_root/'manifest.json')['config']
    if (iv['status']!='completed_motion_spatial_same_body_interventions' or ivm['fit_result_sha256']!=sha256(root/'result.json') or ivm['fit_audit_sha256']!=sha256(audit_root/'result.json')
            or stats['status']!='completed_independent_motion_same_body_statistics_audit' or sm['interventions_result_sha256']!=sha256(iv_root/'result.json') or stats['paired_windows']!=iv['paired_windows']):raise ValueError('Motion intervention provenance differs')
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp['schema']!='flybrain.motion_spatial_port.v1' or cp['actor_qualified'] is not False or cp['arm']!='fine_motion' or cp['parent_candidate']!=receipt:raise ValueError('Explicit unqualified motion schema required')
    require_frozen_motion_parent(cp,components);state=cp['motion_port'];port=MotionSpatialResidualPort(state['reference'],state['scale'],len(components[0].motor.backbone.vision_ports),maximum_logit_delta=cp['maximum_logit_delta'])
    if set(state)!=set(port.state_dict()) or any(v.shape!=port.state_dict()[k].shape or v.dtype!=port.state_dict()[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Motion state dimensions/dtype/value differs')
    if cp['maximum_logit_delta']!=cfg['plan']['maximum_logit_delta'] or float(state['projection.weight'].abs().max())>cfg['plan']['maximum_weight_delta']+1e-7:raise ValueError('Motion change exceeds audited bound')
    port.load_state_dict(state,strict=True);port.eval().requires_grad_(False)
    comparison=read(spec['matched_comparison'])
    if comparison['fine_result_sha256']!=sha256(root/'result.json'):raise ValueError('Matched comparison belongs to another fine candidate')
    receipt=dict(receipt,motion_checkpoint_sha256=sha256(path),motion_fit_result_sha256=sha256(root/'result.json'),motion_runtime_audit_sha256=sha256(audit_root/'result.json'),motion_interventions_sha256=sha256(iv_root/'result.json'),motion_intervention_audit_sha256=sha256(stats_root/'result.json'),motion_direction_windows=stats['hypothesized_direction_windows'],motion_paired_windows=stats['paired_windows'],matched_motion_information_gate_passed=comparison['matched_information_gate_passed'],motion_scope='Unqualified read-only experimental input port; no promotion/actuator authority',actor_qualified=False,vehicle_control_sent=False)
    return components,port,receipt


def require_motion_stream_parity(plan,receipt,geometry_file):
    """Additional live preflight; never infer deployment from training replay."""
    if 'motion' not in plan:return None
    spec=plan['motion'];source=Path(spec['stream_parity']);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];runtime=Path(spec['runtime_parity']);rr=read(runtime/'result.json');rm=read(runtime/'manifest.json')['config']
    required={'src/flybrain_drive/per_view_flyvis_stream.py','src/flybrain_drive/motion_per_view_flyvis_stream.py','src/flybrain_drive/held_visual_prefix.py'}
    if (set(sm['source_sha256'])!=required or sm['program_sha256']!=sha256('scripts/validate_motion_stream_source.py')
            or sha256(source/'executed-entrypoint.py')!=sm['program_sha256'] or rm['program_sha256']!=sha256('scripts/validate_motion_stream_native_runtime.py')
            or sha256(runtime/'executed-entrypoint.py')!=rm['program_sha256']):raise ValueError('Complete unchanged motion parity programs/sources required')
    errors=[sr[k] for k in ('scene_max_difference','motion_max_difference','coarse_max_difference')]+[rr[k] for k in ('maximum_prediction_difference','static_max_difference')]
    if any(not math.isfinite(v) or not 0<=v<=2e-6 for v in errors) or type(rr['events']) is not int or rr['events']<=0 or rm['actor_arrays_sha256']!=sm['actor_arrays_sha256']:raise ValueError('Finite complete same-episode motion parity required')
    if (sr['status']!='completed_motion_stream_source_parity' or not sr['full_episode'] or not sr['held_prefix']
            or sr['features_sha256']!=sha256(source/'stream-features.npz') or sm['geometry_sha256']!=sha256(geometry_file)
            or max(sr['scene_max_difference'],sr['motion_max_difference'],sr['coarse_max_difference'])>2e-6):raise ValueError('Full same-geometry live motion stream parity required')
    for path,expected in sm['source_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Motion live visual source changed '+path)
    if (rr['status']!='completed_motion_stream_native_runtime_parity' or not rr['full_episode'] or rr['events']!=sr['model_events']
            or rr['maximum_prediction_difference']>2e-6 or rr['static_max_difference']>2e-6
            or rm['source_parity_result_sha256']!=sha256(source/'result.json') or rm['source_features_sha256']!=sr['features_sha256']
            or rm['runtime_sha256']!=sha256('src/flybrain_drive/motion_spatial_runtime.py') or rm['candidate']!=receipt):raise ValueError('Full same-candidate motion native runtime parity required')
    return dict(source_result_sha256=sha256(source/'result.json'),runtime_result_sha256=sha256(runtime/'result.json'),stream_sha256=sha256('src/flybrain_drive/motion_per_view_flyvis_stream.py'),events=rr['events'],actor_qualified=False)
