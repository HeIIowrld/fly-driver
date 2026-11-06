"""Audited optical-only overlay for read-only native flight experiments."""
from pathlib import Path
import torch
from .audited_native_flight_candidate import load_audited_native_flight_candidate
from .graph import sha256
from .sac_unified_student import read


def require_visual_only_overlay(candidate,head,dn,body,optical,maximum_delta):
    def exact(left,right):
        return set(left)==set(right) and all(v.shape==right[k].shape and v.dtype==right[k].dtype and torch.equal(v,right[k]) for k,v in left.items())
    for key,module in [('readout',head),('adapter',dn),('body_port',body)]:
        if not exact(candidate[key],module.state_dict()):raise ValueError('Visual overlay changed frozen '+key)
    old=optical.state_dict();new=candidate['optical']
    if set(old)!=set(new):raise ValueError('Optical state schema changed')
    for key,value in old.items():
        if value.shape!=new[key].shape or value.dtype!=new[key].dtype or not torch.isfinite(new[key]).all():raise ValueError('Optical state dimensions/dtype/value changed')
        if key!='projection.weight' and not torch.equal(value,new[key]):raise ValueError('Visual normalization changed')
    if float((new['projection.weight']-old['projection.weight']).abs().max())>maximum_delta+1e-7:
        raise ValueError('Visual change exceeds the audited bound')


def load_readonly_native_candidate(plan):
    components,receipt=load_audited_native_flight_candidate(plan['native'])
    if 'visual' not in plan:return components,receipt
    spec=plan['visual'];root=Path(spec['fit']);r=read(root/'result.json');cfg=read(root/'manifest.json')['config']
    if r['status']!='completed_native_visual_anticipation_fit':raise ValueError('Completed visual fit required')
    path=root/'visual-anticipation.pt'
    if sha256(path)!=r['checkpoint_sha256'] or sha256(path)!=spec['checkpoint_sha256']:raise ValueError('Visual checkpoint changed')
    reference=Path(cfg['plan']['reference_plan'])
    if sha256(reference)!=cfg['plan']['reference_plan_sha256'] or read(reference)['native']!=plan['native']:raise ValueError('Visual/base native lineage differs')
    links=[(spec['arithmetic_audit'],'completed_independent_native_visual_anticipation_arithmetic_audit','fit_result_sha256',root/'result.json'),
           (spec['interventions'],'completed_native_visual_anticipation_interventions','fit_result_sha256',root/'result.json'),
           (spec['intervention_audit'],'completed_independent_native_visual_intervention_audit','interventions_result_sha256',Path(spec['interventions'])/'result.json')]
    for directory,status,key,target in links:
        directory=Path(directory)
        if read(directory/'result.json')['status']!=status or read(directory/'manifest.json')['config'][key]!=sha256(target):raise ValueError('Visual audit/replay belongs to another run')
    iv=read(Path(spec['interventions'])/'result.json');audit=read(Path(spec['intervention_audit'])/'result.json')
    if (audit['verified_model_events']!=sum(e['model_events'] for e in cfg['entries'])
            or audit['verified_pre_labels']!=sum(e['candidate']['pre_count'] for e in r['comparisons'])
            or any(e['full_replay_max_error']>2e-6 for e in iv['comparisons'])):raise ValueError('Full visual trajectory verification missing')
    candidate=torch.load(path,map_location='cpu',weights_only=True)
    if (candidate['schema']!='flybrain.native_visual_anticipatory.v1' or candidate['actor_qualified'] is not False
            or candidate['auxiliary_is_actor_output'] is not False or candidate['reference_plan_sha256']!=sha256(reference)
            or candidate['arm']!=cfg['arguments']['arm']):raise ValueError('Read-only visual candidate schema differs')
    model,head,dn,optical,body,encoder=components
    require_visual_only_overlay(candidate,head,dn,body,optical,cfg['plan']['maximum_weight_delta'])
    optical.load_state_dict(candidate['optical'],strict=True);optical.eval().requires_grad_(False)
    receipt=dict(receipt,visual_checkpoint_sha256=sha256(path),visual_fit_result_sha256=sha256(root/'result.json'),
        visual_arithmetic_audit_sha256=sha256(Path(spec['arithmetic_audit'])/'result.json'),visual_interventions_result_sha256=sha256(Path(spec['interventions'])/'result.json'),
        visual_intervention_audit_sha256=sha256(Path(spec['intervention_audit'])/'result.json'),visual_arm=candidate['arm'],
        visual_direction_windows=audit['expected_direction_windows'],visual_paired_windows=audit['paired_windows'],
        development_two_percent_gate_passed=r['development_two_percent_gate_passed'],actor_qualified=False,vehicle_control_sent=False)
    return components,receipt
