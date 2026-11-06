"""Load an audited native flight/head candidate for read-only experiments.

This loader grants no actor/control qualification. Small offline score gains and
audits are insufficient evidence for autonomous limit driving.
"""
from pathlib import Path
import torch
from .graph import sha256
from .multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from .native_multisensory_extension import NativeMultisensoryExtensionEncoder
from .retinal_residual_port import RetinalResidualPort
from .sac_unified_student import read


def require_preserved_native_components(candidate,native,parent):
    def equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    for key in ('body_port',):
        if set(candidate[key])!=set(native[key]) or any(not equal(v,candidate[key][k]) for k,v in native[key].items()):
            raise ValueError('Audited native sensory component changed')
    old,new=parent['readout'],candidate['readout']
    if set(old)!=set(new):raise ValueError('Readout state contract changed')
    for key in ('training_reference','training_scale'):
        if not equal(old[key],new[key]):raise ValueError('Frozen muscle normalization changed')
    for key in ('projection.weight','projection.bias'):
        if not equal(old[key][1:],new[key][1:]):raise ValueError('Frozen pedal rows changed')


def load_audited_native_flight_candidate(spec):
    # Import here to keep tensor-only preservation checks independent of the
    # research assembly's external checkpoint dependencies.
    from scripts.sac_steering_experiment import assembly
    native_root=Path(spec['fit']);head_root=Path(spec['head'])
    nr=read(native_root/'result.json');hr=read(head_root/'result.json')
    if nr['status']!='completed_thermal_native_body_port_fit' or hr['status']!='completed_native_flight_steering_refit':
        raise ValueError('Completed native and steering experiments required')
    for root,audit_path,status in [(native_root,spec['fit_audit'],'completed_independent_thermal_native_fit_audit'),
                                   (head_root,spec['head_audit'],'completed_independent_native_flight_steering_refit_audit')]:
        audit_root=Path(audit_path)
        if (read(audit_root/'result.json')['status']!=status
                or read(audit_root/'manifest.json')['config']['fit_result_sha256']!=sha256(root/'result.json')):
            raise ValueError('Audit is absent or belongs to another experiment')
    native_path=native_root/'body-extension.pt';head_path=head_root/'native-steering.pt'
    if (sha256(native_path)!=nr['checkpoint_sha256'] or sha256(native_path)!=spec['fit_sha256']
            or sha256(head_path)!=hr['checkpoint_sha256'] or sha256(head_path)!=spec['head_sha256']):
        raise ValueError('Protected native/head checkpoint changed')
    native=torch.load(native_path,map_location='cpu',weights_only=True)
    candidate=torch.load(head_path,map_location='cpu',weights_only=True)
    if (candidate['schema']!='flybrain.native_flight_steering_refit.v1'
            or candidate['native_checkpoint_sha256']!=nr['checkpoint_sha256']
            or candidate['source_checkpoint_sha256']!=native['source_checkpoint_sha256']
            or candidate['actor_qualified'] is not False):
        raise ValueError('Read-only candidate lineage/schema differs')
    source=Path(native['source_fit_directory'])
    if sha256(source/'steering.pt')!=native['source_checkpoint_sha256']:
        raise ValueError('Original visual/muscle assembly changed')
    parent=torch.load(source/'steering.pt',map_location='cpu',weights_only=True)
    require_preserved_native_components(candidate,native,parent)
    cfg=read(native_root/'manifest.json')['config']
    extension=Path(cfg['arguments']['extension']);contract=read(extension/'contract.json')
    if sha256(extension/'contract.json')!=native['extension_contract_sha256']:
        raise ValueError('Original native sensor contract changed')
    model,head,dn,_,lineage=assembly(source);head.load_state_dict(candidate['readout'],strict=True)
    state=parent['optical'];optical=RetinalResidualPort(state['reference'],state['scale'],len(model.motor.backbone.vision_ports));optical.load_state_dict(state)
    state=native['body_port'];port=MultisensoryBodyFeedbackPort(state['reference'],state['scale'],len(model.motor.backbone.body_ports));port.load_state_dict(state)
    encoder=NativeMultisensoryExtensionEncoder(extension/'contract.json',native_identity=contract['native_identity'])
    for module in (model,head,dn,optical,port):module.eval().requires_grad_(False)
    receipt=dict(native_checkpoint_sha256=nr['checkpoint_sha256'],head_checkpoint_sha256=hr['checkpoint_sha256'],
        source_checkpoint_sha256=native['source_checkpoint_sha256'],
        native_result_sha256=sha256(native_root/'result.json'),head_result_sha256=sha256(head_root/'result.json'),
        extension_contract_sha256=native['extension_contract_sha256'],lineage=lineage,
        actor_qualified=False,vehicle_control_sent=False,development_two_percent_gate_passed=hr['development_two_percent_gate_passed'])
    return (model,head,dn,optical,port,encoder),receipt
