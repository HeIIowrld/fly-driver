"""Load the exact current research actor for read-only CPU inference."""
from pathlib import Path
import torch
from .graph import sha256
from .sac_unified_student import read
from .audited_motion_spatial_candidate import load_readonly_motion_candidate
from .joint_dn_decoder_candidate import load_joint_dn_decoder
from .fixture_adjoint_speed_candidate import apply_fixture_adjoint_speed,PowerBoundedSpeedDecisionRuntime
from .native_flight_decision_runtime import prepare_fixed_cpu_backend

def load_current_flight_runtime(checkpoint,expected_sha256):
    path=Path(checkpoint)
    if sha256(path)!=expected_sha256:raise ValueError('Exact current checkpoint required')
    metadata=read(path.parent/'manifest.json')['config'];plan=metadata['plan']
    if any(sha256(f)!=h for f,h in plan['source_sha256'].items()):raise ValueError('Executed learning sources changed')
    candidate=read(read(read(plan['base_port_plan'])['parent_fit_plan'])['candidate_plan'])
    components,motion,parent=load_readonly_motion_candidate(candidate)
    components,calibration,receipt=load_joint_dn_decoder(plan['parent_checkpoint'],components,parent)
    changed,current,loaded=apply_fixture_adjoint_speed(path,components,calibration,motion)
    runtime=PowerBoundedSpeedDecisionRuntime(*changed,motion_port=motion,current_calibration=current,policy_limits=plan)
    warm=prepare_fixed_cpu_backend(runtime.model)
    return runtime,dict(checkpoint_sha256=sha256(path),arm=metadata['arm'],runtime_kind=type(runtime).__name__,speed_boundary_kind=type(runtime.model.speed_command_boundary).__name__,current_calibration_kind=type(current).__name__,motion_features_required=True,torch_version=torch.__version__,torch_file=torch.__file__,warmup=warm,loader_receipt=loaded,actor_qualified=False,vehicle_control_sent=False)
