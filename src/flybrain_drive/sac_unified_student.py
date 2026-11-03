"""Offline teacher-distillation assembly of the existing flight graph.

The three-axis vehicle readout and sensory port adapters are learned engineering
boundaries. They do not establish physiological tuning or vehicle deployment.
No map, heading, world pose, teacher observation or timestamp enters the readout.
"""
import json
from pathlib import Path
import torch
from torch import nn
from .graph import sha256
from .flight_embodied_navigation import EmbodiedVisualFlightCircuit
from .flight_maneuver import PhysicalManeuverCircuit
from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from .flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from .flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from .unified_flight_sensorimotor import UnifiedFlightSensorimotor
from .temporal_body_context import HostFreshnessLimits

BASE=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def assemble_student():
    trdir=BASE/'20261029-t1437-embodied-joint-training'
    opdir=BASE/'20261031-opponent-temporal-graph-trained'
    ctxdir=BASE/'20261031-historical-train00-temporal-context-onepass'
    tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
    if (sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or
        sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or
        sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']):
        raise ValueError('Existing measured flight graph/checkpoint lineage differs')
    spec=read(trdir/'manifest.json')['config']['plan']
    source=read(Path(spec['source'])/'result.json')
    load=lambda path:torch.load(path,map_location='cpu',weights_only=True)
    visual=EmbodiedVisualFlightCircuit(spec['source_spec'],load(source['checkpoint']),sparse_backend='cached_csr_cpu')
    visual.load_state_dict(load(tr['checkpoint'])['model'])
    mp=read(BASE/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model']
    motor=PhysicalManeuverCircuit(mp['spec'],load(mp['checkpoint']),sparse_backend='cached_csr_cpu')
    temporal=OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,
        visual.backbone.scene_command.scale,len(visual.backbone.vision_ports))
    temporal.load_state_dict(load(opdir/'temporal-boundary.pt')['model'])
    boundary=PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports))
    boundary.load_state_dict(load(ctxdir/'context-boundary.pt')['model'])
    body=CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports))
    # Explicit fresh engineering normalization for unrestricted race speeds.
    # This is not a reinterpretation of the historical 30m/s synthetic adapter.
    with torch.no_grad():body.scales[10]=80.;body.scales[11:]=300.
    model=UnifiedFlightSensorimotor(motor,visual.backbone,temporal,boundary,
        body_feedback=body,freshness_limits=HostFreshnessLimits(.05,.15))
    model.requires_grad_(False)
    for boundary in (model.visual,model.temporal,model.view_context,model.body_feedback):
        boundary.requires_grad_(True)
    lineage=dict(visual_sha256=sha256(tr['checkpoint']),motor_sha256=sha256(mp['checkpoint']),
        motor_spec_sha256=sha256(mp['spec']),temporal_sha256=sha256(opdir/'temporal-boundary.pt'),
        context_sha256=sha256(ctxdir/'context-boundary.pt'),
        biological_graph='Existing flight-derived measured chemical topology; fixed during distillation',
        learned_hypotheses='Sensory port adapters and three-axis muscle-to-vehicle readout',
        body_speed_scale_m_s=80.,wheel_rotation_scale_rad_s=300.)
    return model,lineage

class MuscleVehicleReadout(nn.Module):
    """Only existing flight muscle activities drive steer/gas/brake predictions."""
    def __init__(self,muscles):
        super().__init__()
        self.register_buffer('training_reference',torch.zeros(muscles))
        self.register_buffer('training_scale',torch.ones(muscles))
        self.projection=nn.Linear(muscles,3)
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
    def forward(self,muscles):
        value=self.projection((muscles-self.training_reference)/self.training_scale)
        return torch.cat([value[:,:1].tanh(),value[:,1:].sigmoid()],dim=1)

def detach_recurrence(model):
    # Preserve numerical neural/traces/drive values across TBPTT boundaries.
    for key in ('state','previous_drive','previous_scene','previous_raw_scene'):
        value=getattr(model,key)
        if value is not None:setattr(model,key,value.detach())
    if model.traces is not None:model.traces=tuple(x.detach() for x in model.traces)
