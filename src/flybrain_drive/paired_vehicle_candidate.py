"""Offline-only load of a fitted paired muscle decoder; no deployment authority."""
from pathlib import Path
import torch
from .paired_flight_vehicle_readout import PairedFlightVehicleReadout
from .graph import sha256
from .sac_unified_student import read

def load_paired_head(path):
 path=Path(path);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];cp=torch.load(path,map_location='cpu',weights_only=True);plan=m['plan'];state=cp['readout']
 if r['status']!='completed_paired_vehicle_decoder_fit' or cp['schema']!='flybrain.paired_vehicle_readout.v1' or sha256(path)!=r['checkpoint_sha256'] or cp['actor_qualified'] is not False or cp['parent_candidate_plan']!=plan['candidate_plan'] or cp['parent_candidate_plan_sha256']!=plan['candidate_plan_sha256'] or sha256(plan['candidate_plan'])!=plan['candidate_plan_sha256'] or cp['monotone']!=(r['arm']=='monotone') or cp['minimum_power_gain']!=plan['minimum_power_gain'] or cp['maximum_gain']!=plan['maximum_gain']:raise ValueError('Hash-bound unqualified paired head required')
 keys=['difference_scale','context_reference','context_scale','power_reference','power_scale'];head=PairedFlightVehicleReadout(**{k:state[k] for k in keys},monotone=cp['monotone'],minimum_power_gain=cp['minimum_power_gain'],maximum_gain=cp['maximum_gain']);expected=head.state_dict()
 if set(state)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Exact finite paired head tensor contract required')
 head.load_state_dict(state,strict=True);head.eval().requires_grad_(False);head(torch.zeros(1,6))
 return head,dict(paired_decoder_checkpoint_sha256=sha256(path),paired_decoder_monotone=cp['monotone'],parent_candidate_plan_sha256=cp['parent_candidate_plan_sha256'],actor_qualified=False,vehicle_control_sent=False)
