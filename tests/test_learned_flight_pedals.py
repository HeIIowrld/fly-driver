import ast,base64,struct
from pathlib import Path
import pytest,torch
from flybrain_drive.learned_flight_pedal_readout import LearnedFlightPedalReadout
from flybrain_drive.muscle_interaction_vehicle_readout import MuscleInteractionVehicleReadout
from flybrain_drive.current_flight_launch_diagnostics import decode_launch_diagnostics,CLUTCH_OFFSET

def parent():
 state={'training_reference':torch.ones(6)*.5,'training_scale':torch.ones(6),'projection.weight':torch.zeros(3,6),'projection.bias':torch.tensor([0.,-1.,-2.])};state['projection.weight'][1,4:]=.01;state['projection.weight'][2,4:]=-.01
 return MuscleInteractionVehicleReadout(state,degree=1,quadratic_reference=torch.empty(0,dtype=torch.float64),quadratic_scale=torch.empty(0,dtype=torch.float64))

def test_initial_head_exact_parent():
 p=parent();h=LearnedFlightPedalReadout(p);x=torch.rand(17,6);assert torch.equal(h(x),p(x))

def test_genuine_pedal_gradient_and_frozen_parent():
 h=LearnedFlightPedalReadout(parent());x=torch.rand(4,6);loss=(h(x)[:,1:]-torch.tensor([.8,.3])).square().mean();loss.backward();assert h.pedal_projection.weight.grad.abs().sum()>0;assert all(p.grad is None for p in h.parent.parameters());assert torch.count_nonzero(h.pedal_projection.weight.grad[0])==0

def test_learning_does_not_remove_brake_or_enforce_exclusion():
 h=LearnedFlightPedalReadout(parent());x=torch.ones(1,6)*.5
 with torch.no_grad():h.pedal_projection.bias[1:]=0
 assert torch.equal(h(x)[:,1:],torch.tensor([[.5,.5]]))

def test_project_preserves_original_steer_and_power_signs():
 h=LearnedFlightPedalReadout(parent());x=torch.rand(7,6);old=h(x)[:,0].clone()
 with torch.no_grad():h.pedal_projection.weight.fill_(20);h.pedal_projection.bias.fill_(20)
 h.project();h.verify();assert torch.equal(h(x)[:,0],old);assert (h.pedal_projection.weight[1,4:]>0).all();assert (h.pedal_projection.weight[2,4:]<0).all()

def test_native_clutch_offset_against_pinned_abi_without_importing_mapping():
 tree=ast.parse(Path('assets/external/csp-shared-physics-abi-20261102/sim_info.py').read_text(encoding='utf8'));cls=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='SPageFilePhysics');fields=next(v.value for v in cls.body if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_fields_' for t in v.targets));offset=0;found={}
 def width(v):
  if isinstance(v,ast.Name):assert v.id in ['c_float','c_int32'];return 4
  assert isinstance(v,ast.BinOp) and isinstance(v.op,ast.Mult) and isinstance(v.right,ast.Constant);return width(v.left)*v.right.value
 for row in fields.elts:
  name=row.elts[0].value;found[name]=offset;offset+=width(row.elts[1])
 assert found['clutch']==CLUTCH_OFFSET==364;assert found['isAIControlled']==416

@pytest.mark.parametrize('clutch',[0.,.1,1.])
def test_original_same_copy_diagnostic(clutch):
 raw=bytearray(420);struct.pack_into('<i',raw,0,123);struct.pack_into('<i',raw,16,2);struct.pack_into('<i',raw,20,2300);struct.pack_into('<f',raw,364,clutch);v=decode_launch_diagnostics(bytes(raw),1.);assert v['gear']==2 and v['rpms']==2300 and v['clutch']==pytest.approx(clutch);assert v['actor_input'] is False

@pytest.mark.parametrize('clutch',[float('nan'),-.1,1.1])
def test_invalid_native_clutch_rejected(clutch):
 raw=bytearray(420);struct.pack_into('<f',raw,364,clutch)
 with pytest.raises(ValueError):decode_launch_diagnostics(bytes(raw),1.)
