import pytest
import torch
from flybrain_drive.direct_scene_gain_candidate import validate_direct_gain_state

@pytest.mark.parametrize('fault',['extra','reference','dtype','nan','entry_bound','total_bound','frozen_nonzero','arm'])
def test_strict_direct_gain_contract(fault):
 ref=torch.zeros(42);state=dict(reference=ref.clone(),delta=ref.clone());arm='learn_direct'
 if fault=='extra':state['extra']=ref
 if fault=='reference':state['reference'][0]=1
 if fault=='dtype':state['delta']=state['delta'].double()
 if fault=='nan':state['delta'][0]=float('nan')
 if fault=='entry_bound':state['delta'][0]=.3
 if fault=='total_bound':state['delta'].fill_(.1)
 if fault=='frozen_nonzero':state['delta'][0]=.01;arm='frozen_direct'
 if fault=='arm':arm='promoted'
 with pytest.raises(ValueError):validate_direct_gain_state(state,ref,arm)

def test_valid_candidate_and_frozen_control():
 ref=torch.ones(42);state=dict(reference=ref.clone(),delta=torch.zeros(42));validate_direct_gain_state(state,ref,'frozen_direct');state['delta'].fill_(.01);validate_direct_gain_state(state,ref,'learn_direct')
