import copy
import pytest
from torch import nn
from flybrain_drive.audited_motion_spatial_candidate import require_frozen_motion_parent,require_motion_stream_parity


def fixture():
    components=[None,*[nn.Linear(3,2) for _ in range(4)]]
    checkpoint={key:copy.deepcopy(components[i].state_dict()) for key,i in [('readout',1),('adapter',2),('optical',3),('body_port',4)]}
    return checkpoint,components


def test_motion_keeps_all_existing_output_and_sensory_modules():
    require_frozen_motion_parent(*fixture())


@pytest.mark.parametrize('key',['readout','adapter','optical','body_port'])
def test_existing_module_changes_are_rejected(key):
    cp,components=fixture();cp[key]['weight'][0,0]+=1
    with pytest.raises(ValueError,match='frozen'):require_frozen_motion_parent(cp,components)


def test_legacy_plan_requires_no_new_motion_proof():
    assert require_motion_stream_parity({}, {}, 'missing-geometry') is None


def test_motion_plan_cannot_start_without_full_pixel_and_runtime_proofs(tmp_path):
    with pytest.raises(FileNotFoundError):require_motion_stream_parity({'motion':{'stream_parity':str(tmp_path),'runtime_parity':str(tmp_path)}},{},tmp_path/'geometry')
