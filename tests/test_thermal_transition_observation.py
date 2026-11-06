from types import SimpleNamespace
import pytest
from flybrain_drive.thermal_transition_observation import validate_transition_context,validate_transition_native
from flybrain_drive.initial_thermal_contract import validate_thermal_native


def state():
    return dict(generation='g',epoch=2,car_id='bmw_z4_gt3',track_id='monza',tyres_initialization_epoch=2,
        is_online=False,in_menu=False,ai_controlled=False,camera_active=False,expert_probe_active=False,
        brake_settle_active=False,physics_api_allowed=True,speed_kmh=.169,gas=0.,brake=1.)


def test_observation_envelope_does_not_change_final_stationary_contract():
    s=state();validate_transition_context(s,dict(generation='g',epoch=2),(None,))
    sample=SimpleNamespace(builtin_ai_controlled=False,physics=SimpleNamespace(fields=dict(speedKmh=.169,gas=0.,brake=1.)))
    validate_transition_native(sample)
    with pytest.raises(ValueError):validate_thermal_native(sample)


@pytest.mark.parametrize('key,value',[('epoch',3),('generation','other'),('tyres_initialization_epoch',1),('ai_controlled',True),('camera_active',True),('pending_receipt_id','other'),('gas',.02),('brake',.98),('speed_kmh',5.01),('speed_kmh',float('nan'))])
def test_transition_cannot_ignore_authority_epoch_or_brake_hold_loss(key,value):
    s=state();s[key]=value
    with pytest.raises(ValueError):validate_transition_context(s,dict(generation='g',epoch=2),(None,'owned'))
