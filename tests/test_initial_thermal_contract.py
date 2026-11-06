from types import SimpleNamespace
import pytest
from flybrain_drive.initial_thermal_contract import ThermalReadback,validate_thermal_state


def sample(packet,temps,*,received=2.,ai=False):
    return SimpleNamespace(physics=SimpleNamespace(packet_id=packet,received_monotonic_seconds=received,
        fields=dict(speedKmh=0,gas=0,brake=1,tyreCoreTemperature=temps)),builtin_ai_controlled=ai,
        authority_record=lambda:dict(packet_id=packet,same_native_copy=True))


def test_duplicate_or_mismatched_wheels_cannot_verify_temperature():
    r=ThermalReadback(40,after_host_clock=1,initial_packet=10)
    assert not r.observe(sample(10,[40]*4))
    assert not r.observe(sample(11,[40]*4))
    assert not r.observe(sample(11,[40]*4))
    assert r.consecutive==1
    assert not r.observe(sample(12,[40,40,40,43]))
    assert r.consecutive==0
    assert not r.observe(sample(13,[40]*4))
    assert not r.observe(sample(14,[40]*4))
    assert r.observe(sample(15,[40]*4))
    assert len(r.observations)==5


@pytest.mark.parametrize('bad',[
    sample(11,[40]*4,received=.9),sample(9,[40]*4),sample(11,[40]*4,ai=True),
    sample(11,[40,40,40,float('nan')])])
def test_bad_receipt_authority_or_reset_is_rejected(bad):
    r=ThermalReadback(40,after_host_clock=1,initial_packet=10)
    with pytest.raises(ValueError):r.observe(bad)


def test_native_verification_does_not_require_another_initialization():
    state=dict(paused=True,is_online=False,in_menu=False,ai_controlled=False,camera_active=False,
        expert_probe_active=False,brake_settle_active=False,physics_api_allowed=True,
        car_id='bmw_z4_gt3',track_id='monza',speed_kmh=0,gas=0,brake=1,epoch=2,
        tyres_initialization_epoch=2,capabilities=dict(initial_tyre_temperature_v1=True))
    with pytest.raises(ValueError,match='already consumed'):
        validate_thermal_state(state,40,initialization=True)
    validate_thermal_state(state,40,initialization=False)
    state['paused']=False
    validate_thermal_state(state,40,initialization=False)
    with pytest.raises(ValueError):validate_thermal_state(state,40,initialization=True)
