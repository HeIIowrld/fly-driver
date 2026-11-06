from types import SimpleNamespace
import pytest
from flybrain_drive.native_initialization_readiness import NativeInitializationReadiness

def sample(packet,speed=0.,ai=False):
    return SimpleNamespace(builtin_ai_controlled=ai,physics=SimpleNamespace(packet_id=packet,received_monotonic_seconds=float(packet),fields=dict(speedKmh=speed,gas=0.,brake=1.)),authority_record=lambda:{'packet':packet})

def test_startup_motion_and_duplicates_cannot_manufacture_three_settled_packets():
    r=NativeInitializationReadiness()
    assert not r.observe(sample(1,1.45))
    assert not r.observe(sample(2)) and not r.observe(sample(2))
    assert not r.observe(sample(3)) and r.observe(sample(4))
    assert r.consecutive==3 and not any(v['actor_admitted'] for v in r.records)

def test_motion_resets_readiness_without_altering_or_admitting_the_original_sample():
    r=NativeInitializationReadiness();r.observe(sample(1));r.observe(sample(2));assert not r.observe(sample(3,.2))
    assert r.consecutive==0 and r.records[-1]['ready_native'] is False
    assert not r.observe(sample(4)) and not r.observe(sample(5)) and r.observe(sample(6))

@pytest.mark.parametrize('problem',['ai','reset'])
def test_ai_or_packet_reset_fails_instead_of_waiting_through_authority_loss(problem):
    r=NativeInitializationReadiness();r.observe(sample(2))
    with pytest.raises(ValueError):r.observe(sample(3,ai=True) if problem=='ai' else sample(1))
