import pytest
from flybrain_drive.ac_steering_ramp import steering_ramp


def test_ramp_is_bounded_and_reaches_target():
    values=steering_ramp(.6,-.6)
    assert values[-1] == -.6
    assert all(abs(b-a) <= .05000000001 for a,b in zip([.6]+values,values))
    assert steering_ramp(0.,0.) == []


@pytest.mark.parametrize('args',[(float('nan'),0), (0,1.1), (0,1,.051)])
def test_invalid_ramp_rejected(args):
    with pytest.raises(ValueError):
        steering_ramp(*args[:2],**({'maximum_step':args[2]} if len(args)>2 else {}))
