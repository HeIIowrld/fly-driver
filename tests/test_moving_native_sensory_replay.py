import numpy as np
import pytest
from flybrain_drive.native_multisensory_extension import NativeExtensionSample
from flybrain_drive.sensor_contract import readonly_array
from scripts.replay_moving_native_sensory import temperature_counterfactual,action_difference


def sample():
    return NativeExtensionSample(12,10.,'a'*64,readonly_array(np.arange(19)+1),.2,.3,4.)


def test_temperature_intervention_preserves_other_senses_and_source():
    original=sample();modified=temperature_counterfactual(original,110.)
    np.testing.assert_array_equal(modified.raw19[:11],original.raw19[:11])
    np.testing.assert_array_equal(modified.raw19[15:],original.raw19[15:])
    np.testing.assert_array_equal(modified.raw19[11:15],np.full(4,110.))
    np.testing.assert_array_equal(original.raw19,np.arange(19)+1)
    assert (modified.packet_id,modified.received_monotonic_seconds,modified.pitch_rad,modified.roll_native_rad,modified.speed_m_s)==(12,10.,.2,.3,4.)
    assert modified.prefix_sha256!=original.prefix_sha256
    assert modified.prefix_sha256==temperature_counterfactual(original,110.).prefix_sha256
    assert modified.prefix_sha256!=temperature_counterfactual(original,40.).prefix_sha256
    with pytest.raises(ValueError):modified.raw19[0]=0


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-273.15,-300.])
def test_invalid_counterfactual_temperature_rejected(value):
    with pytest.raises(ValueError):temperature_counterfactual(sample(),value)


def test_action_metrics_distinguish_bias_from_variation():
    value=action_difference(np.zeros((2,3)),[[1,0,-1],[-1,0,1]])
    assert value['mean']==[0.,0.,0.] and value['rms']==[1.,0.,1.] and value['maximum_absolute']==[1.,0.,1.]


@pytest.mark.parametrize('reference,candidate',[(np.zeros((0,3)),np.zeros((0,3))),(np.zeros((2,3)),np.zeros((1,3))),(np.zeros((2,3)),np.full((2,3),np.nan))])
def test_unaligned_or_invalid_action_comparison_rejected(reference,candidate):
    with pytest.raises(ValueError):action_difference(reference,candidate)
