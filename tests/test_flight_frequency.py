import numpy as np
import pytest

from flybrain_drive.flight_frequency import fundamental


def test_known_delayed_sinusoid_recovers_gain_phase_and_offset():
    times = np.arange(800)*.01
    delay, frequency = .023, 8.
    signal = .7+.3*np.sin(2*np.pi*frequency*(times-delay))
    result = fundamental(times, signal, frequency)
    assert result['amplitude'] == pytest.approx(.3)
    assert result['offset'] == pytest.approx(.7)
    assert result['phase_degrees'] == pytest.approx(-360*frequency*delay)
    assert result['r_squared'] == pytest.approx(1.)
    with pytest.raises(ValueError):
        fundamental(times[::-1], signal, frequency)
