import numpy as np
import pytest

from flybrain_drive.flight_learning import GOAL_CONTRACT
from flybrain_drive.racing_goal import (
    RACING_GOAL_CONTRACT,
    racing_goal,
    validate_racing_goal_contract,
)


@pytest.mark.parametrize('speed', [60, 100, 200, 350])
def test_racing_ceiling_preserves_high_speeds_without_thirty_kmh_clipping(speed):
    values = racing_goal(speed, 3)
    np.testing.assert_allclose(values[:, 1]*100, speed)
    np.testing.assert_array_equal(values[:, 0], 0)


@pytest.mark.parametrize('speed', [0, -1, 351, float('nan'), True])
def test_invalid_ceiling_rejected(speed):
    with pytest.raises(ValueError):
        racing_goal(speed, 3)


def test_racing_task_refuses_legacy_constant_speed_goal_semantics():
    with pytest.raises(ValueError, match='legacy'):
        validate_racing_goal_contract(GOAL_CONTRACT)
    validate_racing_goal_contract(RACING_GOAL_CONTRACT)
    assert RACING_GOAL_CONTRACT['teacher_curve_profile_actor_input'] is False
