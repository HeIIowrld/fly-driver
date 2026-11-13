import pytest

from flybrain_drive.road_boundary_memory import RoadBoundaryMemory


def section(distance, shift=0.):
    return dict(forward_x_m=distance, valid=True, left_y_m=4. + shift, right_y_m=-4. + shift)


def test_near_conflict_preserves_age_and_does_not_seed_rejected_segments():
    memory = RoadBoundaryMemory(.60)
    memory.update([section(4.), section(8.)])
    for _ in range(3):
        memory.advance(2., 0., .08)
        near, far = memory.update([section(4., 1.), section(8.)], near_conflict_m=.15)
        assert near["source"] == "carried"
        assert near["direct_conflict"]
        assert near["left_y_m"] == 4.
        assert far["source"] == "observed"
    assert near["age_s"] == pytest.approx(.24)
    memory.advance(2., 0., .08)
    near, _ = memory.update([section(4., 1.), section(8.)], near_conflict_m=.15)
    assert near["source"] == "observed"
    assert near["left_y_m"] == 5.


def test_near_conflict_accepts_first_observation_and_keeps_default_direct_priority():
    memory = RoadBoundaryMemory(.60)
    assert memory.update([section(4.), section(8.)], near_conflict_m=.15)[0]["source"] == "observed"
    memory.advance(2., 0., .08)
    assert memory.update([section(4., 1.), section(8.)])[0]["source"] == "observed"
