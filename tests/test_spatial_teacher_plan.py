import pytest

from flybrain_drive.spatial_teacher_plan import next_lease, validate_lease_route

REGIONS = [dict(role='train', start=.27, end=.63),
           dict(role='development', start=.67, end=.84)]


def test_unseen_training_and_separate_development_chosen_before_capture():
    assert next_lease(.32, REGIONS)['role'] == 'train'
    assert next_lease(.68, REGIONS)['role'] == 'development'
    assert next_lease(.95, REGIONS)['role'] == 'transit'
    assert next_lease(.61, REGIONS)['role'] == 'transit'
    assert next_lease(.65, REGIONS)['duration_seconds'] == 3
    assert next_lease(.9, REGIONS)['duration_seconds'] == 30


def test_approach_cannot_skip_development_start_with_recorded_transit_advance():
    # Actual failed-batch route was approaching .67 from roughly .628.
    # A30s .086 advance would land beyond .71, the latest eligible DEV start.
    spline = .628
    leases = []
    for _ in range(8):
        lease = next_lease(spline, REGIONS)
        leases.append(lease)
        if lease['role'] == 'development':
            break
        spline += .0086 if lease['duration_seconds'] == 3 else .086
    assert leases[-1]['role'] == 'development'
    assert len(leases) <= 7
    assert .67 <= spline <= .71
    assert all(e['role'] == 'transit' for e in leases[:-1])


def test_roles_cannot_overlap_or_wrap():
    with pytest.raises(ValueError):
        next_lease(.4, [dict(role='train', start=.2, end=.7),
                        dict(role='development', start=.6, end=.9)])


def receipt(positions):
    def route(p):
        return dict(spline_position=p, actor_input=False, lap_count=0)
    return dict(before=dict(diagnostic_route=route(positions[0])),
                after=dict(diagnostic_route=route(positions[-1])),
                telemetry=[dict(diagnostic_route=route(p), wheels_outside=0) for p in positions])


@pytest.mark.parametrize('positions', [[.32, .5, .64], [.32, .1, .4], [.32, .001, .99]])
def test_entire_crossing_history_rejected_instead_of_selecting_good_frames(positions):
    with pytest.raises(ValueError, match='Whole lease'):
        validate_lease_route(next_lease(.32, REGIONS), receipt(positions))


def test_success_is_only_candidate_admission_not_fit_permission():
    report = validate_lease_route(next_lease(.32, REGIONS), receipt([.32, .35, .4]))
    assert report['role'] == 'train' and report['fit_allowed'] is False
    assert report['entire_history_retained']


def test_unexpected_start_cannot_be_reassigned_after_capture():
    with pytest.raises(ValueError, match='prospective start'):
        validate_lease_route(next_lease(.32, REGIONS), receipt([.34, .36]))
