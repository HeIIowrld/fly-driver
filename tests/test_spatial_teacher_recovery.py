import copy

import pytest

from scripts.collect_spatial_flight_teacher import recovery_candidates


def fixture():
    entries = [dict(path='episode-'+str(i), lease=dict(role='train')) for i in range(3)]
    previous = dict(status='failed', completed_counts=dict(train=3, development=0),
                    episodes=entries)
    audit = dict(status='completed_partial_spatial_history_raw_audit',
                 episodes=[dict(path=e['path'], role='train') for e in entries])
    return previous, audit


def test_recovery_keeps_all_completed_original_roles_and_does_not_promote_failure():
    previous, audit = fixture()
    original = copy.deepcopy(previous)
    assert recovery_candidates(previous, audit) == previous['episodes']
    assert previous == original and previous['status'] == 'failed'


@pytest.mark.parametrize('change', ['subset', 'role', 'duplicate', 'unaudited', 'completed'])
def test_recovery_cannot_choose_subset_change_role_or_assume_unverified_source(change):
    previous, audit = fixture()
    if change == 'subset':
        previous['episodes'].pop()
    elif change == 'role':
        audit['episodes'][0]['role'] = 'development'
    elif change == 'duplicate':
        audit['episodes'][1] = audit['episodes'][0]
    elif change == 'unaudited':
        audit['status'] = 'failed'
    elif change == 'completed':
        previous['status'] = 'completed_spatial_teacher_collection'
    with pytest.raises(ValueError):
        recovery_candidates(previous, audit)
