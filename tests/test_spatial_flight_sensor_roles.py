import copy

import pytest

from scripts.replay_spatial_flight_sensors import verify_roles


def fixture():
    entries = [dict(path='history'+str(i), role='train' if i < 3 else 'development')
               for i in range(4)]
    collection = dict(status='completed_spatial_teacher_collection',
        completed_counts=dict(train=3, development=1),
        episodes=[dict(path=e['path'], lease=dict(role=e['role'])) for e in entries])
    audit = dict(status='completed_whole_spatial_history_raw_audit',
                 original_collection_task_passed=True, episodes=copy.deepcopy(entries))
    return collection, audit


def test_every_complete_fixed_role_is_required():
    verify_roles(*fixture())


@pytest.mark.parametrize('change', ['failed', 'partial', 'omit', 'duplicate', 'reassign',
                                   'evaluation', 'duplicate_collection'])
def test_failed_partial_or_changed_roles_cannot_enter_replay_batch(change):
    collection, audit = fixture()
    if change == 'failed':
        collection['status'] = 'failed'
    elif change == 'partial':
        audit['status'] = 'completed_partial_spatial_history_raw_audit'
    elif change == 'omit':
        audit['episodes'].pop()
    elif change == 'duplicate':
        audit['episodes'][-1] = audit['episodes'][0]
    elif change == 'reassign':
        audit['episodes'][0]['role'] = 'development'
    elif change == 'evaluation':
        audit['episodes'][-1]['role'] = 'evaluation'
        collection['episodes'][-1]['lease']['role'] = 'evaluation'
    elif change == 'duplicate_collection':
        collection['episodes'].append(collection['episodes'][0])
    with pytest.raises(ValueError):
        verify_roles(collection, audit)
