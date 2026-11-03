import copy
import json

import pytest

from flybrain_drive.flight_dataset_lineage import visual_training_lineage
from flybrain_drive.graph import sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def fixture(tmp_path):
    old, data, collection, seal = (tmp_path/name for name in ('old', 'data', 'collection', 'seal'))
    write(old/'dataset.json', {'old': True})
    source = dict(config=dict(dataset_manifest_sha256=sha256(old/'dataset.json')))
    completed, original_plan, episodes, role_seal = [], [], [], {}
    for index, role in enumerate(('train', 'development', 'evaluation')):
        root = collection/f'episode-{index}'
        write(root/'result.json', dict(status='completed'))
        (root/'events.jsonl').write_text('original-'+role, encoding='utf8')
        completed.append(dict(index=index, role=role, path=str(root), child_exit_code=0,
                              result_sha256=sha256(root/'result.json'), decisions=8))
        original_plan.append(dict(index=index, role=role))
        teacher = seal/role
        teacher_episode = dict(file='episode.npz', sha256='target-'+role, steps=8,
                               original_recording=str(root))
        write(teacher/'dataset.json', dict(episodes=[teacher_episode]))
        role_seal[role] = dict(dataset=str(teacher),
                              dataset_manifest_sha256=sha256(teacher/'dataset.json'))
        if role != 'evaluation':
            episodes.append(dict(role=role, observations=8, source_teacher_sha256='target-'+role,
                source_events_sha256=sha256(root/'events.jsonl'),
                original=dict(recording=str(root), teacher_dataset=str(teacher),
                              teacher_episode='episode.npz')))
    write(collection/'collection-plan.json', dict(episodes=original_plan))
    write(collection/'manifest.json', dict(config=dict(
        plan_sha256=sha256(collection/'collection-plan.json'))))
    write(collection/'result.json', dict(episodes=completed))
    write(seal/'result.json', dict(status='completed_fixed_role_sealing', roles_changed=False,
                                  roles=role_seal))
    dataset = dict(episodes=episodes)
    write(data/'dataset.json', dataset)
    plan = dict(data=str(data), new_data_lineage=dict(policy='complete_original_fixed_roles_v1',
                source_dataset=str(old), collection=str(collection), role_seal=str(seal)))
    return plan, source, dataset


def test_explicit_new_data_keeps_frozen_source_and_original_complete_roles(tmp_path):
    plan, source, dataset = fixture(tmp_path)
    report = visual_training_lineage(plan, source, dataset)
    assert report['complete_train_development_histories'] == 2
    assert report['evaluation_in_new_dataset'] is False
    assert report['inherited_source_aid_absence_requalified'] is False


@pytest.mark.parametrize('mutation', ['implicit', 'source_changed', 'reassign', 'duplicate',
                                    'omit', 'truncate', 'evaluation'])
def test_new_data_cannot_bypass_provenance_or_complete_original_roles(tmp_path, mutation):
    plan, source, dataset = fixture(tmp_path)
    if mutation == 'implicit':
        del plan['new_data_lineage']
    elif mutation == 'source_changed':
        write(tmp_path/'old/dataset.json', {'changed': True})
    elif mutation == 'reassign':
        dataset['episodes'][1]['role'] = 'train'
    elif mutation == 'duplicate':
        dataset['episodes'].append(copy.deepcopy(dataset['episodes'][0]))
    elif mutation == 'omit':
        dataset['episodes'].pop()
    elif mutation == 'truncate':
        dataset['episodes'][0]['observations'] = 7
    else:
        dataset['episodes'][0]['original']['recording'] = str(tmp_path/'collection/episode-2')
    with pytest.raises(ValueError):
        visual_training_lineage(plan, source, dataset)


def test_original_same_dataset_guard_does_not_require_transfer_configuration(tmp_path):
    write(tmp_path/'dataset.json', dict(episodes=[]))
    source = dict(config=dict(dataset_manifest_sha256=sha256(tmp_path/'dataset.json')))
    assert visual_training_lineage(dict(data=str(tmp_path)), source, {})['policy'] == (
        'unchanged_original_source_dataset')
