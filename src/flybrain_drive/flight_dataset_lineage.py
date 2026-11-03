"""Explicit transfer of a frozen calibrated source to new, fixed-role visual data."""

import json
from pathlib import Path

from .graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def visual_training_lineage(plan, source_manifest, dataset):
    old_hash = source_manifest['config']['dataset_manifest_sha256']
    new_hash = sha256(Path(plan['data'])/'dataset.json')
    if new_hash == old_hash:
        return dict(policy='unchanged_original_source_dataset', dataset_sha256=new_hash)
    lineage = plan.get('new_data_lineage', {})
    if lineage.get('policy') != 'complete_original_fixed_roles_v1':
        raise ValueError('New data requires explicit fixed-role transfer lineage')
    old_data = Path(lineage['source_dataset'])/'dataset.json'
    if sha256(old_data) != old_hash:
        raise ValueError('Frozen source must retain its original dataset provenance')
    collection = Path(lineage['collection'])
    collection_plan = read(collection/'collection-plan.json')
    if sha256(collection/'collection-plan.json') != read(collection/'manifest.json')[
            'config']['plan_sha256']:
        raise ValueError('Original role assignment plan changed')
    completed = read(collection/'result.json')['episodes']
    expected = {}
    for entry in completed:
        declared = dict(index=entry['index'], role=entry['role'])
        if (collection_plan['episodes'][entry['index']] != declared
                or entry['child_exit_code'] != 0
                or sha256(Path(entry['path'])/'result.json') != entry['result_sha256']):
            raise ValueError('Original completed episode or fixed role changed')
        if entry['role'] in ('train', 'development'):
            expected[Path(entry['path']).resolve()] = entry
    seal = read(Path(lineage['role_seal'])/'result.json')
    if seal['status'] != 'completed_fixed_role_sealing' or seal['roles_changed']:
        raise ValueError('Successful original fixed-role seal required')
    seen = set()
    for entry in dataset['episodes']:
        original = entry['original']
        root = Path(original['recording']).resolve()
        if root not in expected or root in seen or entry['role'] != expected[root]['role']:
            raise ValueError('Duplicated history, changed role or evaluation in new training data')
        seen.add(root)
        role = seal['roles'][entry['role']]
        teacher = Path(role['dataset'])
        if (Path(original['teacher_dataset']).resolve() != teacher.resolve()
                or sha256(teacher/'dataset.json') != role['dataset_manifest_sha256']):
            raise ValueError('New data must use its original role-specific teacher seal')
        sources = [e for e in read(teacher/'dataset.json')['episodes']
                   if Path(e['original_recording']).resolve() == root]
        if (len(sources) != 1 or sources[0]['file'] != original['teacher_episode']
                or sources[0]['sha256'] != entry['source_teacher_sha256']
                or sources[0]['steps'] != entry['observations']
                or entry['observations'] != expected[root]['decisions']
                or entry['source_events_sha256'] != sha256(root/'events.jsonl')):
            raise ValueError('Complete original history and unchanged teacher labels required')
    roles = {e['role'] for e in dataset['episodes']}
    if seen != set(expected) or roles != {'train', 'development'}:
        raise ValueError('All completed original train/development histories required')
    return dict(policy=lineage['policy'], original_source_dataset_sha256=old_hash,
        new_dataset_sha256=new_hash,
        collection_plan_sha256=sha256(collection/'collection-plan.json'),
        complete_train_development_histories=len(seen), evaluation_in_new_dataset=False,
        inherited_source_aid_absence_requalified=False,
        scope='New visual boundary fit uses new fixed-role data; calibrated source retained')
