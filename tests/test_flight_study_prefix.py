import json

import pytest

from flybrain_drive.graph import sha256
from scripts.continue_flight_physical_yaw_study import verified_completed_prefix


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def fixture(tmp_path):
    root, plan = tmp_path/'previous', {'fixed': 'original'}
    blocks, pairs = [], []
    for index in range(2):
        source, audit = root/f'block-{index}', root/f'audit-{index}'
        actual, checked = [], []
        for trial in range(index*4, index*4+4):
            for parent, entries in ((source, actual), (audit, checked)):
                p = parent/f'trial-{trial}'
                save(p/'result.json', {'original_receipt': trial})
                entries.append(dict(path=str(p), result_sha256=sha256(p/'result.json')))
        save(source/'result.json', dict(status='completed_physical_yaw_block',
                                       block=index, trials=actual))
        save(source/'manifest.json', dict(config=dict(plan=plan)))
        pair = [dict(sign=1, iae_improvement_fraction=-.1),
                dict(sign=-1, iae_improvement_fraction=.1)]
        save(audit/'result.json', dict(status='completed_physical_yaw_block_audit',
            block=index, audits=checked, pairs=pair))
        save(audit/'manifest.json', dict(config=dict(
            source_result_sha256=sha256(source/'result.json'))))
        blocks.append(dict(index=index, original=str(source), audit=str(audit),
                           audit_sha256=sha256(audit/'result.json')))
        pairs.extend(pair)
    save(root/'manifest.json', dict(config=dict(plan=plan)))
    save(root/'result.json', dict(status='failed', full_study_complete=False,
                                 blocks=blocks, pairs=pairs))
    save(root/'block-2/result.json', dict(status='failed', trials=[]))
    return root, plan, root/'block-0'


def test_recovery_preserves_every_audited_trial_and_negative_pair(tmp_path):
    root, plan, first = fixture(tmp_path)
    prefix = verified_completed_prefix(root, plan, first)
    assert len(prefix['blocks']) == 2 and len(prefix['pairs']) == 4
    assert prefix['pairs'][0]['iae_improvement_fraction'] == -.1
    assert prefix['provenance']['preserved_actual_trials'] == 8
    assert prefix['provenance']['outcome_selection'] is False


def test_changed_plan_cannot_reuse_prior_outcomes(tmp_path):
    root, _, first = fixture(tmp_path)
    with pytest.raises(ValueError, match='unchanged plan'):
        verified_completed_prefix(root, {'fixed': 'changed'}, first)


def test_modified_actual_receipt_cannot_be_preserved_as_original(tmp_path):
    root, plan, first = fixture(tmp_path)
    save(first/'trial-0/result.json', {'changed': True})
    with pytest.raises(ValueError, match='actual trial changed'):
        verified_completed_prefix(root, plan, first)


def test_no_automatic_reexecution_of_partial_actor_trial_block(tmp_path):
    root, plan, first = fixture(tmp_path)
    save(root/'block-2/result.json', dict(status='failed', trials=[{'index': 8}]))
    with pytest.raises(ValueError, match='zero actor trials'):
        verified_completed_prefix(root, plan, first)
