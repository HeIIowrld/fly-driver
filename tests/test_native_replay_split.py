"""Split provenance rejects relabelled training rows and overlapping recordings."""

import json

import pytest

from flybrain_drive.graph import sha256
from scripts.audit_native_student_replay import split_evidence


def recording(root, identity, image, events):
    dataset = root / 'dataset'
    dataset.mkdir(parents=True)
    (dataset / 'dataset.json').write_text(json.dumps(dict(
        episodes=[dict(sha256=image)], raw_events_sha256=events)), encoding='utf8')
    (root / 'sample-quality.json').write_text(json.dumps([dict(
        source_sdk_sequence=identity, source_native_packet_id=identity,
        image_sha256=image)]), encoding='utf8')
    return dataset


def test_same_manifest_is_training_replay_and_different_requires_provenance(tmp_path):
    train = recording(tmp_path / 'train', 1, 'image1', 'events1')
    other = recording(tmp_path / 'other', 2, 'image2', 'events2')
    model = dict(training_teacher_manifest_sha256=sha256(train / 'dataset.json'))
    assert not split_evidence(model, train)['held_out']
    with pytest.raises(ValueError, match='training-dataset'):
        split_evidence(model, other)
    assert split_evidence(model, other, train)['held_out']
    with pytest.raises(ValueError, match='provenance'):
        split_evidence(model, other, other)


@pytest.mark.parametrize('identity,image,events', [
    (1, 'different-image', 'different-events'),
    (2, 'image1', 'different-events'),
    (2, 'different-image', 'events1'),
])
def test_recording_overlap_is_not_held_out(tmp_path, identity, image, events):
    train = recording(tmp_path / 'train', 1, 'image1', 'events1')
    other = recording(tmp_path / 'other', identity, image, events)
    model = dict(training_teacher_manifest_sha256=sha256(train / 'dataset.json'))
    with pytest.raises(ValueError, match='overlaps'):
        split_evidence(model, other, train)


def test_merged_training_sources_must_be_hash_verified_and_disjoint(tmp_path):
    train = recording(tmp_path / 'train', 1, 'image1', 'events1')
    other = recording(tmp_path / 'other', 2, 'image2', 'events2')
    merged = tmp_path / 'merged'
    merged.mkdir()
    manifest = dict(episodes=[dict(sha256='image1')], sources=[dict(
        directory=str(train), sha256=sha256(train / 'dataset.json'))])
    (merged / 'dataset.json').write_text(json.dumps(manifest), encoding='utf8')
    model = dict(training_teacher_manifest_sha256=sha256(merged / 'dataset.json'))
    assert split_evidence(model, other, merged)['held_out']
    with pytest.raises(ValueError, match='overlaps'):
        split_evidence(model, train, merged)
    with (train / 'dataset.json').open('a', encoding='utf8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='source manifest changed'):
        split_evidence(model, other, merged)
