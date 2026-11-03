"""Repackage one completed local camera source and verify identity with its NAS archive."""

import argparse
import json
import os
import re
import time
from pathlib import Path

from flybrain_drive.ac_camera_archive import CameraArchive, bundle_camera_files
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    recorded = json.loads((root/'result.json').read_text(encoding='utf8'))
    if (recorded['status'] != 'segment_completed'
            or not recorded['vision']['archive_hashes_verified']):
        raise ValueError('Completed hash-verified source recording required')
    request = recorded['vision']['producer_terminal']['request_id']
    if not re.fullmatch(r'async-[0-9a-f]{32}', request):
        raise ValueError('Owned camera request ID required')
    source = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-camera'
    original = root/'vision/archive.json'
    expected = json.loads(original.read_text(encoding='utf8'))
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862880,
        script_sha256=sha256(__file__), source_result_sha256=sha256(root/'result.json'),
        original_archive_manifest_sha256=sha256(original), actuator_messages_sent=0))
    started = time.perf_counter()
    receipt = bundle_camera_files(
        [source/request, source/'client-recordings'/request], run/'vision')
    transferred = time.perf_counter()
    actual = json.loads((run/'vision/archive.json').read_text(encoding='utf8'))
    if actual != expected:
        raise ValueError('Bundled files differ from the original camera archive')
    with CameraArchive(run/'vision') as reader:
        reader.verify_all()
    verified = time.perf_counter()
    write_json(run/'result.json', dict(status='completed', receipt=receipt,
        source_member_manifests_identical=True, files=len(actual),
        source_bytes=sum(x['bytes'] for x in actual),
        bundle_create_transfer_verify_seconds=transferred-started,
        replay_stage_verify_seconds=verified-transferred, total_seconds=verified-started,
        measurements='single real archive, concurrent I/O uncontrolled; no speedup guarantee',
        duplicated_training_samples=0, actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
