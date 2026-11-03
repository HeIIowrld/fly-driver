"""Independent raw-byte/RGB replay of paused MMF qualification; no training or actuation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_shared_texture import parse_shared_handle
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.recording)
    report = read(source/'result.json')
    manifest = read(source/'manifest.json')['config']
    request = manifest['request']
    if (report['status'] != 'completed_paused_mmf_same_shot_qualification'
            or report['groups'] != 8 or report['producer_metadata_files'] != 0
            or report['consumer_ack_files'] != 0 or report['producer_terminal']['frames'] != 8
            or report['producer_terminal']['status'] != 'completed'):
        raise ValueError('Completed original eight-group paused qualification required')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        program_sha256=sha256(__file__), source_manifest_sha256=sha256(source/'manifest.json'),
        source_result_sha256=sha256(source/'result.json'),
        original_history_sha256=sha256(source/'history.json')))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    history = read(source/'history.json')
    if len(history) != 8:
        raise ValueError('Entire original qualification history required')
    compared = 0
    copied_files = []
    for index, item in enumerate(history):
        raw = (source/f'frame-{index:04d}.json').read_bytes()
        frame, ack = json.loads(raw), read(source/f'ack-{index:04d}.json')
        receipt = item['receipt']
        if (item['index'] != index or item['stage'] != 'reference_verified'
                or frame != item['frame'] or ack != item['ack']
                or receipt['sequence'] != 2*(index+1)
                or receipt['raw_sha256'] != hashlib.sha256(raw).hexdigest()
                or receipt['repeated_slot_copies_equal'] is not True
                or receipt['producer_slot_modified'] is not False
                or frame['metadata_transport'] != 'mmf_slot_sha256_v1'
                or frame['request_id'] != ack['request_id'] or ack['request_id'] != request['id']
                or frame['generation'] != ack['generation']
                or ack['generation'] != request['expected_generation']
                or frame['index'] != ack['capture_id'] or frame['index'] != index):
            raise ValueError('Original raw metadata/ACK identity differs')
        views = {v['view_id']: v for v in frame['views']}
        copies = {c['view_id']: c for c in item['copied']}
        if set(views) != {'front', 'left', 'right', 'rear'} or set(copies) != set(views):
            raise ValueError('Four original views and copies required')
        for name, view in views.items():
            copied = copies[name]
            if (copied['source_texture_modified'] is not False
                    or copied['original_handle'] != parse_shared_handle(
                        view['shared_texture_handle'])
                    or not receipt['started_at'] <= receipt['finished_at'] <= copied['started_at']
                    <= copied['finished_at'] <= ack['copied_before_ack']
                    <= copied['ack_write_finished_at']):
                raise ValueError('Original metadata-read/copy/ACK ordering differs')
            saved = source/f'{index:04d}-{name}-rgb.npy'
            png = source/f'{index:04d}-{name}-reference.png'
            rgb = np.load(saved, allow_pickle=False)
            with Image.open(png) as image:
                reference = np.asarray(image.convert('RGB'))
            if (rgb.dtype != np.uint8 or rgb.shape != (180, 320, 3)
                    or not np.array_equal(rgb, reference)):
                raise ValueError('Independent same-shot PNG replay differs')
            compared += 1
            copied_files.extend(dict(file=str(p), sha256=sha256(p)) for p in (saved, png))
    write_json(run/'result.json', dict(status='completed_raw_paused_mmf_replay',
        groups=8, pixel_comparisons=compared, maximum_rgb_error=0,
        verified_pixel_files=copied_files,
        actual_host_receipt_order_verified=True, paused_only=True,
        physical_exposure_synchronization_verified=False, live_stream_qualified=False,
        training_admissible=False, student_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
