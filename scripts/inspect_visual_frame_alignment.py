"""Sample original TRAIN images and check their capture/time alignment to saved neural activity."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Original plan mismatch')
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    replay = json.loads(Path(plan['sensor_replay']).read_text(encoding='utf8'))
    entries = [e for e in replay['episodes'] if e['role'] == 'train']
    if [e['id'] for e in entries] != ['episode-04-train', 'episode-05-train',
                                      'episode-06-train']:
        raise ValueError('Original three train histories required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'plan_sha256': args.plan_sha256, 'role': 'exploratory_training_evidence',
        'student_deployed': False})
    indices = [0, 147, 294, 441, 588, 735]
    contact = Image.new('RGB', (3*320, 6*200), (0, 0, 0))
    draw = ImageDraw.Draw(contact)
    rows, source_hashes = [], {}
    for col, entry in enumerate(entries):
        root = Path(entry['recording'])
        events_path = root/'events.jsonl'
        responses_path = Path(entry['output'])/'responses.npz'
        if sha256(responses_path) != entry['responses_sha256']:
            raise ValueError('Original author response hash mismatch')
        source_hashes[str(events_path)] = sha256(events_path)
        source_hashes[str(responses_path)] = sha256(responses_path)
        with events_path.open(encoding='utf8') as stream:
            events = [json.loads(line) for line in stream]
        with load_array(responses_path, allow_pickle=False) as data:
            captures = data['capture_ids'].copy()
            clocks = data['decision_monotonic_seconds'].copy()
        if len(events) != len(captures) or len(events) != 739:
            raise ValueError('Full original history must be retained')
        zip_path = root/'vision/camera.zip'
        with ZipFile(zip_path) as archive:
            for row, index in enumerate(indices):
                event = events[index]
                views = [v for v in event['camera']['views'] if v['view_id'] == 'front']
                if len(views) != 1:
                    raise ValueError('Exactly one original front view required')
                view = views[0]
                if (int(captures[index]) != event['camera']['capture_id']
                        or float(clocks[index]) != event['decision_monotonic_seconds']):
                    raise ValueError('Saved neural capture or decision clock mismatch')
                contents = archive.read(Path(view['file']).name)
                if hashlib.sha256(contents).hexdigest() != view['file_sha256']:
                    raise ValueError('Original archived image differs from recorded hash')
                with Image.open(BytesIO(contents)) as image:
                    if image.size != (320, 180):
                        raise ValueError('Original front frame shape changed')
                    contact.paste(image.convert('RGB'), (col*320, row*200+20))
                draw.text((col*320+4, row*200+3), f"{entry['id']} / {index}", fill='white')
                rows.append({'episode': entry['id'], 'event_index': index,
                    'capture_id': int(captures[index]), 'decision_clock': float(clocks[index]),
                    'original_bmp_sha256': view['file_sha256'], 'archive': str(zip_path),
                    'archive_member': Path(view['file']).name})
        print(json.dumps({'episode': entry['id'], 'sampled': len(indices)}), flush=True)
    for path, expected in source_hashes.items():
        if sha256(path) != expected:
            raise ValueError('Original source changed during inspection')
    image_path = run/'front-contact-sheet.png'
    contact.save(image_path)
    write_json(run/'result.json', {'status': 'completed_exploratory_sample_alignment',
        'source_hashes': source_hashes, 'rows': rows, 'contact_sheet_sha256': sha256(image_path),
        'sample_indices': indices, 'sampled_neural_capture_and_clock_exact': True,
        'exposure_simulation_alignment_verified': False, 'geometry_labels_created': False,
        'development_opened': False, 'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
