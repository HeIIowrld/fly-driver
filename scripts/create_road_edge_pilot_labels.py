"""Record four manually inspected, uncertain road-edge intervals, for offline pilot only."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

# Manual observations of the road-facing edge at y=120 in original 320x180 frames.
# Bounds intentionally encode human visual uncertainty; absent sides remain unknown.
OBSERVATIONS = [
    ('episode-04-train', 0, 'right', 260, 285),
    ('episode-04-train', 735, 'right', 285, 319),
    ('episode-05-train', 588, 'right', 225, 255),
    ('episode-06-train', 294, 'left', 100, 120),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment-result', required=True)
    parser.add_argument('--alignment-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.alignment_result) != args.alignment_sha256:
        raise ValueError('Original alignment result changed')
    alignment = json.loads(Path(args.alignment_result).read_text(encoding='utf8'))
    if (alignment['status'] != 'completed_exploratory_sample_alignment'
            or alignment['sampled_neural_capture_and_clock_exact'] is not True):
        raise ValueError('Expected original sampled alignment')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'alignment_sha256': args.alignment_sha256, 'role': 'manual_uncertain_pilot',
        'actor_input': False, 'actor_deployed': False})
    sheet = Image.new('RGB', (640, 400), 'black')
    draw = ImageDraw.Draw(sheet)
    annotations = []
    for i, (episode, index, side, xmin, xmax) in enumerate(OBSERVATIONS):
        matches = [r for r in alignment['rows']
                   if r['episode'] == episode and r['event_index'] == index]
        if len(matches) != 1 or not (0 <= xmin <= xmax < 320):
            raise ValueError('Unique aligned frame and valid pixel interval required')
        row = matches[0]
        with ZipFile(row['archive']) as archive:
            contents = archive.read(row['archive_member'])
        if hashlib.sha256(contents).hexdigest() != row['original_bmp_sha256']:
            raise ValueError('Original frame hash mismatch')
        with Image.open(BytesIO(contents)) as image:
            if image.size != (320, 180):
                raise ValueError('Expected original camera geometry')
            x, y = (i % 2)*320, (i // 2)*200+20
            sheet.paste(image.convert('RGB'), (x, y))
        draw.text((x+2, y-18), f'{episode} {index} {side}', fill='white')
        draw.line((x+xmin, y+120, x+xmax, y+120), fill=(255, 0, 255), width=3)
        annotations.append({'episode': episode, 'event_index': index,
            'capture_id': row['capture_id'], 'decision_clock': row['decision_clock'],
            'source_bmp_sha256': row['original_bmp_sha256'], 'image_width': 320,
            'image_height': 180, 'edge_side': side, 'row_y': 120,
            'x_inclusive_interval': [xmin, xmax], 'other_edge': 'unknown',
            'corner_direction_or_distance_label': 'unknown'})
    image = run/'pilot-overlays.png'
    sheet.save(image)
    write_json(run/'result.json', {'status': 'completed_manual_uncertain_pilot',
        'annotations': annotations, 'overlay_sha256': sha256(image),
        'semantics': 'approximate image-space road-facing edge; not track map or physical distance',
        'manually_inspected_not_independently_reviewed': True,
        'label_intervals_not_exact_pixels': True,
        'role': 'train_pilot_only', 'eligible_for_independent_evaluation': False,
        'models_trained': 0, 'actor_deployed': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
