"""Inventory installed Monza metadata for offline teacher evidence, not actor input."""

import argparse
import configparser
import json
import math
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track-directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.track_directory)
    assets = {'sections': root/'data/sections.ini', 'ai_lane': root/'ai/fast_lane.ai',
              'ideal_line': root/'data/ideal_line.ai', 'main_mesh': root/'monza.kn5'}
    for path in assets.values():
        if not path.is_file():
            raise ValueError('Installed track asset missing: ' + str(path))
    hashes = {name: sha256(path) for name, path in assets.items()}
    config = configparser.ConfigParser(interpolation=None)
    config.read(assets['sections'], encoding='utf8')
    sections = []
    for name in config.sections():
        start, end = float(config[name]['IN']), float(config[name]['OUT'])
        text = config[name]['TEXT'].strip()
        if not (name.startswith('SECTION_') and math.isfinite(start) and math.isfinite(end)
                and 0 <= start <= end <= 1 and text):
            raise ValueError('Unexpected installed section data')
        sections.append({'source_section': name, 'in': start, 'out': end, 'name': text})
    if not sections:
        raise ValueError('No sections in installed file')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'teacher_only': True, 'actor_input': False, 'licensed_asset_copies': 0})
    if any(sha256(path) != hashes[name] for name, path in assets.items()):
        raise ValueError('Installed source changed while auditing')
    result = {'status': 'completed_read_only_inventory', 'track': 'monza',
        'assets': {name: {'path': str(path), 'sha256': hashes[name],
                         'bytes': path.stat().st_size} for name, path in assets.items()},
        'sections': sections, 'section_coordinates_assumed_normalized_spline': False,
        'section_intervals_are_visible_corner_labels': False,
        'ai_lane_and_mesh_formats_parsed': False, 'image_boundary_labels_obtained': False,
        'coordinate_alignment_with_recorded_frames_verified': False,
        'source_assets_copied': False, 'teacher_only': True, 'actor_deployed': False,
        'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
