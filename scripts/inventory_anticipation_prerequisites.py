"""Read-only R3 content availability and unregistered body-axis inventory."""
import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa')
    provenance = [Path('assets/external') / name / 'provenance.json' for name in
                  ['acgym-teacher-candidates-20261102', 'vectorq-miata-teacher-20261102']]
    run = begin_run(a.output, dict(seed=780306, program_sha256=sha256(__file__),
        game_root=str(root), teacher_provenance_sha256={str(p): sha256(p) for p in provenance},
        actor_deployed=False, simulator_changes=0))
    tracks = []
    for folder in sorted((root / 'content/tracks').iterdir()):
        if folder.is_dir():
            ui = sorted(folder.glob('ui/**/ui_track.json'))
            tracks.append(dict(id=folder.name, metadata=[dict(path=str(p), sha256=sha256(p)) for p in ui]))
    teachers = []
    for path in provenance:
        for row in json.loads(path.read_text(encoding='utf8')):
            original = Path(row['path'])
            actual = sha256(original)
            if actual != row['sha256']:
                raise ValueError('Teacher candidate original hash changed')
            teachers.append(dict(path=str(original), sha256=actual,
                availability='Local file hash verified; not new-track teacher qualification'))
    write_json(run / 'result.json', dict(status='completed_readonly_prerequisite_inventory',
        tracks=tracks, track_folder_count=len(tracks), local_teacher_files=teachers,
        qualified_teacher_scope='Previously qualified BMW Z4 GT3/Monza only; no additional teacher qualified in this inventory',
        unregistered_body_channels=['pitch angular rate', 'roll angular rate', 'body acceleration X/Y/Z'],
        body_contract_sha256=sha256('configs/ac-virtual-body-v1.json'),
        body_registration_next='Independent rotation/acceleration unit/sign/time witnesses before any valid-mask change',
        relative_wind='Requires independently registered source/axes/time; speed is not substituted',
        track_change_api_verified_in_this_run=False, new_data_collected=0, optimizer_updates=0,
        simulator_changes=0, vehicle_control_sent=False,
        termination_reason='Local metadata and candidate hashes read; no licensed content copied'))


if __name__ == '__main__':
    main()
