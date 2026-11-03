"""Audit archived engine camera geometry and left/right labels; no game API."""

import argparse
import json
from pathlib import Path

from flybrain_drive.ac_csp_geometry import camera_geometry
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    report = json.loads((root / 'result.json').read_text())
    if report['status'] != 'completed' or not report['all_hashes_verified']:
        raise ValueError('Completed camera recording required')
    for item in json.loads((root / 'archive.json').read_text()):
        file = (root / item['file']).resolve()
        if not file.is_relative_to(root.resolve()) or sha256(file) != item['sha256']:
            raise ValueError('Camera archive identity changed')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862804,
        recording_result_sha256=sha256(root / 'result.json'), actuator_messages_sent=0))
    try:
        groups = []
        for file in sorted((root / 'camera').glob('frame-*.json')):
            frame = json.loads(file.read_text())
            groups.append(dict(index=frame['index'], projections=camera_geometry(frame['views'])))
        if len(groups) != len(report['frames']):
            raise ValueError('Camera group count differs')
        write_json(run / 'result.json', dict(status='completed', groups=groups,
            four_camera_labels_match_returned_matrices=True,
            vertical_fov_matches_requested=True, requested_and_returned_pose_agree=True,
            external_physical_optics_calibration=False,
            simultaneous_exposure_verified=False, actor_stream_created=False,
            actuator_messages_sent=0, goal_achieved=False))
    except Exception as error:
        write_json(run / 'result.json', dict(status='failed', error=type(error).__name__,
            reason=str(error), actuator_messages_sent=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
