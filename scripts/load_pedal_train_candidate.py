"""Audit a whole prospective TRAIN candidate in RAM; save only small reference metadata."""

import argparse
import json
from pathlib import Path

from flybrain_drive.pedal_train_loader import load_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--capture-manifest-sha256', required=True)
    parser.add_argument('--pedal-continuity-review')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    derived = load_candidate(args.capture,
        expected_capture_manifest_sha256=args.capture_manifest_sha256,
        pedal_review_bytes=Path(args.pedal_continuity_review).read_bytes()
            if args.pedal_continuity_review else None)
    # Original arrays/held tasks/camera refs remain in RAM or their original payloads.
    # Only summary/fingerprints/reasons are derivative metadata on disk.
    report = {k: v for k, v in derived.items() if k not in (
        'arrays', 'original_held_tasks', 'original_camera_references', 'original_native420_sha256')}
    report.update(schema='flybrain.pedal_train_candidate.whole_loader.v1',
                  optimizer_updates=0, fit_allowed=False, source_payloads_copied=0)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(observations=derived['observations'], fit_allowed=False,
                          state_frames_dropped=0, metadata_output=args.output)))


if __name__ == '__main__':
    main()
