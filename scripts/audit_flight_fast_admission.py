"""Read-only admission audit: every child's actual sources, clocks and terminal process receipt."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_fast_stream import FAST_SOURCES, verified_fast_plan
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
                                    plan_sha256=sha256(args.plan)))
    result = dict(status='failed', goal_achieved=False, actor_qualified=False,
                  actuator_messages_sent=0, ui_actions_sent=0, trials=[])
    try:
        proof = verified_fast_plan(plan)
        assert proof is not None
        root = Path(plan['fast_stream_verification'])
        aggregate, manifest = read(root/'result.json'), read(root/'manifest.json')
        sources = ['src/flybrain_drive/'+module+'.py' for module in FAST_SOURCES]
        sources += ['scripts/verify_flight_fast_stream.py',
                    'scripts/verify_flight_fast_stream_trial.py',
                    'scripts/record_ac_engine_ai_probe.py']
        expected = {file: sha256(file) for file in sources}
        assert all(manifest['source_sha256'][file] == digest for file, digest in expected.items())
        for entry in aggregate['trials']:
            trial = Path(entry['path'])
            child_manifest, child = read(trial/'manifest.json'), read(trial/'result.json')
            assert all(child_manifest['source_sha256'][file] == digest
                       for file, digest in expected.items())
            process_path = root/(trial.name+'-process.json')
            process = read(process_path)
            assert process['status'] == 'observed_worker_termination'
            assert not process['process']['active'] and process['process']['termination_signaled']
            assert process['process']['exit_code'] == process['child_returncode'] == 0
            assert not process['worker_restarted'] and process['failure_cleanup'] is None
            with load_array(trial/'outputs.npz', allow_pickle=False) as archive:
                walls = archive['wall_seconds']
                assert walls.shape == (257,) and np.isfinite(walls).all() and (walls >= 0).all()
                assert (np.percentile(walls, [50, 95, 100]).tolist()
                        == child['wall_seconds_percentiles'])
            result['trials'].append(dict(path=str(trial), child_sources_match=True,
                original_process_signaled_exit0=True, raw_timing_percentiles_match=True,
                process_receipt_sha256=sha256(process_path)))
        assert len(result['trials']) == 12
        result.update(status='completed_fast_stream_admission_audit', certificate=proof,
                      verified_inference_sources=expected, source_checks=12*len(expected),
                      physical_stability_repaired=False, navigation_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
