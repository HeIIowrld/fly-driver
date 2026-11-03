"""Replay all four complete geographic histories; keep roles, clocks and invalid body channels."""

import argparse
import json
import subprocess
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_roles(collection, audit):
    if (collection['status'] != 'completed_spatial_teacher_collection'
            or collection['completed_counts'] != dict(train=3, development=1)
            or audit['status'] != 'completed_whole_spatial_history_raw_audit'
            or audit['original_collection_task_passed'] is not True):
        raise ValueError('Complete geographic batch and whole raw audit required')
    selected = [e for e in collection['episodes'] if e['lease']['role'] != 'transit']
    expected = {str(Path(e['path'])): e['lease']['role'] for e in selected}
    actual = {str(Path(e['path'])): e['role'] for e in audit['episodes']}
    if (len(selected) != 4 or len(audit['episodes']) != 4 or len(actual) != 4 or expected != actual
            or list(actual.values()).count('train') != 3
            or list(actual.values()).count('development') != 1):
        raise ValueError('Every preassigned history and original role required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan = read(a.plan)
    collection, audited = Path(plan['collection']), Path(plan['audits'])
    terminal, audit = read(collection/'result.json'), read(audited/'result.json')
    verify_roles(terminal, audit)
    if read(audited/'manifest.json')['config']['collection_result_sha256'] != sha256(
            collection/'result.json'):
        raise ValueError('Audit belongs to a different collection')
    run = begin_run(a.output, dict(plan=plan, plan_sha256=sha256(a.plan),
        program_sha256=sha256(__file__), audit_result_sha256=sha256(audited/'result.json'),
        collection_result_sha256=sha256(collection/'result.json'), optimizer_updates=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', episodes=[], optimizer_updates=0,
                  student_deployed=False, goal_achieved=False)
    bootstrap = ('import sys,site,runpy,gc; p=sys.argv.pop(1); site.addsitedir(p); '
        'sys.path.insert(0,p); runpy.run_module("scripts.replay_flyvis_cameras",run_name="__main__"); '
        'gc.collect(); import torch; torch.set_default_device("cpu"); torch.cuda.empty_cache()')
    try:
        for entry in audit['episodes']:
            for source, digest in plan['source_sha256'].items():
                if sha256(source) != digest:
                    raise ValueError('Pinned sensor replay source changed')
            root = Path(entry['path'])
            raw = Path(entry['raw_audit'])
            if (sha256(root/'result.json') != entry['original_result_sha256']
                    or sha256(raw/'result.json') != entry['raw_audit_result_sha256']
                    or read(root/'manifest.json')['config']['sensory_configuration'] != read(
                        plan['camera_config'])):
                raise ValueError('Exact original recording, camera and raw audit required')
            output = run/root.name
            body16, body21, vision = (output/k for k in ('body16', 'body21', 'vision'))
            commands = [
                [plan['cpu_python'], '-m', 'scripts.replay_virtual_body', '--recording', str(root),
                 '--output', str(body16)],
                [plan['cpu_python'], '-m', 'scripts.replay_virtual_body', '--recording', str(root),
                 '--navigation-contract', plan['body_contract'], '--legacy-reference', str(body16),
                 '--output', str(body21)],
                [plan['gpu_python'], '-c', bootstrap, plan['gpu_site'], '--recording', str(root),
                 '--body-replay', str(body21), '--model-dir', plan['model_dir'],
                 '--cache-root', plan['cache_root'], '--camera-config', plan['camera_config'],
                 '--fractional-frozen-tail', '--retinotopic-types', 'L1,L2,Mi1,Tm3,Tm1,Tm9',
                 '--output', str(vision)]]
            output.mkdir()
            for index, command in enumerate(commands):
                with (output/('step-'+str(index)+'.log')).open('x', encoding='utf8') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                    write_json(run/'progress.json', dict(active=True, episode=root.name,
                        role=entry['role'], step=index, pid=child.pid,
                        completed=len(result['episodes'])))
                    print(json.dumps(dict(episode=root.name, step=index, pid=child.pid)), flush=True)
                    code = child.wait()
                if code:
                    raise RuntimeError('Sensor replay child failed; retain original result/log')
            b, v = read(body21/'result.json'), read(vision/'result.json')
            if (b['status'] != 'completed' or not b['legacy16_whole_replay_exact']
                    or v['status'] != 'completed' or v['weights_changed']
                    or v['observations'] != entry['observations']):
                raise ValueError('Whole original body/visual episode replay required')
            result['episodes'].append(dict(id=root.name, role=entry['role'],
                recording=str(root), body_replay=str(body21), output=str(vision),
                responses_sha256=v['responses_sha256'], result_sha256=sha256(vision/'result.json'),
                body_result_sha256=sha256(body21/'result.json'), observations=v['observations']))
            write_json(run/'partial-result.json', result)
        result.update(status='completed_spatial_flight_sensor_replay', roles_unchanged=True,
                      author_weights_changed=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
        write_json(run/'progress.json', dict(active=False, status=result['status'],
                                             completed=len(result['episodes'])))


if __name__ == '__main__':
    main()
