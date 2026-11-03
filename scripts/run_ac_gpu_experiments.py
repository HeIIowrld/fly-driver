"""Bounded independent GPU experiments; no live game/control interfaces."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def gpu_sample(process_ids):
    """Windows per-process counters; never attribute other apps' compute to this suite."""
    if os.name != 'nt' or not process_ids:
        return None
    pattern = '^pid_(' + '|'.join(str(int(p)) for p in sorted(process_ids)) + ')_'
    command = (
        "$e=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine"
        f" | Where-Object Name -Match '{pattern}'"
        " | Select-Object Name,UtilizationPercentage);"
        "$m=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory"
        f" | Where-Object Name -Match '{pattern}'"
        " | Select-Object Name,DedicatedUsage,SharedUsage);"
        "@{engines=$e;memory=$m}|ConvertTo-Json -Compress -Depth 4"
    )
    try:
        result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
            capture_output=True, text=True, timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            return dict(status='unavailable', returncode=result.returncode)
        sample = json.loads(result.stdout)
        compute = [e['UtilizationPercentage'] for e in sample.get('engines', [])
                   if 'Compute' in e['Name']]
        sample['compute_counter_values_valid'] = bool(compute) and all(
            isinstance(v, (int, float)) and 0 <= v <= 100 for v in compute
        ) and sum(compute) <= 100
        # Keep raw values: Windows can emit huge invalid values around context removal.
        return sample
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return dict(status='unavailable', error=type(error).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    workers, experiments = config['workers'], config['experiments']
    if type(workers) is not int or not 1 <= workers <= 3 or not experiments:
        raise ValueError('One to three workers and explicit finite experiments required')
    outputs = [Path(e['output']).resolve() for e in experiments]
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        raise ValueError('Unique unused experiment outputs required')
    run = begin_run(args.output, dict(configuration=config, entry_script_sha256=sha256(__file__),
        experiment_config_hashes={e['config']: sha256(e['config']) for e in experiments},
        scope=__doc__, seed=config['seed'], ui_actions_sent=0, actuator_messages_sent=0))
    started = time.perf_counter()
    pending, active, completed, samples = list(experiments), [], [], []
    try:
        while pending or active:
            while pending and len(active) < workers:
                experiment = pending.pop(0)
                log = (run / (experiment['name'] + '.log')).open('w', encoding='utf8')
                command = [sys.executable, '-m', 'scripts.train_multiview',
                           '--config', experiment['config'], '--teacher', config['dataset'],
                           '--output', experiment['output']]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                active.append(dict(experiment=experiment, process=process, log=log,
                                   started=time.perf_counter()))
            ids = set()
            for job in active[:]:
                device_file = Path(job['experiment']['output']) / 'training-device.json'
                try:
                    ids.add(json.loads(device_file.read_text(encoding='utf8'))['process_id'])
                except (OSError, ValueError, KeyError):
                    pass
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                report_path = Path(job['experiment']['output']) / 'result.json'
                report = json.loads(report_path.read_text(encoding='utf8')) \
                    if report_path.exists() else {}
                completed.append(dict(name=job['experiment']['name'],
                    output=job['experiment']['output'],
                    returncode=code, status=report.get('status', 'missing_result'),
                    seconds=time.perf_counter()-job['started'],
                    checkpoint_sha256=report.get('checkpoint_sha256'),
                    peak_gpu_allocated_bytes=report.get('peak_gpu_memory_allocated_bytes')))
                active.remove(job)
            samples.append(dict(seconds=time.perf_counter()-started,
                                active_jobs=len(active), process_ids=sorted(ids),
                                counters=gpu_sample(ids)))
            write_json(run/'progress.json', dict(completed=completed,
                active=[j['experiment']['name'] for j in active], pending=len(pending)))
            write_json(run/'gpu-samples.json', samples)
            if pending or active:
                time.sleep(2)
        write_json(run/'result.json', dict(
            status='completed' if all(e['returncode'] == 0 and e['status'] == 'completed'
                                      for e in completed) else 'failed',
            workers=workers, experiments=completed, seconds=time.perf_counter()-started,
            max_overlapping_jobs=max(s['active_jobs'] for s in samples),
            counter_scope='Only training process IDs; Windows counters are not resident VRAM proof',
            ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))
    finally:
        # Interrupt only processes created by this suite; never touch other GPU workloads.
        for job in active:
            if job['process'].poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/pid', str(job['process'].pid), '/t', '/f'],
                        capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    job['process'].terminate()
                job['process'].wait(timeout=10)
            job['log'].close()


if __name__ == '__main__':
    main()
