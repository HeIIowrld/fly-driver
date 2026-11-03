"""One-shot cleanup of explicitly failed zero-actuator verifiers with pending teardown."""

import argparse
import ctypes
import json
import os
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.win_process_receipt import ProcessReceipt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    targets = [(26764, 'runs/flight-yaw-trial-verification-20261029-t1159'),
               (27384, 'runs/flight-linear-yaw-drive-verification-20261029-t1232/full')]
    run = begin_run(a.output, dict(scope=__doc__, targets=targets,
                                  cancellation_reason='Explicit recorded verifier exceptions'))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0, records=[])
    try:
        state, modified = read_published_json(
            Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
        assert time.time()-modified < 2 and state['paused'] and not state['camera_active']
        assert state['ai_controlled'] is False
        axis = read('configs/ac-axis-probe.json')
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        assert sdk.GetVJDStatus(axis['device_id']) == 1
        with NativePhysicsReader(timeout=.15) as reader:
            sample = reader.receive()
        assert abs(sample.fields['speedKmh']) < .1
        assert sample.fields['gas'] < .02 and sample.fields['brake'] > .98
        write_json(run/'session-before.json', state)
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.TerminateProcess.restype = wintypes.BOOL
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        for pid, path in targets:
            failure = read(Path(path)/'result.json')
            assert failure['status'] == 'failed' and failure['actuator_messages_sent'] == 0
            receipt = ProcessReceipt(pid)
            try:
                before = receipt.snapshot()
                if not before['active']:
                    result['records'].append(dict(pid=pid, before=before,
                                                  termination_requested=False, after=before))
                    continue
                assert before['exit_requested'] and before['exit_code_reported'] == 1
                query = (f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | "
                         'Select-Object ProcessId,ExecutablePath,CommandLine,CreationDate | '
                         'ConvertTo-Json -Compress')
                observed = subprocess.run(['pwsh', '-NoLogo', '-NoProfile', '-NonInteractive',
                    '-Command', query], capture_output=True, text=True, encoding='utf8',
                    check=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                identity = json.loads(observed.stdout)
                command = identity['CommandLine'].replace('\\', '/')
                assert identity['ProcessId'] == pid
                assert Path(identity['ExecutablePath']).name.lower() == 'python.exe'
                assert 'scripts.verify_flight_yaw_trial' in command and path in command
                handle = api.OpenProcess(0x101001, False, pid)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    requested = bool(api.TerminateProcess(handle, 1))
                    error = 0 if requested else ctypes.get_last_error()
                    waited = api.WaitForSingleObject(handle, 2000)
                    result['records'].append(dict(pid=pid, failed_run=path,
                        failed_result_sha256=sha256(Path(path)/'result.json'), before=before,
                        identity_marker_verified=True, created=identity['CreationDate'],
                        termination_requested=requested, termination_error=error,
                        wait_after_request=waited, after=receipt.snapshot()))
                finally:
                    api.CloseHandle(handle)
            finally:
                receipt.close()
            write_json(run/'partial-result.json', result)
        old = ProcessReceipt(46748)
        try:
            result['earlier_engine_restart_wait_confirmed'] = old.snapshot()
            assert result['earlier_engine_restart_wait_confirmed']['termination_signaled']
            assert result['earlier_engine_restart_wait_confirmed']['exit_code'] == 0
        finally:
            old.close()
        result.update(status='observed_owned_failure_cleanup',
            all_target_terminations_signaled=all(
                not r['after']['active'] for r in result['records']))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
