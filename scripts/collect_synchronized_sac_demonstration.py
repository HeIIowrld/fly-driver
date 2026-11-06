"""Own one camera process and one 7M SAC episode, retaining unqualified data.

Start paused in observed forward gear/full brake. The teacher loads before
camera warmup and uses one verified owned CSP resume. No desktop input.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.ac_published_state import read_published_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis-config',required=True)
    p.add_argument('--calibration-result',required=True)
    p.add_argument('--expected-lua-sha256',required=True)
    p.add_argument('--thermal-initialization-run',help='Experimental condition: audited initial thermal run; not actor input')
    p.add_argument('--alignment-plan',default='runs/teacher-synchronized-20261102/camera-alignment-qualification-plan-memory-receive.json')
    p.add_argument('--teacher-config',default='assets/external/acgym-teacher-candidates-20261102/step_07000000.json')
    p.add_argument('--moving-stop-reference', help='Teacher-only moving stop calibration')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--source-timing-diagnostic',action='store_true')
    p.add_argument('--full-source-timing-diagnostic',action='store_true')
    p.add_argument('--lifecycle-only-camera-status',action='store_true')
    p.add_argument('--mmf-stop-signal',action='store_true')
    p.add_argument('--memory-only-receive-checks',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.full_source_timing_diagnostic:a.source_timing_diagnostic=True
    run=begin_run(a.output,dict(arguments=vars(a),seed=780203,
        teacher_configuration_sha256=sha256(a.teacher_config),
        alignment_plan_sha256=sha256(a.alignment_plan),
        training_admissible=False,student_optimizer_updates=0))
    result=dict(status='planned',ui_actions_sent=0,training_admissible=False,student_optimizer_updates=0)
    snapshots=run/'executed-sources';snapshots.mkdir()
    source_receipts=[]
    for name in ['scripts/collect_synchronized_sac_demonstration.py','scripts/collect_owned_source_body_camera.py',
            'scripts/evaluate_sac_teacher_curriculum.py','scripts/control_ac_session.py',
            'src/flybrain_drive/native_body_file_process.py','src/flybrain_drive/ac_csp_vision.py',
            'src/flybrain_drive/owned_source_camera.py','src/flybrain_drive/ac_latest_tcp.py',
            'src/flybrain_drive/source_native_snapshot.py','src/flybrain_drive/teacher_curriculum.py',
            'src/flybrain_drive/control_guard_metadata_process.py',
            'src/flybrain_drive/ac_metadata_mmf.py','src/flybrain_drive/ac_published_state.py',
            'src/flybrain_drive/owned_file_change_watch.py',
            'src/flybrain_drive/camera_receive_observer.py',
            'src/flybrain_drive/thermal_teacher_start.py','scripts/audit_initial_thermal_evidence.py',
            'integrations/ac/flybrain_telemetry/flybrain_telemetry.py']:
        path=Path(name);target=snapshots/path.name;target.write_bytes(path.read_bytes())
        if sha256(target)!=sha256(path):
            raise ValueError('Executed source snapshot differs')
        source_receipts.append(dict(path=name,snapshot=target.name,sha256=sha256(target)))
    write_json(run/'executed-source-receipts.json',source_receipts)
    if not a.execute:
        write_json(run/'result.json',result)
        return
    nonce=uuid.uuid4().hex
    local=Path(os.environ['LOCALAPPDATA'])/'flybrain'
    coordination=local/'camera-coordination'/('demonstration-'+nonce)
    logs=local/'demonstration-launcher-logs'/nonce
    coordination.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    camera=teacher=None
    handles=[]
    try:
        for name in ('camera.log','teacher.log'):
            handles.append((logs/name).open('x',encoding='utf8'))
        command=[sys.executable,'-m','scripts.collect_owned_source_body_camera',
            '--coordination-directory',str(coordination),'--expected-lua-sha256',a.expected_lua_sha256,
            '--frames',('550' if a.source_timing_diagnostic and not a.full_source_timing_diagnostic else '1600'),'--period','.09',
            ('--source-timing-diagnostic' if a.source_timing_diagnostic else '--source-body-recording'),
            '--output',str(run/'camera')]
        if a.full_source_timing_diagnostic:command.append('--source-body-recording')
        if a.lifecycle_only_camera_status:command.append('--lifecycle-only-camera-status')
        if a.mmf_stop_signal:command.append('--mmf-stop-signal')
        if a.memory_only_receive_checks:command.append('--memory-only-receive-checks')
        camera=subprocess.Popen(command,cwd=Path.cwd(),stdout=handles[0],stderr=subprocess.STDOUT,
            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0)
        write_json(run/'ownership.json',dict(coordination=str(coordination),logs=str(logs),
            camera_pid=camera.pid,camera_command=command))
        deadline=time.perf_counter()+90
        while not (coordination/'ready.json').exists():
            if camera.poll() is not None:
                raise RuntimeError('Camera process ended before prepared readiness')
            if time.perf_counter()>deadline:
                raise TimeoutError('Camera process did not prepare within 90 seconds')
            time.sleep(.1)
        ready=json.loads((coordination/'ready.json').read_text(encoding='utf8'))
        if ready['recorder_pid']!=camera.pid:
            raise ValueError('Camera readiness PID differs from owned child')
        command=[sys.executable,'-m','scripts.evaluate_sac_teacher_curriculum',
            '--axis-config',a.axis_config,'--calibration-result',a.calibration_result,
            '--teacher-config',a.teacher_config,'--transport','tcp','--source-native',
            '--require-quiescent-file-bridge',
            '--shift-handling','policy','--max-gap-m','12',
            '--stage-seconds','3','--full-laps','1','--camera-plan',str(coordination/'ready.json'),
            '--execute','--output',str(run/'teacher')]
        capped=(a.moving_stop_reference is not None and
            json.loads(Path(a.moving_stop_reference).read_text(encoding='utf-8')).get('teacher_pacing_profile')
            =='monza_straight_120_v1')
        if capped:command+=['--targets','20,40,60,80,120','--final-seconds','55']
        else:command.append('--unlimited-final-stage')
        if a.thermal_initialization_run:
            command+=['--thermal-initialization-run',a.thermal_initialization_run]
        if a.moving_stop_reference:
            command+=['--moving-stop-reference',a.moving_stop_reference]
        teacher=subprocess.Popen(command,cwd=Path.cwd(),stdout=handles[1],stderr=subprocess.STDOUT,
            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0)
        write_json(run/'teacher-process.json',dict(pid=teacher.pid,command=command,
            scheduling_priority='ABOVE_NORMAL_PRIORITY_CLASS; no realtime priority or changed freshness budgets'))
        deadline=time.perf_counter()+450
        while teacher.poll() is None:
            if a.source_timing_diagnostic:
                health,_=read_published_json(coordination/'health.json')
                if health.get('frames',0)>=(1600 if a.full_source_timing_diagnostic else 550):
                    (run/'teacher/stop').touch(exist_ok=True)
            if (run/'stop').exists() or time.perf_counter()>deadline or camera.poll() is not None:
                if (run/'teacher').exists():
                    (run/'teacher/stop').touch(exist_ok=True)
                if time.perf_counter()>deadline:
                    raise TimeoutError('Owned teacher exceeded finite 450-second lifetime')
            time.sleep(.1)
        tr=json.loads((run/'teacher/result.json').read_text(encoding='utf8'))
        result.update(teacher_status=tr['status'],teacher_laps=tr.get('laps'),
            teacher_result_sha256=sha256(run/'teacher/result.json'))
        result['status']='collected_unqualified_demonstration' if tr['status']=='curriculum_completed' else 'incomplete_teacher_demonstration'
        if tr['status']=='teacher_moving_stop_calibration_completed':
            result['status']='collected_teacher_moving_stop_calibration'
    except Exception as error:
        result.update(status='failed',error=type(error).__name__,reason=str(error))
    finally:
        if teacher is not None and teacher.poll() is None:
            if (run/'teacher').exists():
                (run/'teacher/stop').touch(exist_ok=True)
            try:
                teacher.wait(timeout=15)
            except subprocess.TimeoutExpired:
                result['owned_teacher_still_running']=teacher.pid
        if teacher is None or teacher.poll() is not None:
            (coordination/'stop').touch(exist_ok=True)
        if camera is not None:
            try:
                camera.wait(timeout=90)
            except subprocess.TimeoutExpired:
                result['owned_camera_still_running']=camera.pid
        for handle in handles:
            handle.close()
        if (run/'camera/result.json').exists():
            cr=json.loads((run/'camera/result.json').read_text(encoding='utf8'))
            result.update(camera_status=cr['status'],camera_frames=cr.get('frames'),
                camera_result_sha256=sha256(run/'camera/result.json'))
            if cr['status'] not in ('stopped_owned_recording','completed_owned_recording'):
                result['status']='failed_camera_demonstration'
        result.update(camera_exitcode=camera.poll() if camera else None,
            teacher_exitcode=teacher.poll() if teacher else None,local_sources_retained=str(logs))
        write_json(run/'result.json',result)
        print(json.dumps(result))


if __name__=='__main__':
    main()
