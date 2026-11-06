"""Summarize original run results; never infer student actions from orchestration defaults."""
import argparse,json
from pathlib import Path
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def collect(base):
    entries=[]
    for file in sorted(base.rglob('result.json')):
        root=file.parent
        if not (root/'manifest.json').exists():continue
        m=read(root/'manifest.json')['config'];args=m.get('arguments',{})
        if 'checkpoint' not in args and 'current_checkpoint' not in args:continue
        r=read(file)
        if 'actor_kind' not in args:continue
        student='checkpoint' in args
        terminal=r.get('actuator_terminal',{})
        feedback_path=root/'actuator/actuator-feedback.json'
        feedback=read(feedback_path) if feedback_path.exists() else []
        requests=[e for e in terminal.get('events',[]) if e['kind']=='student_request' and e.get('receipt',{}).get('driver_accepted')]
        preflight_path=root/'ready-preflight.json'
        preflight=read(preflight_path) if preflight_path.exists() else {}
        source_dir=root/'actor' if student else root
        actor=read(source_dir/'owner-terminal.json') if (source_dir/'owner-terminal.json').exists() else {}
        start_temp=preflight.get('native_fields',{}).get('tyreCoreTemperature')
        source_times=[e['when'] for e in requests]
        item=dict(directory=str(root),kind='student' if student else 'moving_shadow',actor_kind=args['actor_kind'],
            checkpoint_sha256=args.get('checkpoint_sha256',args.get('current_checkpoint_sha256')),
            status=r['status'],student_sdk_accepted=len(requests),student_sdk_attempts=terminal.get('student_request_attempts',len(requests)),
            first_to_last_student_request_seconds=(source_times[-1]-source_times[0]) if source_times else 0.,
            maximum_native_speed_kmh=max((abs(v['fields']['speedKmh']) for v in feedback),default=None),
            maximum_native_displacement_m=max((v.get('displacement_m',0.) for v in feedback),default=None),
            actual_start_temperatures_c=start_temp,actual_student_motion_verified=r.get('actual_student_motion_verified',False),
            stop_reason=terminal.get('stop_reason',r.get('reason',r.get('termination_reason'))),
            stationary_feedback_verified=terminal.get('stationary_feedback_verified'),device_released=terminal.get('device_released'),
            owner_outputs=actor.get('outputs',0),pause_returncode=r.get('pause_returncode'),result_sha256=sha256(file),
            ui_actions_sent=r.get('ui_actions_sent'),full_lap_verified=False)
        if r.get('owner_process_terminal'):item['owner_exitcode']=r['owner_process_terminal']['exitcode']
        elif r.get('owner_terminal',{}).get('exitcode') is not None:item['owner_exitcode']=r['owner_terminal']['exitcode']
        if r.get('actuator_process_terminal'):item['actuator_exitcode']=r['actuator_process_terminal']['exitcode']
        entries.append(item)
    return entries


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/student-closed-loop-20261105');p.add_argument('--output',required=True);a=p.parse_args()
    base=Path(a.base);out=Path(a.output)
    if out.exists():raise ValueError('Unique summary receipt required')
    entries=collect(base)
    write_json(out,dict(schema='flybrain.current_flight.closed_loop_review.v1',runs=entries,
        total_student_sdk_accepted=sum(v['student_sdk_accepted'] for v in entries),
        total_student_sdk_attempts=sum(v['student_sdk_attempts'] for v in entries),
        actor_qualified=False,full_lap_verified=False,goal_status='active',
        authority='Original inner actuator/actor result and raw feedback, not outer sequence default counters'))
    print(json.dumps(dict(runs=len(entries),student_sdk_accepted=sum(v['student_sdk_accepted'] for v in entries))))

if __name__=='__main__':main()
