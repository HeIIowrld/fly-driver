"""Read-only audit of SAC teacher, current session and historical control profile."""
import argparse
import configparser
import json
import time
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat


def ini(path):
    config = configparser.ConfigParser(interpolation=None, strict=False)
    with Path(path).open(encoding='utf-8-sig') as stream:
        config.read_file(stream)
    return {section:dict(config[section]) for section in config.sections()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text(encoding='utf8'))
    if sha256(__file__)!=plan['program_sha256']:raise ValueError('program hash mismatch')
    for name,digest in plan['pinned'].items():
        if sha256(name)!=digest:raise ValueError('pinned source differs: '+name)
    run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),actor=False))
    result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0)
    try:
        teacher=json.loads(Path(plan['teacher_inspection']).read_text(encoding='utf8'))
        manifest=json.loads(Path(plan['historical_manifest']).read_text(encoding='utf8'))
        history=json.loads(Path(plan['historical_audit']).read_text(encoding='utf8'))
        replay=json.loads(Path(plan['replay_result']).read_text(encoding='utf8'))
        native=json.loads(Path(plan['native_bracket_result']).read_text(encoding='utf8'))
        state,modified,_=read_session_heartbeat(plan['session_status']);validate_heartbeat(state,time.time()-modified,mutating=False)
        original=ini(plan['historical_controls']);current=ini(plan['current_controls'])
        changes=[]
        for section in sorted(set(original)|set(current)):
            for key in sorted(set(original.get(section,{}))|set(current.get(section,{}))):
                x=original.get(section,{}).get(key);y=current.get(section,{}).get(key)
                if x!=y:changes.append({'section':section,'key':key,'historical':x,'current':y})
        historical_axis=manifest['config']['axis_configuration'];teacher_cfg=manifest['config']['teacher_configuration']
        if sha256(plan['teacher_checkpoint'])!=teacher['sha256'] or teacher_cfg['checkpoint_sha256']!=teacher['sha256']:
            raise ValueError('teacher checkpoint lineage mismatch')
        assists=ini(plan['current_assists']).get('ASSISTS',{})
        result.update(status='completed',current_car=state.get('car_id'),current_track=state.get('track_id'),
            historical_car=historical_axis['car_model'],historical_track=historical_axis['track'],
            car_matches=state.get('car_id')==historical_axis['car_model'],
            track_matches=state.get('track_id')==historical_axis['track'],
            current_controls_sha256=sha256(plan['current_controls']),
            historical_controls_sha256=sha256(plan['historical_controls']),
            control_files_byte_identical=sha256(plan['current_controls'])==sha256(plan['historical_controls']),
            control_ini_changes=changes,control_ini_semantic_equal=not changes,
            controls_file_not_proof_of_loaded_runtime=True,
            current_assists_file={key:assists.get(key) for key in ('stability_control','auto_brake','auto_shifter','abs','traction_control')},
            current_assists_file_sha256=sha256(plan['current_assists']),
            current_setup_file_sha256=sha256(plan['current_setup']),
            original_teacher_vehicle_setup_available=False,
            teacher_checkpoint_sha256=teacher['sha256'],teacher_profile=teacher_cfg['profile'],
            exact_teacher_training_manifest_available=teacher_cfg['training_run_exact_configuration_available'],
            previous_short_run={k:history.get(k) for k in ('teacher_requests','feedback_count','maximum_raw_sdk_speed_kmh','maximum_recorded_displacement_m','maximum_tyres_out','feedback_gap_range_m')},
            teacher_full_lap_verified=False,teacher_reconstruction_checks=replay['checks'],
            teacher_replay_rows=replay['teacher_replayed_rows'],
            reconstruction_torch_numpy_max_error=replay['torch_numpy_action_max_absolute_error'],
            native_csp_bracket=native['observed_before_native_after'],
            scope='Read-only compatibility audit. Vehicle setup, live settings and teacher lap quality remain unverified; no training or vehicle control.')
    except Exception as exc:result.update(error=type(exc).__name__,reason=str(exc))
    write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
