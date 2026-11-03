"""Independently audit bounded camera/native alignment before any student fit.

Keep exact-exposure flags false. A passing certificate is scoped to original
preceding native copy brackets, recorded virtual camera poses and pixel motion.
Renderer/world diagnostics are audit-only and never emitted as actor arrays.
"""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_geometry import camera_geometry
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.camera_body_alignment import (optical_camera,fundamental,
    sampson_pixel_errors,image_gray,tracked_pixels,source_render_witness)
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def native_samples(path,expected_hash):
    if sha256(path)!=expected_hash:
        raise ValueError('Independent native body archive hash differs')
    rows=[];previous=None;expected=0
    with path.open(encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line);authority=row['authority'];copy=row['copy']
            if row['sequence']!=expected or copy.get('synthetic_replay'):
                raise ValueError('Original continuous native copies required')
            expected+=1
            raw=base64.b64decode(authority['prefix_base64'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=authority['prefix_sha256']:
                raise ValueError('Native original bytes differ')
            sample=decode_teacher_physics(raw,authority['received_monotonic_seconds'])
            brackets=[copy[k] for k in ('first_copy_start','first_copy_end','second_copy_start','second_copy_end')]
            if (not np.isfinite(brackets).all() or any(a>b for a,b in zip(brackets,brackets[1:]))
                    or abs(authority['received_monotonic_seconds']-brackets[-1])>1e-9):
                raise ValueError('Original native copy brackets/receipt differ')
            if sample.builtin_ai_controlled or sample.physics.packet_id!=authority['packet_id']:
                raise ValueError('Native authority/packet differs')
            if previous is not None and sample.physics.packet_id<previous:
                raise ValueError('Native packet reset inside episode')
            if sample.physics.packet_id!=previous:
                rows.append(dict(packet_id=sample.physics.packet_id,
                    receipt=authority['received_monotonic_seconds'],
                    first_copy_start=copy['first_copy_start'],copy_end=copy['second_copy_end'],
                    prefix=raw[:PHYSICS_PREFIX.size],speed_m_s=sample.physics.fields['speedKmh']/3.6))
            previous=sample.physics.packet_id
    times=np.array([r['copy_end'] for r in rows])
    if len(rows)<100 or not np.all(np.diff(times)>0):
        raise ValueError('Advancing original native body records required')
    return rows,times


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--diagnostic-incomplete',action='store_true')
    p.add_argument('--plan',default='runs/teacher-synchronized-20261102/camera-alignment-qualification-plan-memory-receive.json')
    p.add_argument('--timing-validator-recovery',help='Separate hash-bound certificate for the declared1600-frame validator cap false rejection')
    a=p.parse_args();root=Path(a.recording)
    parent=read(root/'result.json');camera=read(root/'camera/result.json')
    teacher=read(root/'teacher/result.json')
    plan_path=Path(a.plan)
    plan=read(plan_path)
    run=begin_run(a.output,dict(arguments=vars(a),seed=plan['seed'],
        preregistered_plan_sha256=sha256(plan_path),source_result_sha256=sha256(root/'result.json'),
        camera_result_sha256=sha256(root/'camera/result.json'),teacher_result_sha256=sha256(root/'teacher/result.json'),
        student_optimizer_updates=0,actuator_messages_sent=0))
    result=dict(status='failed',training_admissible=False,student_optimizer_updates=0,
        exact_gpu_exposure_timestamp_verified=False,qualification_scope='bounded_virtual_camera_pose_native_copy_alignment')
    try:
        recovered=False
        if a.timing_validator_recovery:
            recovery_path=Path(a.timing_validator_recovery);recovery=read(recovery_path/'result.json')
            if (recovery['status']!='recovered_original_camera_timing_validation'
                    or recovery.get('eligible_for_independent_audit') is not True
                    or recovery['training_admissible'] is not False
                    or recovery['source_result_sha256']!=sha256(root/'result.json')
                    or recovery['camera_result_sha256']!=sha256(root/'camera/result.json')
                    or recovery['teacher_result_sha256']!=sha256(root/'teacher/result.json')
                    or recovery['source_plan_sha256']!=sha256(plan_path)
                    or recovery['source_candidate_sha256']!=plan['source_candidate_sha256']
                    or recovery['original_camera_bundle_sha256']!=camera['camera_archive']['bundle_sha256']
                    or recovery['source_timing']['status']!='complete'
                    or recovery['original_frames']!=camera['frames']
                    or teacher['status']!='curriculum_completed'
                    or camera['native_body_close']['status']!='completed'):
                raise ValueError('Original timing-validator recovery certificate differs')
            recovered=True
            result['timing_validator_recovery_sha256']=sha256(recovery_path/'result.json')
            result['original_collection_status']=parent['status']
            result['original_camera_status']=camera['status']
        healthy=(parent['status']=='collected_unqualified_demonstration'
            and teacher['status']=='curriculum_completed' and camera['native_body_close']['status']=='completed')
        healthy=healthy or recovered
        if not healthy and not a.diagnostic_incomplete:
            raise ValueError('Complete teacher and independent native/camera collection required')
        if healthy:
            subprocess.run([sys.executable,'-m','scripts.audit_teacher_recovery_laps',
                '--recording',str(root/'teacher'),'--output',str(run/'lap-proof')],check=True,cwd=Path.cwd(),
                stdout=subprocess.DEVNULL)
            lap=read(run/'lap-proof/result.json')
            result['lap_proof_sha256']=sha256(run/'lap-proof/result.json')
        else:
            lap=dict(verified_valid_laps=0,post_stop_verified=False,crossings=[])
            result['lap_proof_not_attempted_in_diagnostic']=True
        if healthy and (lap['verified_valid_laps']<1 or not lap['post_stop_verified']):
            raise ValueError('Original complete valid teacher lap and stopped controls required')
        body,body_times=native_samples(root/'camera/native-body.jsonl',camera['native_body_sha256'])
        limits=plan['timing'];pixel_limits=plan['pixels']
        frames=[];stream=[];geometric=[];render_errors=[];native_ages=[];native_upper_ages=[];body_speed_errors=[]
        cv2.setNumThreads(1)
        ownership=read(root/'camera/ownership-plan.json')
        if ownership['source_sha256']!=plan['source_candidate_sha256']:
            raise ValueError('Installed camera source differs from preregistered witness')
        if plan.get('expected_camera_status_policy'):
            echo=camera['camera_close']['producer_terminal'].get('camera_status_policy_echo',{})
            if (echo.get('policy')!=plan['expected_camera_status_policy']
                    or echo.get('request_id')!=ownership['request_id']
                    or echo.get('generation')!=ownership['generation']):
                raise ValueError('Original owned camera status policy terminal differs')
        if plan.get('expected_camera_stop_policy'):
            echo=camera['camera_close']['producer_terminal'].get('camera_stop_policy_echo',{})
            if (echo.get('policy')!=plan['expected_camera_stop_policy']
                    or echo.get('request_id')!=ownership['request_id']
                    or echo.get('generation')!=ownership['generation']):
                raise ValueError('Original owned MMF stop policy terminal differs')
        with CameraArchive(root/'camera/vision') as archive:
            if recovered:
                raw=archive.read('source-timing-v1.json')
                if sha256_bytes(raw)!=recovery['source_timing_sha256'] or archive.json('consumer-result.json')['error'] is not None:
                    raise ValueError('Recovered original timing or actual camera consumer differs')
            clock=archive.json('clock-initial.json')
            final_clock=archive.json('clock-final.json')
            if max(clock['uncertainty_seconds'],final_clock['uncertainty_seconds'])>.01:
                raise ValueError('Camera/host clock uncertainty exceeds registered 10ms')
            stream=[json.loads(x) for x in archive.read('stream.jsonl').decode('utf8').splitlines()]
            for index,row in enumerate(stream):
                frame=archive.json(Path(row['file']).name)
                if (row['capture_id']!=index or frame['index']!=index
                        or frame['request_id']!=ownership['request_id']
                        or frame['generation']!=clock['generation'] or frame['ideal_line_shown'] is not False
                        or frame['aid_probe_only'] is not False):
                    raise ValueError('Four-view camera source identity/aid contract differs')
                camera_geometry(frame['views'])
                admitted=True;native_indices=[]
                view_rows={v['view_id']:v for v in row['views']}
                for view in frame['views']:
                    name=view['view_id'];record=view_rows[name]
                    if sha256_bytes(archive.read(view['file']))!=record['file_sha256']:
                        raise ValueError('Original image bytes differ from camera receipt')
                    lower=record['render_started_lower']
                    quantization=clock.get('host_clock_quantization_seconds',0.)
                    expected_lower=view['render_started_clock']+clock['offset_lower_seconds']-quantization
                    if abs(expected_lower-lower)>max(1e-9,2*quantization):
                        raise ValueError('Recorded render lower differs from original clock calibration')
                    upper=view['render_started_clock']+clock['offset_upper_seconds']+quantization
                    j=int(np.searchsorted(body_times,lower,side='right')-1)
                    native_indices.append(j)
                    try:
                        source_render_witness(view)
                        if j<0 or not 0<=upper-body[j]['first_copy_start']<=.025:
                            raise ValueError('No preceding native copy within registered 25ms')
                        age=lower-body[j]['first_copy_start'];native_ages.append(age*1000)
                        native_upper_ages.append((upper-body[j]['first_copy_start'])*1000)
                        body_speed_errors.append(abs(view['source_body_before_render']['speed_m_s']-body[j]['speed_m_s']))
                    except ValueError as error:
                        admitted=False;render_errors.append(dict(index=index,view=name,reason=str(error)))
                frames.append(dict(metadata=frame,row=row,admitted=admitted,native_indices=native_indices))
            previous={}
            for index,record in enumerate(frames):
                frame=record['metadata']
                for view in frame['views']:
                    name=view['view_id'];gray=image_gray(archive.read(view['file']))
                    prior=previous.get(name)
                    if prior is not None and record['admitted'] and frames[prior[0]]['admitted']:
                        old_index,old_view,old_gray=prior
                        camera_one,camera_two=optical_camera(old_view),optical_camera(view)
                        baseline=np.linalg.norm(camera_two[2]-camera_one[2])
                        if baseline>.2 and view['source_body_before_render']['speed_m_s']>5:
                            first,second=tracked_pixels(old_gray,gray)
                            entry=dict(first_index=old_index,index=index,view=name,tracks=len(first),passed=False)
                            if len(first)>=pixel_limits['minimum_tracks']:
                                errors=sampson_pixel_errors(first,second,fundamental(camera_one,camera_two))
                                median,p90=map(float,np.quantile(errors,[.5,.9]))
                                entry.update(median_px=median,p90_px=p90,
                                    passed=median<=1 and p90<=2)
                                dt=view['render_started_clock']-old_view['render_started_clock']
                                accel=abs(view['source_body_before_render']['local_angular_velocity'][1]
                                    -old_view['source_body_before_render']['local_angular_velocity'][1])/dt
                                if accel>.3 and index>3 and index<len(frames)-3:
                                    shifted={};shift_times={}
                                    for offset in (-2,-1,1,2):
                                        old={v['view_id']:v for v in frames[old_index+offset]['metadata']['views']}[name]
                                        new={v['view_id']:v for v in frames[index+offset]['metadata']['views']}[name]
                                        shift_times[str(offset)]=new['render_started_clock']-view['render_started_clock']
                                        shifted[str(offset)]=float(np.median(sampson_pixel_errors(first,second,
                                            fundamental(optical_camera(old),optical_camera(new)))))
                                    entry['counterfactual_median_px']=shifted
                                    entry['counterfactual_actual_time_shift_seconds']=shift_times
                            geometric.append(entry)
                    previous[name]=(index,view,gray)
        summary={}
        for name in ('front','left','right','rear'):
            rows=[r for r in geometric if r['view']==name]
            turns=[r for r in rows if 'counterfactual_median_px'in r]
            actual=np.median([r['median_px'] for r in turns]) if turns else None
            shifted={str(k):float(np.median([r['counterfactual_median_px'][str(k)] for r in turns]))
                for k in (-2,-1,1,2)} if turns else {}
            summary[name]=dict(moving_pairs=len(rows),passing_fraction=sum(r['passed'] for r in rows)/len(rows) if rows else 0,
                identifiable_turn_pairs=len(turns),turn_actual_median_px=float(actual) if actual is not None else None,
                turn_counterfactual_medians_px=shifted,
                actual_pose_better_than_shifted=bool(actual is not None and all(actual+.02<v for v in shifted.values())))
        camera_ok=all(s['passing_fraction']>=.8 and s['identifiable_turn_pairs']>=30
            and s['actual_pose_better_than_shifted'] for s in summary.values())
        events=[json.loads(x) for x in (root/'teacher/events.jsonl').read_text(encoding='utf8').splitlines()]
        requests=[e for e in events if e['kind']=='teacher_request']
        pairings=[];received=np.array([f['row']['received'] for f in frames])
        for request in requests:
            decision=request['decision_started_host_time']
            i=int(np.searchsorted(received,decision,side='right')-1)
            j=int(np.searchsorted(body_times,decision,side='right')-1)
            valid=i>=0 and j>=0 and frames[i]['admitted'] and decision-body[j]['receipt']<=.05
            if i>=0:
                valid=valid and all(0<=decision-v['render_started_lower']<=.15 for v in frames[i]['row']['views'])
            pairings.append(dict(camera_index=i,body_index=j,eligible=bool(valid)))
        full_start=lap['crossings'][0]['host_time'] if lap['crossings'] else float('inf')
        full_end=lap['crossings'][-1]['host_time'] if lap['crossings'] else -float('inf')
        full=[pair['eligible'] for pair,request in zip(pairings,requests)
            if full_start<=request['host_time']<=full_end]
        coverage=sum(full)/len(full) if full else 0.
        passed=bool(healthy and camera_ok and coverage==1 and lap['verified_valid_laps']>=1)
        result.update(status='qualified_bounded_camera_body_alignment' if passed else 'diagnostic_alignment_incomplete',
            training_admissible=passed,complete_healthy_collection=healthy,frames=len(frames),
            admitted_frames=sum(f['admitted'] for f in frames),source_render_rejections=render_errors,
            original_native_records=len(body),source_to_render_prior_age_p50_p95_max_ms=np.percentile(native_ages,[50,95,100]).tolist() if native_ages else None,
            source_to_render_latest_bound_p50_p95_max_ms=np.percentile(native_upper_ages,[50,95,100]).tolist() if native_upper_ages else None,
            registered_25ms_test_uses_latest_possible_render_including_clock_uncertainty=True,
            preceding_body_speed_error_p50_p95_max_m_s=np.percentile(body_speed_errors,[50,95,100]).tolist() if body_speed_errors else None,
            per_view_pixel_validation=summary,full_lap_label_coverage=coverage,
            source_to_driver_age_p50_p95_max_ms=lap.get('source_to_driver_age_p50_p95_max_ms'),
            original_receipts_preserved=True,renderer_truth_actor_inputs=0,
            counterfactual_timing='Original adjacent source poses; nominal +/-0.1/0.2s represented by +/-1/2 actual 90ms frames, actual shifts preserved per pair; no interpolated observations',
            camera_bundle_sha256=camera['camera_archive']['bundle_sha256'],native_body_sha256=camera['native_body_sha256'])
        write_json(run/'pairings.json',pairings)
        write_json(run/'pixel-pairs.json',geometric)
        result['pairings_sha256']=sha256(run/'pairings.json')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    write_json(run/'result.json',result)
    print(json.dumps(result))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__=='__main__':
    main()
