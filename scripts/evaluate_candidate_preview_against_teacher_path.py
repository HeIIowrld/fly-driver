"""Compare fixed visual preview with privileged original future teacher path; never actor input."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_geometry import camera_geometry, matrix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.teacher_path_preview import future_path_preview


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','raw-audit','preview','body-context','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    a=p.parse_args();candidate,raw,preview,body=map(Path,(a.candidate,a.raw_audit,a.preview,a.body_context))
    root=candidate/'recording';ep=root/'events.jsonl';rp=root/'control/api-receipt.json'
    outer,rr,pr,br=[read(x/'result.json') for x in (candidate,raw,preview,body)]
    rc,pc,bc=[read(x/'manifest.json')['config'] for x in (raw,preview,body)]
    if (outer['status']!='completed_preassigned_pace_recording' or outer['role']!=a.role
            or outer['fit_allowed'] is not False or rr['status']!='completed'
            or pr['status']!='completed_visual_preview_commands_diagnostic' or pr['role']!=a.role
            or br['status']!='completed_high_rate_per_view_body'
            or any(x['events_sha256']!=sha256(ep) for x in (rc,pc,bc))
            or sha256(preview/'commands.npz')!=pr['commands_sha256']
            or sha256(body/'per-view-body.npz')!=br['arrays_sha256']
            or pc['body_sha256']!=br['arrays_sha256']
            or outer['route_admission']['entire_history_retained'] is not True):
        raise ValueError('Exact fixed role, whole route and audited original sources required')
    telemetry=read(rp)['telemetry']
    events=[json.loads(s) for s in ep.read_text(encoding='utf8').splitlines()]
    with np.load(preview/'commands.npz',allow_pickle=False) as f:
        commands,clocks=f['commands'].copy(),f['decision_clock'].copy()
    with np.load(body/'per-view-body.npz',allow_pickle=False) as f:
        speed,valid=f['current_values'][:,16].copy(),f['current_valid'][:,16].copy()
        if not np.array_equal(f['decisions'],clocks):raise ValueError('Original body clocks differ')
    if len(events)!=len(clocks) or commands.shape!=(2,len(events),2):
        raise ValueError('Complete original prediction history required')
    targets=np.zeros((len(events),3,2)); masks=np.zeros((len(events),3),dtype=bool)
    failures=Counter()
    with CameraArchive(root/'vision') as archive:
        for i,event in enumerate(events):
            if event['index']!=i or event['decision_monotonic_seconds']!=clocks[i]:
                raise ValueError('Original ordered decision identity differs')
            cam=event['camera'];raw_frame=archive.read(Path(cam['file']).name)
            if hashlib.sha256(raw_frame).hexdigest()!=cam['file_sha256']:
                raise ValueError('Original frame metadata hash differs')
            frame=json.loads(raw_frame)
            if frame['generation']!=cam['generation'] or frame['index']!=cam['capture_id']:
                raise ValueError('Original frame identity differs')
            try:
                camera_geometry(frame['views'])
                front=next(v for v in frame['views'] if v['view_id']=='front')
                targets[i],masks[i]=future_path_preview(telemetry,front['render_started_clock'],
                    front['renderer_only_world_look'],matrix(front['view_matrix_diagnostic'])[:3,0])
            except ValueError as error:
                failures[str(error)]+=1
    keep=masks[:,2]&valid&(speed>1)&(np.abs(commands[0,:,0])<1.999)
    predicted=np.zeros(len(events));predicted[keep]=-commands[0,keep,0]/speed[keep]
    truth=targets[:,2,1]
    if not keep.any():raise ValueError('No 20m teacher path with valid moving body')
    informative=keep&(np.abs(truth)>=.001)
    delta=predicted[keep]-truth[keep]
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,events_sha256=sha256(ep),
        receipt_sha256=sha256(rp),raw_audit_sha256=sha256(raw/'result.json'),
        preview_sha256=pr['commands_sha256'],body_sha256=br['arrays_sha256'],
        teacher_module_sha256=sha256('src/flybrain_drive/teacher_path_preview.py'),
        scope='post-hoc privileged teacher path evaluation; never actor input'))
    np.savez_compressed(run/'evaluation.npz',decision_clock=clocks,target=targets,target_valid=masks,
                        prediction_curvature=predicted,evaluation_valid=keep,informative=informative)
    out=dict(status='completed_teacher_path_evaluation',role=a.role,events=len(events),
        target20_valid=int(masks[:,2].sum()),evaluated=int(keep.sum()),informative_turns=int(informative.sum()),
        rmse=float(np.sqrt(np.mean(delta**2))),zero_baseline_rmse=float(np.sqrt(np.mean(truth[keep]**2))),
        mean_error=float(np.mean(delta)),sign_accuracy=float(np.mean(np.sign(predicted[informative])==
            np.sign(truth[informative]))) if informative.any() else None,
        left_turns=int((informative&(truth<0)).sum()),right_turns=int((informative&(truth>0)).sum()),
        frame_geometry_failures=dict(failures),evaluation_sha256=sha256(run/'evaluation.npz'),
        truth_is_actual_teacher_path_not_optimal_line=True,truth_never_actor_input=True,
        physical_sensor_exposure_verified=False,dependent_frames_not_independent_trials=True,
        fit_allowed=False,optimizer_updates=0,actor_deployed=False)
    write_json(run/'result.json',out);print(json.dumps(out))


if __name__=='__main__':main()
