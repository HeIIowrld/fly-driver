"""Reconstruct teacher-path targets independently and check saved score arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_geometry import camera_geometry,matrix
from flybrain_drive.diagnostic_path_projection import FutureTrace
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','evaluation','preview','body-context','output'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.candidate)/'recording';evaluation=Path(a.evaluation)
    preview,body=Path(a.preview),Path(a.body_context)
    result,config=read(evaluation/'result.json'),read(evaluation/'manifest.json')['config']
    events,receipt=root/'events.jsonl',root/'control/api-receipt.json'
    arrays=evaluation/'evaluation.npz'
    if (result['status']!='completed_teacher_path_evaluation'
            or sha256(events)!=config['events_sha256'] or sha256(receipt)!=config['receipt_sha256']
            or sha256(arrays)!=result['evaluation_sha256']
            or sha256(preview/'commands.npz')!=config['preview_sha256']
            or sha256(body/'per-view-body.npz')!=config['body_sha256']):
        raise ValueError('Exact original and fixed evaluation sources required')
    rows=[json.loads(s) for s in events.read_text(encoding='utf8').splitlines()]
    with np.load(arrays,allow_pickle=False) as f: x={k:f[k].copy() for k in f.files}
    with np.load(preview/'commands.npz',allow_pickle=False) as f: commands=f['commands'].copy()
    with np.load(body/'per-view-body.npz',allow_pickle=False) as f:
        speed,valid=f['current_values'][:,16].copy(),f['current_valid'][:,16].copy()
    trace=FutureTrace(read(receipt)['telemetry']);maximum=0.
    with CameraArchive(root/'vision') as archive:
        for i,event in enumerate(rows):
            if event['index']!=i or event['decision_monotonic_seconds']!=x['decision_clock'][i]:
                raise ValueError('Original sequence differs')
            cam=event['camera'];raw=archive.read(Path(cam['file']).name)
            if hashlib.sha256(raw).hexdigest()!=cam['file_sha256']:raise ValueError('Frame bytes differ')
            frame=json.loads(raw);camera_geometry(frame['views'])
            front=next(v for v in frame['views'] if v['view_id']=='front')
            start,points,available=trace.at(front['render_started_clock'])
            forward,right=(np.asarray(z)[[0,2]] for z in (front['renderer_only_world_look'],
                           matrix(front['view_matrix_diagnostic'])[:3,0]))
            forward,right=forward/np.linalg.norm(forward),right/np.linalg.norm(right)
            displacement=(points-start)[:,[0,2]];f,r=displacement@forward,displacement@right
            chord2=np.sum(displacement**2,axis=1);eligible=available&(f>0)&(chord2>.01)
            reconstructed=np.zeros((3,2));reconstructed[eligible,0]=np.arctan2(r[eligible],f[eligible])
            reconstructed[eligible,1]=2*r[eligible]/chord2[eligible]
            if not np.array_equal(eligible,x['target_valid'][i]):raise ValueError('Target mask differs')
            maximum=max(maximum,float(np.max(np.abs(reconstructed-x['target'][i]))))
    if maximum>1e-9:raise ValueError('Independent target values differ')
    keep=x['target_valid'][:,2]&valid&(speed>1)&(np.abs(commands[0,:,0])<1.999)
    if not np.array_equal(keep,x['evaluation_valid']):raise ValueError('Evaluation mask differs')
    prediction=-commands[0,keep,0]/speed[keep];truth=x['target'][keep,2,1]
    np.testing.assert_allclose(prediction,x['prediction_curvature'][keep],rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean((prediction-truth)**2)),result['rmse'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean(truth**2)),result['zero_baseline_rmse'],rtol=0,atol=1e-12)
    informative=keep&(np.abs(x['target'][:,2,1])>=.001)
    if not np.array_equal(informative,x['informative']):raise ValueError('Turn mask differs')
    sign=float(np.mean(np.sign(x['prediction_curvature'][informative])==np.sign(x['target'][informative,2,1])))
    np.testing.assert_allclose(sign,result['sign_accuracy'],rtol=0,atol=1e-12)
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(events),
        receipt_sha256=sha256(receipt),evaluation_sha256=sha256(arrays),scope=__doc__))
    out=dict(status='completed',events=len(rows),target_max_abs_error=maximum,
        masks_and_metrics_match=True,independent_preview_model_execution=False,
        physical_timing_or_driving_verified=False,fit_allowed=False)
    write_json(run/'result.json',out);print(json.dumps(out))


if __name__=='__main__':main()
