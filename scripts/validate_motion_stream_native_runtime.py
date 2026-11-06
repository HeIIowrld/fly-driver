"""Original-byte native replay using features produced by the live visual stream."""
import argparse,base64,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.motion_spatial_runtime import MotionSpatialDecisionRuntime,MotionSpatialObservation
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.train_qualified_sac_unified import context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate-plan','source-parity','fit','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--episode',type=int,required=True);a=p.parse_args();torch.set_num_threads(1)
    plan=read(a.candidate_plan);components,port,receipt=load_readonly_motion_candidate(plan)
    if port is None:raise ValueError('Audited fine-motion candidate required')
    root=Path(a.fit);fm=read(root/'manifest.json')['config'];fr=read(root/'result.json')
    if fr['checkpoint_sha256']!=receipt['motion_checkpoint_sha256']:raise ValueError('Candidate checkpoint differs')
    _,_,entries,items,extras,_,_,_,_,_,_,_,_,_=load_motion_experiment(fm['plan'],fm['arguments']['arm']);e=a.episode;d,c,provenance=items[e];source=Path(a.source_parity);sr=read(source/'result.json');sm=read(source/'manifest.json')['config']
    if sr['status']!='completed_motion_stream_source_parity' or not sr['full_episode'] or sr['features_sha256']!=sha256(source/'stream-features.npz') or sm['actor_arrays_sha256']!=provenance['arrays_sha256']:raise ValueError('Exact full-source parity required')
    for path,expected in sm['source_sha256'].items():
        if sha256(path)!=expected:raise ValueError('Live visual source changed '+path)
    with np.load(source/'stream-features.npz') as z:arrays={k:z[k].copy() for k in z.files}
    mapping=arrays['event_to_camera'];np.testing.assert_array_equal(arrays['camera_indices'][mapping],d['camera_index']);static_error=float(abs(arrays['static'][mapping]-extras[e].numpy()).max())
    if static_error>2e-6:raise ValueError('Original static visual path differs')
    cfg=read(Path(entries[e]['data'])/'manifest.json')['config'];native=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';qr=read(Path(cfg['arguments']['qualification'])/'result.json')
    if sha256(native)!=qr['native_body_sha256']:raise ValueError('Original native body stream changed')
    samples={};needed=set(map(int,d['body_packet']))
    for line in native.open(encoding='utf8'):
        auth=json.loads(line)['authority'];pid=auth['packet_id']
        if pid not in needed or pid in samples:continue
        raw=base64.b64decode(auth['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('Native prefix hash differs')
        samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
    if set(samples)!=needed:raise ValueError('Full original native packet coverage required')
    with np.load(root/'predictions.npz') as z:expected=z[f'prediction_{e}'].copy()
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),candidate=receipt,candidate_plan_sha256=sha256(a.candidate_plan),source_parity_result_sha256=sha256(source/'result.json'),source_features_sha256=sr['features_sha256'],runtime_sha256=sha256('src/flybrain_drive/motion_spatial_runtime.py'),actor_arrays_sha256=provenance['arrays_sha256'],optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,seed=None));shutil.copyfile(__file__,out/'executed-entrypoint.py')
    result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False);runtime=None;before=gc.isenabled();gc.disable();maximum=0.;ages=[];predictions=[];start=time.perf_counter()
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Audited CPU numerical mode unavailable')
        runtime=MotionSpatialDecisionRuntime(*components,motion_port=port);write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(components[0]));runtime.begin_episode(c['episode_id'])
        for i in range(len(d['scene'])):
            j=int(mapping[i]);observation=MotionSpatialObservation(c['episode_id'],float(d['camera_receipt'][i]),tuple(map(float,d['render_lowers'][i])),arrays['coarse'][j],arrays['motion'][j])
            output=runtime.decide(decision_time=float(d['decision_time'][i]),scene_features=arrays['scene'][j],static_features=arrays['static'][j],body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])],motion_observation=observation)
            maximum=max(maximum,float(abs(output['prediction']-expected[i]).max()))
            if maximum>2e-6:raise ValueError('Live visual features differ from original learned neural trajectory')
            ages.append(output['estimated_output_body_age_seconds']);predictions.append(output['prediction'])
            if i%1000==0:write_json(out/'progress.json',dict(event=i,total=len(d['scene']),maximum_prediction_difference=maximum,elapsed_seconds=time.perf_counter()-start))
        np.savez_compressed(out/'predictions.npz',prediction=np.asarray(predictions))
        result.update(status='completed_motion_stream_native_runtime_parity',full_episode=True,episode=e,events=len(predictions),maximum_prediction_difference=maximum,static_max_difference=static_error,output_body_age_max_seconds=max(ages),predictions_sha256=sha256(out/'predictions.npz'),state_resets=1,termination_reason='Full live-visual-derived features and original native bytes preserve learned neural trajectory; no actuator authority, live timing or unseen-corner claim')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),events=len(predictions))
    finally:
        if runtime is not None:runtime.close()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
