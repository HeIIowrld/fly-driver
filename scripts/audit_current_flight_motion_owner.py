"""Independently bind and replay actual read-only CPU-owner motion publications."""
import argparse,base64,gc,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.current_flight_runtime import load_current_flight_runtime
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate,require_motion_stream_parity
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.motion_spatial_runtime import MotionSpatialDecisionRuntime,MotionSpatialObservation
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,NavigationBodyObservation,decode_navigation_body_prefix


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    root=Path(a.shadow);r=read(root/'result.json');m=read(root/'manifest.json')['config'];plan=read(m['arguments']['candidate_plan']);components,port,receipt=load_readonly_motion_candidate(plan)
    if port is None or receipt!=m['candidate'] or not r['motion_features_enabled']:raise ValueError('Exact motion-enabled live candidate required')
    proof=require_motion_stream_parity(plan,receipt,'runs/thermal-driving-20261103/spatial-recovery-01/pool-geometry.npz')
    if proof!=read(root/'motion-preflight.json'):raise ValueError('Original live motion preflight differs')
    for name,key in [('events.json','events_sha256'),('owner-ticker-events.json','ticker_events_sha256'),('published-visual-inputs.npz','published_visual_inputs_sha256')]:
        if sha256(root/name)!=r[key]:raise ValueError('Original motion artifact changed '+name)
    if sha256(root/'owner-native-copy.json')!=r['native_sha256']:raise ValueError('Original native source changed')
    ready=read(root/'worker-ready.json');owner=read(root/'owner-terminal.json')
    actor=ready['receipt'];args=m['arguments']
    if not actor['motion_features_required'] or actor['checkpoint_sha256']!=args['current_checkpoint_sha256'] or owner['actor_receipt']!=actor or actor['runtime_kind']!='PowerBoundedSpeedDecisionRuntime':raise ValueError('Exact current CPU actor not established')
    scenes={e['capture_id']:e for e in read(root/'events.json') if e.get('status')=='scene_published'};ticker=read(root/'owner-ticker-events.json')
    rejected=[]
    if ticker and ticker[-1]['status']=='rejected':rejected=[ticker[-1]];ticker=ticker[:-1]
    if not ticker or any(e['status']!='shadow_output' for e in ticker):raise ValueError('Continuous admitted output prefix required')
    with np.load(root/'published-visual-inputs.npz') as z:arrays={k:z[k].copy() for k in z.files}
    ids=arrays['capture_ids'];position={int(i):j for j,i in enumerate(ids)}
    if len(position)!=len(ids) or set(position)!=set(scenes) or len(ids)!=r['published_motion_groups']:raise ValueError('Complete unique published visual coverage required')
    for capture,j in position.items():
        if hashlib.sha256(arrays['motion'][j].tobytes()).hexdigest()!=scenes[capture]['motion_features_sha256']:raise ValueError('Original motion publication bytes differ')
        np.testing.assert_allclose(arrays['coarse'][j],arrays['scene'][j].reshape(4,9,16)[:,1:].reshape(-1),atol=2e-6,rtol=0)
    packets={}
    for row in read(root/'owner-native-copy.json'):
        auth=row['authority'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('Original native bytes differ')
        packets.setdefault(auth['packet_id'],(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds']))
    episode=read(root/'resume-identity.json')['request_id'];encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=m['identity'])
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),shadow_result_sha256=sha256(root/'result.json'),candidate=receipt,published_visual_inputs_sha256=r['published_visual_inputs_sha256'],optimizer_updates=0,vehicle_control_sent=False,seed=None,actor_qualified=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');runtime=None;before=gc.isenabled();gc.disable();predictions=[];maximum=0.;result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Original numerical mode unavailable')
        runtime,loaded=load_current_flight_runtime(args['current_checkpoint'],args['current_checkpoint_sha256']);runtime.begin_episode(episode)
        for event in ticker:
            capture=event['capture_id'];camera=scenes[capture];j=position[capture];raw,received=packets[event['body_packet']]
            if received!=event['body_first_receipt'] or event['motion_features_sha256']!=camera['motion_features_sha256']:raise ValueError('Live CPU transport/source binding differs')
            if event['camera_received']!=camera['camera_received'] or event['render_start_lowers']!=camera['render_start_lowers']:raise ValueError('Live camera clocks differ')
            current=encoder.encode(decode_navigation_body_prefix(raw,received),episode_id=episode,decision_monotonic_seconds=event['decision_time'])
            history=tuple(NavigationBodyObservation(episode,v['packet'],v['receipt'],camera['camera_received'],np.asarray(v['values']),np.asarray(v['valid']),encoder.contract_sha256) for v in camera['view_body'])
            context=per_view_temporal_body_context(history,current,render_start_lowers=tuple(camera['render_start_lowers']),camera_receipt_time=camera['camera_received'],decision_time=event['decision_time'])
            observation=MotionSpatialObservation(episode,camera['camera_received'],tuple(camera['render_start_lowers']),arrays['coarse'][j],arrays['motion'][j])
            output=runtime.decide(decision_time=event['decision_time'],scene_features=arrays['scene'][j],static_features=arrays['static'][j],body_context=context,native_sample=decode_native_extension(raw,received),motion_observation=observation)
            maximum=max(maximum,float(abs(output['prediction']-np.asarray(event['prediction'])).max()))
            if maximum>2e-6:raise ValueError('Original live neural outputs not reproduced')
            predictions.append(output['prediction'])
        np.savez_compressed(out/'predictions.npz',prediction=np.asarray(predictions))
        result.update(status='completed_independent_current_actor_motion_audit',outputs_verified=len(predictions),terminal_rejections_preserved=rejected,whole_shadow_passed=r['status']=='completed_current_actor_teacher_moving_shadow',published_groups_verified=len(ids),maximum_prediction_difference=maximum,motion_transport_hashes_verified=True,same_camera_coarse_reference_verified=True,full_continuous_neural_replay=True,actual_pixel_neural_reconstruction=False,predictions_sha256=sha256(out/'predictions.npz'),termination_reason='Actual retained motion/scene/static publication bytes, per-output hashes, clocks, native prefixes, owner readiness and full continuous neural outputs checked; full episode source-pixel parity is separate evidence, not a new live-pixel replay or student-driving qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),outputs_verified=len(predictions))
    finally:
        if runtime is not None:runtime.close()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    write_json(out/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
