"""Recorded moving sensory interventions with original continuous circuit state.

Feature counterfactuals are offline hypotheses, never physical temperature trials.
"""
import argparse,base64,gc,hashlib,json,shutil,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_native_visual_candidate import load_readonly_native_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.flight_readonly_preroll import ReadOnlyFlightPreroll
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from flybrain_drive.sensor_contract import readonly_array
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,NavigationBodyObservation,decode_navigation_body_prefix


def temperature_counterfactual(sample,temperature):
    if not np.isfinite(temperature) or temperature<=-273.15:raise ValueError('Finite temperature above absolute zero required')
    raw=sample.raw19.copy();raw[11:15]=temperature
    identity=hashlib.sha256(b'offline-temperature-feature-counterfactual:'+sample.prefix_sha256.encode()+raw.tobytes()).hexdigest()
    return replace(sample,raw19=readonly_array(raw),prefix_sha256=identity)


def action_difference(reference,candidate):
    reference,candidate=np.asarray(reference),np.asarray(candidate)
    if reference.shape!=candidate.shape or reference.ndim!=2 or reference.shape[1]!=3 or not len(reference):raise ValueError('Aligned nonempty three-axis predictions required')
    delta=candidate-reference
    if not np.isfinite(delta).all():raise ValueError('Finite predictions required')
    return dict(maximum_absolute=np.abs(delta).max(axis=0).tolist(),rms=np.sqrt(np.mean(delta**2,axis=0)).tolist(),mean=delta.mean(axis=0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--visual-replay',required=True);p.add_argument('--output',required=True);p.add_argument('--preroll-seconds',type=float);p.add_argument('--original-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    root=Path(a.shadow);m=read(root/'manifest.json')['config'];original=read(root/'result.json')
    visual=Path(a.visual_replay);vr=read(visual/'result.json');vm=read(visual/'manifest.json')['config']
    if vr['status']!='completed_retained_moving_visual_phase_replay' or vm['source_result_sha256']!=sha256(root/'result.json') or sha256(visual/'features.npz')!=vr['features_sha256']:raise ValueError('Original pixel replay binding required')
    ids=vr['passes'][0]['capture_ids']
    with np.load(visual/'features.npz') as z:features={i:(z['reference_scene'][j].copy(),z['reference_static'][j].copy()) for j,i in enumerate(ids)}
    for name,key in [('events.json','events_sha256'),('owner-ticker-events.json','ticker_events_sha256')]:
        if sha256(root/name)!=original[key]:raise ValueError('Original shadow artifact changed')
    if sha256(root/'owner-native-copy.json')!=original['owner_summary']['native_sha256']:raise ValueError('Original body source changed')
    events=read(root/'events.json');ticker=read(root/'owner-ticker-events.json');scenes={e['capture_id']:e for e in events if e.get('status')=='scene_published'}
    if not ticker or any(e['status']!='shadow_output' for e in ticker):raise ValueError('Completed all-output shadow required')
    raw_packets={}
    for row in read(root/'owner-native-copy.json'):
        auth=row['authority'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('Native source bytes changed')
        raw_packets.setdefault(auth['packet_id'],(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds']))
    plan=read(m['arguments']['candidate_plan']);episode=read(root/'resume-identity.json')['request_id']
    encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=m['identity']);inputs=[]
    for event in ticker:
        camera=scenes[event['capture_id']];raw,receipt=raw_packets[event['body_packet']]
        if receipt!=event['body_first_receipt']:raise ValueError('Body first receipt differs')
        current=encoder.encode(decode_navigation_body_prefix(raw,receipt),episode_id=episode,decision_monotonic_seconds=event['decision_time'])
        history=tuple(NavigationBodyObservation(episode,v['packet'],v['receipt'],camera['camera_received'],np.asarray(v['values']),np.asarray(v['valid']),encoder.contract_sha256) for v in camera['view_body'])
        context=per_view_temporal_body_context(history,current,render_start_lowers=tuple(camera['render_start_lowers']),camera_receipt_time=camera['camera_received'],decision_time=event['decision_time'])
        inputs.append((event,context,decode_native_extension(raw,receipt),features[event['capture_id']]))
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),shadow_result_sha256=sha256(root/'result.json'),visual_replay_result_sha256=sha256(visual/'result.json'),candidate_plan=plan,seed=None,optimizer_updates=0,vehicle_control_sent=False,scope='Same original body/clock/pixel features; temperature fields or optical projection only; hypothetical feature interventions, no physical response/action correctness proof'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');arrays={};branches=[];runtime=None;before=gc.isenabled();gc.disable()
    result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Original flush mode unsupported')
        for mode in (['original'] if a.original_only else ['original','temperature40','temperature80','temperature110','optical_neutral']):
            components,receipt=load_readonly_native_candidate(plan)
            if mode=='optical_neutral':
                with torch.no_grad():components[3].projection.weight.zero_()
            runtime=NativeFlightDecisionRuntime(*components);prepare_fixed_cpu_backend(components[0]);runtime.begin_episode(episode)
            preroll=ReadOnlyFlightPreroll(runtime,minimum_seconds=a.preroll_seconds) if a.preroll_seconds is not None else None
            predictions=[];muscles=[];timings=[];ood=[]
            for event,context,sample,(scene,static) in inputs:
                if mode.startswith('temperature'):sample=temperature_counterfactual(sample,float(mode[len('temperature'):]))
                kwargs=dict(decision_time=event['decision_time'],scene_features=scene,static_features=static,body_context=context,native_sample=sample)
                output=preroll.decide(capture_id=event['capture_id'],**kwargs)['output'] if preroll else runtime.decide(**kwargs)
                predictions.append(output['prediction']);muscles.append(output['muscles']);timings.append(output['inference_seconds']);ood.append(output['thermal_out_of_training_support'])
            prediction=np.asarray(predictions);arrays[mode+'_prediction']=prediction;arrays[mode+'_muscles']=np.asarray(muscles);arrays[mode+'_thermal_ood']=np.asarray(ood)
            if mode=='original':
                original_difference=float(abs(prediction-np.asarray([e['prediction'] for e in ticker])).max())
                if original_difference>2e-6:raise ValueError('Original moving predictions not reconstructed: '+str(original_difference))
            branches.append(dict(mode=mode,events=len(prediction),inference_seconds_percentiles=np.percentile(timings,[50,95,99,100]).tolist(),difference_from_original=action_difference(arrays['original_prediction'],prediction),first_prediction=prediction[0].tolist(),last_prediction=prediction[-1].tolist(),temperature_ood_outputs=int(np.any(ood,axis=1).sum())))
            if preroll:
                write_json(run/(mode+'-preroll.json'),preroll.events)
                ready=[(i,e) for i,e in enumerate(preroll.events) if e['prepared']]
                if not ready:raise ValueError('No sensory-prepared continuation')
                branches[-1]['preroll_first_prepared_event']=ready[0][0]
                branches[-1]['preroll_first_prepared']=ready[0][1]
                branches[-1]['preroll_first_prepared_prediction']=prediction[ready[0][0]].tolist()
            runtime.close();runtime=None;write_json(run/'progress.json',branches)
        arrays['decision_time']=np.asarray([e['decision_time'] for e in ticker]);arrays['capture_id']=np.asarray([e['capture_id'] for e in ticker]);arrays['body_packet']=np.asarray([e['body_packet'] for e in ticker])
        result.update(status='completed_moving_native_sensory_interventions',events=len(ticker),original_prediction_max_difference=original_difference,branches=branches,physical_temperature_adaptation_verified=False,action_correctness_verified=False,termination_reason='Original retained pixel predictions reconstructed; continuous original-state feature interventions only. No student actuation, training, lap or limit-driving qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),branches=branches)
    finally:
        if runtime is not None:runtime.close()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    np.savez_compressed(run/'arrays.npz',**arrays);result['arrays_sha256']=sha256(run/'arrays.npz');write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
