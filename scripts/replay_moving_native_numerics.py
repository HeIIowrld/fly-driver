"""Reconstruct original moving student decisions; diagnose CPU subnormal timing."""
import argparse,base64,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_native_flight_candidate import load_audited_native_flight_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,NavigationBodyObservation,decode_navigation_body_prefix


def count_subnormal(tensor):
    if tensor is None:return 0
    assert tensor.device.type=='cpu' and tensor.dtype==torch.float32
    bits=tensor.detach().contiguous().view(torch.int32)
    return int((((bits & 0x7f800000)==0)&((bits & 0x007fffff)!=0)).sum())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--visual-replay',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1)
    root=Path(a.shadow);m=read(root/'manifest.json')['config'];original=read(root/'result.json')
    visual_root=Path(a.visual_replay);vr=read(visual_root/'result.json');vm=read(visual_root/'manifest.json')['config']
    assert vr['status']=='completed_retained_moving_visual_phase_replay' and vm['source_result_sha256']==sha256(root/'result.json')
    assert sha256(visual_root/'features.npz')==vr['features_sha256']
    ids=vr['passes'][0]['capture_ids']
    with np.load(visual_root/'features.npz') as z:features={i:(z['reference_scene'][j].copy(),z['reference_static'][j].copy()) for j,i in enumerate(ids)}
    events=read(root/'events.json');ticker=read(root/'owner-ticker-events.json')
    assert sha256(root/'events.json')==original['events_sha256'] and sha256(root/'owner-ticker-events.json')==original['ticker_events_sha256']
    scenes={e['capture_id']:e for e in events if e.get('status')=='scene_published'}
    assert sha256(root/'owner-native-copy.json')==original['owner_summary']['native_sha256']
    raw_packets={}
    for row in read(root/'owner-native-copy.json'):
        auth=row['authority'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==auth['prefix_sha256']
        raw_packets.setdefault(auth['packet_id'],(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds']))
    spec=read(m['arguments']['candidate_plan'])['native']
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),shadow_result_sha256=sha256(root/'result.json'),
        visual_replay_result_sha256=sha256(visual_root/'result.json'),optimizer_updates=0,vehicle_control_sent=False,seed=None,
        scope='Original decision clocks/body/retained-pixel features; offline numerical diagnosis, no live or control qualification'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');passes=[];records=[];arrays={};before=gc.isenabled();gc.disable()
    try:
        for flush in [False,True]:
            assert torch.set_flush_denormal(flush)
            components,receipt=load_audited_native_flight_candidate(spec);model=components[0]
            runtime=NativeFlightDecisionRuntime(*components);prepare_fixed_cpu_backend(model)
            episode=read(root/'resume-identity.json')['request_id'];runtime.begin_episode(episode)
            encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=m['identity'])
            predictions=[];timings=[];errors=[];failure=None
            for i,event in enumerate(ticker):
                camera=scenes[event['capture_id']];body_raw,receipt_time=raw_packets[event['body_packet']]
                assert receipt_time==event['body_first_receipt']
                current=encoder.encode(decode_navigation_body_prefix(body_raw,receipt_time),episode_id=episode,decision_monotonic_seconds=event['decision_time'])
                history=tuple(NavigationBodyObservation(episode,v['packet'],v['receipt'],camera['camera_received'],np.asarray(v['values']),np.asarray(v['valid']),encoder.contract_sha256) for v in camera['view_body'])
                context=per_view_temporal_body_context(history,current,render_start_lowers=tuple(camera['render_start_lowers']),camera_receipt_time=camera['camera_received'],decision_time=event['decision_time'])
                scene,static=features[event['capture_id']]
                record=dict(flush_denormal=flush,event=i,original_status=event['status']);records.append(record)
                try:
                    output=runtime.decide(decision_time=event['decision_time'],scene_features=scene,static_features=static,body_context=context,native_sample=decode_native_extension(body_raw,receipt_time))
                    predictions.append(output['prediction']);timings.append(output['inference_seconds'])
                    if event['status']=='shadow_output':
                        difference=float(abs(output['prediction']-event['prediction']).max());errors.append(difference)
                        if difference>2e-6:raise ValueError('Original moving prediction not reconstructed')
                    record['inference_seconds']=output['inference_seconds']
                except Exception as error:
                    failure=dict(event=i,reason=type(error).__name__+': '+str(error),rejection=runtime.last_rejection)
                    record['failure']=failure
                record['state_subnormal_count']=count_subnormal(model.state)
                record['drive_subnormal_count']=count_subnormal(model.previous_drive)
                if failure:break
            state=model.state.detach().numpy().copy();arrays['state_flush_'+str(flush)]=state
            arrays['prediction_flush_'+str(flush)]=np.asarray(predictions)
            passes.append(dict(flush_denormal=flush,admitted_events=len(predictions),original_events=len(ticker),failure=failure,
                state_subnormal_max=max(r['state_subnormal_count'] for r in records if r['flush_denormal']==flush),
                prediction_comparisons=len(errors),original_prediction_max_difference=max(errors) if errors else None,
                inference_seconds_percentiles=np.percentile(timings,[50,95,99,100]).tolist() if timings else []))
            runtime.close();write_json(run/'progress.json',passes)
        result=dict(status='completed_moving_native_numerics_diagnostic',passes=passes,actor_qualified=False,vehicle_control_sent=False,
            live_execution_changed=False,termination_reason='Original moving predictions and hidden-state subnormal diagnostics; numerical candidate remains offline and requires independent audit/live validation')
    except Exception as error:result=dict(status='failed',reason=type(error).__name__+': '+str(error),passes=passes,actor_qualified=False)
    finally:
        torch.set_flush_denormal(False)
        gc.enable() if before else gc.disable()
    write_json(run/'records.json',records);np.savez_compressed(run/'arrays.npz',**arrays)
    result.update(records_sha256=sha256(run/'records.json'),arrays_sha256=sha256(run/'arrays.npz'));write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
