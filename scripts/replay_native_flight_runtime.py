"""Replay qualified native bytes through the read-only flight decision runtime."""
import argparse,base64,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder,decode_native_extension
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.sac_steering_experiment import assembly
from scripts.train_qualified_sac_unified import load_episode,context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-plan');p.add_argument('--fit',required=True);p.add_argument('--fit-audit',required=True)
    p.add_argument('--episode',type=int,choices=[0,1,2],default=2)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.limit<0:raise ValueError('Nonnegative limit; zero means full episode')
    root=Path(a.fit);r=read(root/'result.json');m=read(root/'manifest.json')['config'];audit=read(Path(a.fit_audit)/'result.json')
    if r['status']!='completed_thermal_native_body_port_fit' or audit['status']!='completed_independent_thermal_native_fit_audit':raise ValueError('Completed training and independent audit required')
    if read(Path(a.fit_audit)/'manifest.json')['config']['fit_result_sha256']!=sha256(root/'result.json'):raise ValueError('Audit belongs to another fit')
    path=root/'body-extension.pt';assert sha256(path)==r['checkpoint_sha256']
    cp=torch.load(path,map_location='cpu',weights_only=True);source=Path(cp['source_fit_directory'])
    assert sha256(source/'steering.pt')==cp['source_checkpoint_sha256']
    extension=Path(m['arguments']['extension']);contract=read(extension/'contract.json')
    assert sha256(extension/'contract.json')==cp['extension_contract_sha256']
    entries=three_episode_cohort(m['arguments']['split']);entry=entries[a.episode];d,c,provenance=load_episode(entry['data'])
    assert provenance['arrays_sha256']==entry['arrays_sha256']
    spatial=Path(m['arguments']['spatial']);sr=read(spatial/'result.json');sp=spatial/f'spatial-{a.episode}.npz'
    assert sha256(sp)==sr['files_sha256'][sp.name]
    with np.load(sp) as z:
        j=z['event_to_camera'];np.testing.assert_array_equal(z['camera_indices'][j],d['camera_index'])
        static=np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),z['discarded_grid8'][j]],1)
    cfg=read(Path(entry['data'])/'manifest.json')['config'];native=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl'
    cert=read(Path(cfg['arguments']['qualification'])/'result.json');assert sha256(native)==cert['native_body_sha256']
    samples={}
    with native.open(encoding='utf-8') as f:
        for line in f:
            v=json.loads(line)['authority'];pid=v['packet_id']
            if pid in samples:continue
            raw=base64.b64decode(v['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['prefix_sha256']
            sample=decode_native_extension(raw[:PHYSICS_PREFIX.size],v['received_monotonic_seconds']);assert sample.packet_id==pid;samples[pid]=sample
    model,head,dn,_,_=assembly(source)
    parent=torch.load(source/'steering.pt',map_location='cpu',weights_only=True);state=parent['optical']
    optical=RetinalResidualPort(state['reference'],state['scale'],len(model.motor.backbone.vision_ports));optical.load_state_dict(state)
    state=cp['body_port'];port=MultisensoryBodyFeedbackPort(state['reference'],state['scale'],len(model.motor.backbone.body_ports));port.load_state_dict(state)
    encoder=NativeMultisensoryExtensionEncoder(extension/'contract.json',native_identity=contract['native_identity'])
    candidate_receipt=None
    reference_root=root;reference_result=r;reference_key=f'candidate_{a.episode}'
    if a.candidate_plan:
        from flybrain_drive.audited_native_visual_candidate import load_readonly_native_candidate
        candidate_plan=read(a.candidate_plan)
        (model,head,dn,optical,port,encoder),candidate_receipt=load_readonly_native_candidate(candidate_plan)
        if candidate_receipt['native_checkpoint_sha256']!=r['checkpoint_sha256']:raise ValueError('Replay native fit differs')
        if 'visual' in candidate_plan:
            reference_root=Path(candidate_plan['visual']['fit']);reference_result=read(reference_root/'result.json');reference_key=f'prediction_{a.episode}'
        else:
            reference_root=Path(candidate_plan['native']['head']);reference_result=read(reference_root/'result.json')
    runtime=NativeFlightDecisionRuntime(model,head,dn,optical,port,encoder)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),runtime_sha256=sha256('src/flybrain_drive/native_flight_decision_runtime.py'),
        fit_result_sha256=sha256(root/'result.json'),fit_audit_sha256=sha256(Path(a.fit_audit)/'result.json'),entry=entry,
        candidate=candidate_receipt,reference_predictions_sha256=sha256(reference_root/'predictions.npz'),native_source_sha256=sha256(native),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,
        scope='Continuous recorded episode, original clock domain and native bytes; actual host computation50ms gate. No live-camera or physical exposure claim'))
    write_json(run/'backend-initialization.json',prepare_fixed_cpu_backend(model))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');shutil.copyfile('src/flybrain_drive/native_flight_decision_runtime.py',run/'executed-runtime.py')
    out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0);pred=[];timings=[];ages=[];maximum_delta=0.;start=time.monotonic()
    try:
        runtime.begin_episode(entry['episode_id']);n=len(d['scene']) if not a.limit else min(a.limit,len(d['scene']))
        for i in range(n):
            obs=runtime.decide(decision_time=float(d['decision_time'][i]),scene_features=d['scene'][i],static_features=static[i],body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])])
            pred.append(obs['prediction']);timings.append(obs['inference_seconds']);ages.append(obs['estimated_output_body_age_seconds']);maximum_delta=max(maximum_delta,obs['maximum_body_delta'])
            if i%1000==0:write_json(run/'progress.json',dict(event=i,total=n,elapsed_seconds=time.monotonic()-start))
        assert sha256(reference_root/'predictions.npz')==reference_result['predictions_sha256']
        with np.load(reference_root/'predictions.npz') as z:err=float(abs(np.asarray(pred)-z[reference_key][:n]).max())
        if err>2e-6:raise ValueError('Runtime differs from protected offline trained candidate')
        out.update(status='completed_native_flight_runtime_replay',events=n,full_episode=n==len(d['scene']),source_prediction_max_difference=err,
            inference_seconds_percentiles=np.percentile(timings,[50,95,99,100]).tolist(),output_body_age_max_seconds=max(ages),maximum_body_delta=maximum_delta,
            old_body_masks_unchanged=True,termination_reason='Qualified original native bytes passed same-packet causal runtime, recurrent replay parity and50ms computed-output guard; not live or autonomous qualification')
    except Exception as error:
        out.update(error=type(error).__name__,reason=str(error),events=len(pred),rejected_event=len(pred),runtime_poisoned=runtime.poisoned,rejection=runtime.last_rejection,termination_reason='Explicit runtime parity/freshness failure; no retry, reset or deployment')
    finally:
        runtime.close();np.savez_compressed(run/'predictions.npz',prediction=np.asarray(pred),inference_seconds=np.asarray(timings),output_body_age_seconds=np.asarray(ages))
    out['predictions_sha256']=sha256(run/'predictions.npz');write_json(run/'result.json',out)
    if out['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
