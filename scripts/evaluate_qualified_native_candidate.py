"""Full qualified new-episode evaluation of the audited visual/native flight actor.

Original recurrent timing and masks persist. Teacher labels only score outputs.
"""
import argparse,base64,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_native_visual_candidate import load_readonly_native_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension,assessed_extension_values
from flybrain_drive.sac_unified_student import read
from scripts.diagnose_sac_anticipation import load_verified
from scripts.improve_sac_anticipation import command_metrics
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.fit_bounded_sac_steering import pre_mask
from scripts.train_qualified_sac_unified import context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['split','spatial','candidate-plan','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--episode',type=int,choices=[0,1,2],required=True);p.add_argument('--case',choices=['normal','optical_neutral','native_neutral'],default='normal');p.add_argument('--flush-denormal',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);entries=three_episode_cohort(a.split);entry=entries[a.episode];d=load_verified(entry);actor=Path(entry['data']);contract=read(actor/'contract.json');cfg=read(actor/'manifest.json')['config']
    spatial=Path(a.spatial);sr=read(spatial/'result.json');sm=read(spatial/'manifest.json')['config'];sp=spatial/f'spatial-{a.episode}.npz'
    if sr['status']!='completed_qualified_static_spatial_recovery' or sm['entries']!=entries or sha256(sp)!=sr['files_sha256'][sp.name]:raise ValueError('Same qualified cohort spatial recovery required')
    with np.load(sp) as z:
        j=z['event_to_camera'];np.testing.assert_array_equal(z['camera_indices'][j],d['camera_index']);static=np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),z['discarded_grid8'][j]],1)
    source=Path(cfg['arguments']['recording']);cert=read(Path(cfg['arguments']['qualification'])/'result.json');native=source/'camera/native-body.jsonl'
    if sha256(native)!=cert['native_body_sha256']:raise ValueError('Original qualified native source changed')
    needed=set(map(int,d['body_packet']));samples={}
    for line in native.open(encoding='utf-8'):
        auth=json.loads(line)['authority'];packet=auth['packet_id']
        if packet in samples or packet not in needed:continue
        raw=base64.b64decode(auth['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('Native bytes changed')
        sample=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
        if sample.packet_id!=packet:raise ValueError('Native packet mismatch')
        samples[packet]=sample
    if set(samples)!=needed:raise ValueError('Actor body packets missing from source')
    sign_source=Path('runs/steering-command-loss-20261103/fit-03/manifest.json');sign=read(sign_source)['config']['sign_evidence']
    if not sign['training_only'] or sign['agreement']<.9:raise ValueError('Original TRAIN-only steering sign evidence required')
    pre=pre_mask(d,sign['sign'])
    if not pre.any():raise ValueError('No pre-corner labels; do not report a zero error')
    plan=read(a.candidate_plan);components,receipt=load_readonly_native_candidate(plan)
    if a.case=='optical_neutral':
        with torch.no_grad():components[3].projection.weight.zero_()
    if a.case=='native_neutral':
        with torch.no_grad():components[4].projection.weight.zero_()
    runtime=NativeFlightDecisionRuntime(*components)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),runtime_sha256=sha256('src/flybrain_drive/native_flight_decision_runtime.py'),entry=entry,spatial_result_sha256=sha256(spatial/'result.json'),candidate_plan_sha256=sha256(a.candidate_plan),candidate=receipt,native_source_sha256=sha256(native),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,external_racing_ceiling_kmh=300,sign_evidence=sign,sign_source_sha256=sha256(sign_source),scope='Original full episode including burn-in, scoring only original qualified full-lap mask. New TRAIN candidate diagnostic, not sealed generalization, actual student lap or action-optimality proof. Neutralization removes named adapter only; all other inputs unchanged.'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');shutil.copyfile('src/flybrain_drive/native_flight_decision_runtime.py',run/'executed-runtime.py')
    write_json(run/'backend-initialization.json',prepare_fixed_cpu_backend(components[0]));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0);pred=[];muscles=[];ood=[];outside=[];timings=[];ages=[];native_values=[];native_valid=[];before=gc.isenabled();gc.disable();started=time.perf_counter()
    try:
        if a.flush_denormal and not torch.set_flush_denormal(True):raise ValueError('Requested numerical mode unsupported')
        runtime.begin_episode(entry['episode_id'])
        for i,t in enumerate(d['decision_time']):
            sample=samples[int(d['body_packet'][i])]
            obs=runtime.decide(decision_time=float(t),scene_features=d['scene'][i],static_features=static[i],body_context=context(d,contract,i),native_sample=sample)
            pred.append(obs['prediction']);muscles.append(obs['muscles']);ood.append(obs['thermal_out_of_training_support']);outside.append(obs['acceleration_out_of_assessed_scope']);timings.append(obs['inference_seconds']);ages.append(obs['estimated_output_body_age_seconds'])
            # Encoder has already admitted the exact same packet/receipt at this decision.
            values,valid,_,_=assessed_extension_values(sample.raw19[None],[sample.pitch_rad],[sample.roll_native_rad],[sample.speed_m_s],components[5].contract)
            native_values.append(values[0]);native_valid.append(valid[0])
            if i%1000==0:write_json(run/'progress.json',dict(event=i,total=len(d['scene']),elapsed_seconds=time.perf_counter()-started))
        pred=np.asarray(pred);mask=d['fit_mask']
        metrics=dict(full_lap=command_metrics(pred,d['labels'],mask),precorner=command_metrics(pred,d['labels'],pre))
        for name,subset in [('speed_below100',d['body'][:,16]*3.6<100),('speed_atleast100',d['body'][:,16]*3.6>=100)]:
            selected=mask&subset
            if selected.any():metrics[name]=dict(count=int(selected.sum()),**command_metrics(pred,d['labels'],selected))
        result.update(status='completed_qualified_native_candidate_evaluation',case=a.case,events=len(pred),full_lap_supervised_labels=int(mask.sum()),precorner_labels=int(pre.sum()),metrics=metrics,thermal_ood_full_lap_counts=np.asarray(ood)[mask].sum(0).tolist(),acceleration_outside_assessed_envelope_full_lap=int(np.asarray(outside)[mask].sum()),inference_seconds_percentiles=np.percentile(timings,[50,95,99,100]).tolist(),output_body_age_max_seconds=max(ages),termination_reason='Full original episode with persistent state, valid original source/body clocks and50ms output gate; qualified full-lap request prediction only, no student physical control or lap qualification.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),events_completed=len(pred),last_rejection=runtime.last_rejection)
    finally:
        runtime.close();torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    np.savez_compressed(run/'predictions.npz',prediction=np.asarray(pred),muscles=np.asarray(muscles),native_values=np.asarray(native_values,dtype=np.float32),native_valid=np.asarray(native_valid,dtype=bool),thermal_ood=np.asarray(ood),acceleration_outside=np.asarray(outside),inference_seconds=np.asarray(timings),output_body_age=np.asarray(ages))
    result.update(predictions_sha256=sha256(run/'predictions.npz'),elapsed_seconds=time.perf_counter()-started);write_json(run/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
