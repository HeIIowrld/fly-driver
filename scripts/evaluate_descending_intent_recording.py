"""Original qualified recording preservation check for a functional overlay."""
import argparse,base64,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.functional_motion_overlay import load_functional_motion_overlay
from flybrain_drive.descending_intent_recalibration import load_descending_intent
from flybrain_drive.graph import sha256
from flybrain_drive.motion_spatial_runtime import MotionSpatialDecisionRuntime,MotionSpatialObservation
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from scripts.improve_sac_anticipation import command_metrics
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.train_qualified_sac_unified import context


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--episode',type=int,choices=[0,1,2],required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);checkpoint=Path(a.checkpoint);fm=read(checkpoint.parent/'manifest.json')['config'];fitplan=fm['plan'];stimulus=read(Path(fitplan['training_stimuli'])/'manifest.json')['config']['plan'];candidate=read(stimulus['candidate_plan']);components,motion,parent=load_readonly_motion_candidate(candidate);schema=torch.load(checkpoint,map_location='cpu',weights_only=True)['schema'];assert schema=='flybrain.descending_intent.v1';receipt=load_descending_intent(checkpoint,components,motion,parent);original=Path(candidate['motion']['fit']);om=read(original/'manifest.json')['config'];_,_,entries,items,extras,_,_,_,pre,features,maps,coarse,_,_=load_motion_experiment(om['plan'],'fine_motion');e=a.episode;d,c,_=items[e]
    with np.load(original/'predictions.npz') as z:baseline=z[f'prediction_{e}'].copy()
    cfg=read(Path(entries[e]['data'])/'manifest.json')['config'];native=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';qr=read(Path(cfg['arguments']['qualification'])/'result.json')
    if sha256(native)!=qr['native_body_sha256']:raise ValueError('Original qualified native body source changed')
    samples={};needed=set(map(int,d['body_packet']))
    for line in native.open(encoding='utf8'):
        auth=json.loads(line)['authority'];pid=auth['packet_id']
        if pid not in needed or pid in samples:continue
        raw=base64.b64decode(auth['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('Original native prefix bytes differ')
        samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
    if set(samples)!=needed:raise ValueError('Complete original packet coverage required')
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),functional_fit_result_sha256=sha256(checkpoint.parent/'result.json'),candidate=receipt,entry=entries[e],original_motion_fit_result_sha256=sha256(original/'result.json'),preservation_maximum_ratio=fitplan['actual_recording_preservation_maximum_ratio'],seed=None,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope='Consumed original same-course development replay, not new-track selection or physical driving improvement'));shutil.copyfile(__file__,out/'executed-entrypoint.py');runtime=None;before=gc.isenabled();gc.disable();prediction=[];muscles=[];ages=[];result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False);start=time.perf_counter()
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Fixed CPU numerical mode unavailable')
        runtime=MotionSpatialDecisionRuntime(*components,motion_port=motion);write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(components[0]));runtime.begin_episode(c['episode_id'])
        for i in range(len(d['scene'])):
            j=int(maps[e][i]);observation=MotionSpatialObservation(c['episode_id'],float(d['camera_receipt'][i]),tuple(map(float,d['render_lowers'][i])),coarse[e][j],features[e][j].numpy());output=runtime.decide(decision_time=float(d['decision_time'][i]),scene_features=d['scene'][i],static_features=extras[e][i].numpy(),body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])],motion_observation=observation);prediction.append(output['prediction']);muscles.append(output['muscles']);ages.append(output['estimated_output_body_age_seconds'])
            if i%1000==0:write_json(out/'progress.json',dict(event=i,total=len(d['scene']),elapsed_seconds=time.perf_counter()-start))
        prediction=np.asarray(prediction);metrics={}
        for name,mask in [('all',d['fit_mask']),('pre',pre[e])]:metrics[name]=dict(parent=command_metrics(baseline,d['labels'],mask),functional=command_metrics(prediction,d['labels'],mask))
        bound=fitplan['actual_recording_preservation_maximum_ratio'];passed=all(metrics['all']['functional']['rmse'][k]<=bound*metrics['all']['parent']['rmse'][k] for k in range(3)) and metrics['pre']['functional']['rmse'][0]<=bound*metrics['pre']['parent']['rmse'][0];np.savez_compressed(out/'predictions.npz',prediction=prediction,muscles=np.asarray(muscles));result.update(status='completed_functional_motion_recording_preservation',episode=e,events=len(prediction),metrics=metrics,preservation_gate_passed=bool(passed),output_body_age_max_seconds=max(ages),predictions_sha256=sha256(out/'predictions.npz'),termination_reason='Original full native-byte/camera recurrence replay for synthetic functional overlay; comparison to parent is consumed same-course preservation, not autonomous or unseen-corner qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),events=len(prediction))
    finally:
        if runtime is not None:runtime.close()
        pass # no temporal attachment in this experiment
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
