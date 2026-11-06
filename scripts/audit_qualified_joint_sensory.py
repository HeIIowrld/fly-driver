"""Independent frozen-state, readout/metrics and optional full runtime replay audit."""
import argparse,base64,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from scripts.new_condition_joint_experiment import load_joint_experiment
from scripts.train_qualified_sac_unified import context


def exact(left,right):
    return set(left)==set(right) and all(v.dtype==right[k].dtype and v.shape==right[k].shape and torch.equal(v,right[k]) for k,v in left.items())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--full-runtime-replay',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    root=Path(a.fit);r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];plan=cfg['plan'];components,receipt,entries,items,extras,native,valid,baseline,pre=load_joint_experiment(plan)
    if r['status']!='completed_qualified_joint_sensory_fit' or sha256(root/'joint-sensory.pt')!=r['checkpoint_sha256'] or sha256(root/'predictions.npz')!=r['predictions_sha256']:raise ValueError('Completed original joint sensory fit required')
    cp=torch.load(root/'joint-sensory.pt',map_location='cpu',weights_only=True)
    if cp['schema']!='flybrain.qualified_joint_sensory.v1' or cp['actor_qualified'] is not False or cp['parent_candidate_plan_sha256']!=plan['candidate_plan_sha256'] or not exact(cp['readout'],components[1].state_dict()) or not exact(cp['adapter'],components[2].state_dict()):raise ValueError('Frozen readout/DN or parent changed')
    deltas={}
    for key,module in [('optical',components[3]),('body_port',components[4])]:
        old=module.state_dict();new=cp[key]
        if set(old)!=set(new):raise ValueError('Sensory state schema changed')
        for name,value in new.items():
            if value.dtype!=old[name].dtype or value.shape!=old[name].shape or not torch.isfinite(value).all():raise ValueError('Sensory state dimensions/dtype/values changed')
            if name!='projection.weight' and not torch.equal(value,old[name]):raise ValueError('Frozen normalization changed')
        delta=float(abs(new['projection.weight']-old['projection.weight']).max())
        if not 0<delta<=plan['maximum_weight_delta']+1e-7:raise ValueError('Bounded actual sensory update required')
        deltas[key]=delta;module.load_state_dict(new,strict=True)
    sd={k:v.numpy().astype(float) for k,v in components[1].state_dict().items()};reports=[];predictions={}
    with np.load(root/'predictions.npz') as z:
        for e,(d,c,_) in enumerate(items):
            pred=z[f'prediction_{e}'];mus=z[f'muscles_{e}'];predictions[e]=pred.copy()
            if pred.shape!=(len(d['labels']),3) or mus.shape!=(len(pred),6):raise ValueError('Full trajectory arrays required')
            logits=((mus-sd['training_reference'])/sd['training_scale'])@sd['projection.weight'].T+sd['projection.bias'];expected=np.column_stack([np.tanh(logits[:,0]),1/(1+np.exp(-logits[:,1])),1/(1+np.exp(-logits[:,2]))]);readout_error=float(abs(pred-expected).max());assert readout_error<=2e-6
            for key,mask,values in [('candidate',d['fit_mask'],pred),('candidate_pre',pre[e],pred),('baseline',d['fit_mask'],baseline[e]),('baseline_pre',pre[e],baseline[e])]:
                error=values[mask].astype(float)-d['labels'][mask].astype(float);metric=r['comparisons'][e][key]
                np.testing.assert_allclose(np.sqrt(np.mean(error**2,axis=0)),metric['rmse'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.mean(abs(error),axis=0),metric['mae'],atol=1e-12,rtol=0);assert abs(float(np.mean(error**2*[4,1,1]))-metric['weighted_mse'])<1e-12
            reports.append(dict(episode=e,events=len(pred),full_lap_labels=int(d['fit_mask'].sum()),pre_labels=int(pre[e].sum()),independent_muscle_readout_max_difference=readout_error))
    dev=r['comparisons'][2];gate=dev['candidate']['rmse'][0]<=.98*dev['baseline']['rmse'][0] and dev['candidate_pre']['rmse'][0]<=.98*dev['baseline_pre']['rmse'][0] and all(dev['candidate']['rmse'][k]<=1.02*dev['baseline']['rmse'][k] for k in (1,2));assert bool(gate)==r['development_two_percent_gate_passed']
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256=sha256(root/'result.json'),fit_checkpoint_sha256=sha256(root/'joint-sensory.pt'),runtime_sha256=sha256('src/flybrain_drive/native_flight_decision_runtime.py'),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));shutil.copyfile(__file__,run/'executed-entrypoint.py')
    runtime=None;runtime_reports=[];before=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
    try:
        if a.full_runtime_replay:
            if not torch.set_flush_denormal(True):raise ValueError('Original numerical mode unsupported')
            runtime=NativeFlightDecisionRuntime(*components);write_json(run/'backend-initialization.json',prepare_fixed_cpu_backend(components[0]))
            for e,(d,c,_) in enumerate(items):
                acfg=read(Path(entries[e]['data'])/'manifest.json')['config'];source=Path(acfg['arguments']['recording'])/'camera/native-body.jsonl';certificate=read(Path(acfg['arguments']['qualification'])/'result.json');assert sha256(source)==certificate['native_body_sha256'];needed=set(map(int,d['body_packet']));samples={}
                for line in source.open(encoding='utf-8'):
                    row=json.loads(line)['authority'];pid=row['packet_id']
                    if pid not in needed or pid in samples:continue
                    raw=base64.b64decode(row['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['prefix_sha256'];samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],row['received_monotonic_seconds'])
                if set(samples)!=needed:raise ValueError('Native packet missing')
                runtime.begin_episode(entries[e]['episode_id']);maximum=0.;timings=[];ages=[];start=time.perf_counter()
                for i,t in enumerate(d['decision_time']):
                    output=runtime.decide(decision_time=float(t),scene_features=d['scene'][i],static_features=extras[e][i].numpy(),body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])]);difference=float(abs(output['prediction']-predictions[e][i]).max());maximum=max(maximum,difference)
                    if difference>2e-6:raise ValueError('Full original neural trajectory not reproduced')
                    timings.append(output['inference_seconds']);ages.append(output['estimated_output_body_age_seconds'])
                    if i%1000==0:write_json(run/'progress.json',dict(episode=e,event=i,total=len(d['scene']),full_replay_max_difference=maximum,elapsed_seconds=time.perf_counter()-start))
                runtime_reports.append(dict(episode=e,events=len(d['scene']),full_replay_max_difference=maximum,output_body_age_max_seconds=max(ages),inference_seconds_percentiles=np.percentile(timings,[50,95,99,100]).tolist()));write_json(run/'completed-runtime-episodes.json',runtime_reports)
        result.update(status='completed_independent_qualified_joint_sensory_audit',episodes=reports,projection_deltas=deltas,development_two_percent_gate_passed=bool(gate),full_runtime_replay_requested=a.full_runtime_replay,full_runtime_episodes=runtime_reports,optimizer_updates=0,termination_reason='Frozen readout/DN/normalizations, bounded learned projections, all muscle formulas/qualified metrics and requested original-byte full runtime trajectories verified; no actor promotion, physical adaptation or student lap claim.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),completed_runtime_episodes=runtime_reports)
    finally:
        if runtime is not None:runtime.close()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
