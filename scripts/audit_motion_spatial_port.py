"""Independent frozen-state/muscle/metric and typed motion-runtime replay audit."""
import argparse,base64,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.motion_spatial_residual_port import MotionSpatialResidualPort
from flybrain_drive.motion_spatial_runtime import MotionSpatialDecisionRuntime,MotionSpatialObservation
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.train_qualified_sac_unified import context


def equal(left,right):return set(left)==set(right) and all(v.dtype==right[k].dtype and v.shape==right[k].shape and torch.equal(v,right[k]) for k,v in left.items())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--full-runtime-replay',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.fit);r=read(root/'result.json');m=read(root/'manifest.json')['config'];plan=m['plan'];arm=m['arguments']['arm']
    (components,receipt,entries,items,extras,native,valid,baseline,pre,features,maps,coarse,reference,scale)=load_motion_experiment(plan,arm)
    if r['status']!='completed_motion_spatial_port_fit' or sha256(root/'motion-spatial.pt')!=r['checkpoint_sha256'] or sha256(root/'predictions.npz')!=r['predictions_sha256']:raise ValueError('Completed original motion fit required')
    cp=torch.load(root/'motion-spatial.pt',map_location='cpu',weights_only=True)
    if cp['schema']!='flybrain.motion_spatial_port.v1' or cp['arm']!=arm or cp['parent_candidate']!=receipt or cp['actor_qualified'] is not False or cp['maximum_logit_delta']!=plan['maximum_logit_delta']:raise ValueError('Motion checkpoint lineage/bound differs')
    for key,index in [('readout',1),('adapter',2),('optical',3),('body_port',4)]:
        if not equal(cp[key],components[index].state_dict()):raise ValueError('Original frozen component changed '+key)
    port=MotionSpatialResidualPort(reference,scale,len(components[0].motor.backbone.vision_ports),maximum_logit_delta=plan['maximum_logit_delta'])
    expected_state=port.state_dict()
    if set(cp['motion_port'])!=set(expected_state):raise ValueError('Motion state schema changed')
    for key,value in cp['motion_port'].items():
        if value.dtype!=expected_state[key].dtype or value.shape!=expected_state[key].shape or not torch.isfinite(value).all():raise ValueError('Motion state dtype/dimensions/value changed')
    expected_updates=sum(len(np.unique(np.flatnonzero(items[e][0]['fit_mask'])//plan['window'])) for e in (0,1))
    if r['optimizer_updates']!=expected_updates or cp['optimizer_updates']!=expected_updates:raise ValueError('Original supervision/window update count differs')
    for key in ['reference','scale']:
        if not torch.equal(cp['motion_port'][key],port.state_dict()[key]):raise ValueError('TRAIN-only motion normalization changed')
    delta=float(cp['motion_port']['projection.weight'].abs().max());assert 0<delta<=plan['maximum_weight_delta']+1e-7
    state={k:v.numpy().astype(float) for k,v in components[1].state_dict().items()};predictions={};reports=[]
    with np.load(root/'predictions.npz') as z:
        for e,(d,c,_) in enumerate(items):
            pred=z[f'prediction_{e}'];mu=z[f'muscles_{e}'];assert pred.shape==(len(d['labels']),3) and mu.shape==(len(pred),6);predictions[e]=pred.copy();logits=((mu-state['training_reference'])/state['training_scale'])@state['projection.weight'].T+state['projection.bias'];expected=np.column_stack([np.tanh(logits[:,0]),1/(1+np.exp(-logits[:,1])),1/(1+np.exp(-logits[:,2]))]);error=float(abs(expected-pred).max());assert error<=2e-6
            for name,mask,value in [('candidate',d['fit_mask'],pred),('candidate_pre',pre[e],pred),('baseline',d['fit_mask'],baseline[e]),('baseline_pre',pre[e],baseline[e])]:
                difference=value[mask].astype(float)-d['labels'][mask].astype(float);metric=r['comparisons'][e][name];np.testing.assert_allclose(np.sqrt(np.mean(difference**2,axis=0)),metric['rmse'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.mean(abs(difference),axis=0),metric['mae'],atol=1e-12,rtol=0);assert abs(float(np.mean(difference**2*[4,1,1]))-metric['weighted_mse'])<1e-12
            reports.append(dict(episode=e,events=len(pred),qualified_labels=int(d['fit_mask'].sum()),pre_labels=int(pre[e].sum()),independent_muscle_readout_max_difference=error))
    dev=r['comparisons'][2];gate=dev['candidate']['rmse'][0]<=.98*dev['baseline']['rmse'][0] and dev['candidate_pre']['rmse'][0]<=.98*dev['baseline_pre']['rmse'][0] and all(dev['candidate']['rmse'][k]<=1.02*dev['baseline']['rmse'][k] for k in (1,2))
    if bool(gate)!=r['development_two_percent_gate_passed']:raise ValueError('Development gate differs')
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256=sha256(root/'result.json'),fit_checkpoint_sha256=sha256(root/'motion-spatial.pt'),runtime_sha256=sha256('src/flybrain_drive/motion_spatial_runtime.py'),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');runtime=None;replays=[];before=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
    try:
        if a.full_runtime_replay:
            if not torch.set_flush_denormal(True):raise ValueError('Original numerical mode unsupported')
            runtime=MotionSpatialDecisionRuntime(*components,motion_port=port);write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(components[0]))
            for e,(d,c,_) in enumerate(items):
                cfg=read(Path(entries[e]['data'])/'manifest.json')['config'];source=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';cert=read(Path(cfg['arguments']['qualification'])/'result.json');assert sha256(source)==cert['native_body_sha256'];samples={};needed=set(map(int,d['body_packet']))
                for line in source.open(encoding='utf-8'):
                    auth=json.loads(line)['authority'];pid=auth['packet_id']
                    if pid not in needed or pid in samples:continue
                    raw=base64.b64decode(auth['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==auth['prefix_sha256'];samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
                if set(samples)!=needed:raise ValueError('Original body packet missing')
                def step(i):
                    j=int(maps[e][i]);obs=MotionSpatialObservation(c['episode_id'],float(d['camera_receipt'][i]),tuple(map(float,d['render_lowers'][i])),coarse[e][j],features[e][j].numpy())
                    return runtime.decide(decision_time=float(d['decision_time'][i]),scene_features=d['scene'][i],static_features=extras[e][i].numpy(),body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])],motion_observation=obs)
                with torch.no_grad():port.projection.weight.zero_()
                runtime.begin_episode(entries[e]['episode_id']);zero_error=0.
                for i in range(96):zero_error=max(zero_error,float(abs(step(i)['prediction']-baseline[e][i]).max()))
                if zero_error>2e-6:raise ValueError('Zero motion module did not preserve original prefix')
                port.load_state_dict(cp['motion_port'],strict=True);runtime.begin_episode(entries[e]['episode_id']);maximum=0.;ages=[];timings=[];start=time.perf_counter()
                for i in range(len(d['scene'])):
                    output=step(i);difference=float(abs(output['prediction']-predictions[e][i]).max());maximum=max(maximum,difference)
                    if difference>2e-6:raise ValueError('Typed motion runtime differs from stored neural trajectory')
                    ages.append(output['estimated_output_body_age_seconds']);timings.append(output['inference_seconds'])
                    if i%1000==0:write_json(out/'progress.json',dict(episode=e,event=i,total=len(d['scene']),full_replay_max_difference=maximum,elapsed_seconds=time.perf_counter()-start))
                replays.append(dict(episode=e,events=len(d['scene']),zero_port_prefix_max_difference=zero_error,full_replay_max_difference=maximum,output_body_age_max_seconds=max(ages),inference_seconds_percentiles=np.percentile(timings,[50,95,99,100]).tolist()));write_json(out/'completed-runtime-episodes.json',replays)
        result.update(status='completed_independent_motion_spatial_port_audit',arm=arm,episodes=reports,maximum_weight_delta=delta,full_runtime_requested=a.full_runtime_replay,full_runtime_episodes=replays,optimizer_updates=0,termination_reason='Original components/normalization frozen, independent muscle formulas/masks/metrics and requested full typed same-camera/original-byte runtime replays checked. Zero module prefix preserves parent. No live actor connection, new-corner or student lap qualification.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),completed_runtime_episodes=replays)
    finally:
        if runtime is not None:runtime.close()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
