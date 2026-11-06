"""Learn native19 only at existing flight body ports; all old circuit/head frozen.

Whole recorded episodes, persistent neural state, TRAIN-only normalization.
Temperature-neutral replay is an information intervention, not simulated physics.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.sac_unified_student import detach_recurrence,read
from scripts.improve_sac_anticipation import command_metrics
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.sac_steering_experiment import assembly
from scripts.train_qualified_sac_unified import load_episode,context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['split','spatial','extension','source','plan','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(780514);np.random.seed(780514)
    plan=read(a.plan)
    if (plan['program_sha256']!=sha256(__file__) or plan['cohort_sha256']!=sha256(a.split)
            or plan['source_checkpoint_sha256']!=sha256(Path(a.source)/'steering.pt')):
        raise ValueError('Sealed program/cohort/protected source differs')
    entries=three_episode_cohort(a.split);items=[load_episode(e['data']) for e in entries]
    extension=Path(a.extension);er=read(extension/'result.json');sr=read(Path(a.spatial)/'result.json')
    audit_path=Path(plan['extension_audit_result']);audit=read(audit_path)
    audit_config=read(audit_path.parent/'manifest.json')['config']
    if (sha256(audit_path)!=plan['extension_audit_sha256']
            or audit['status']!='completed_native_extension_streaming_audit'
            or audit_config['extension_result_sha256']!=sha256(extension/'result.json')
            or audit_config['entries']!=entries):
        raise ValueError('Completed independent native-byte audit for this exact extension/cohort required')
    if (er['status']!='completed_assessed_native_multisensory_preparation'
            or er.get('eligible_for_experimental_offline_component_fit') is not True
            or sr['status']!='completed_qualified_static_spatial_recovery'):
        raise ValueError('Completed qualified static/assessed native sources required')
    if sha256(extension/'contract.json')!=er['contract_sha256']:raise ValueError('Native contract changed')
    contract=read(extension/'contract.json')
    NativeMultisensoryExtensionEncoder(extension/'contract.json',native_identity=contract['native_identity'])
    if sha256(extension/'training-normalization.npz')!=er['files_sha256']['training-normalization.npz']:
        raise ValueError('TRAIN-only normalization changed')
    with np.load(extension/'training-normalization.npz') as z:
        reference=torch.tensor(z['reference']);scale=torch.tensor(z['scale'])
    extras=[];sensations=[];validities=[]
    for i,(e,(d,c,provenance)) in enumerate(zip(entries,items)):
        if provenance['arrays_sha256']!=e['arrays_sha256']:raise ValueError('Prepared cohort arrays changed')
        sf=Path(a.spatial)/f'spatial-{i}.npz';ef=extension/f'extension-{i}.npz'
        if sha256(sf)!=sr['files_sha256'][sf.name] or sha256(ef)!=er['files_sha256'][ef.name]:
            raise ValueError('Static/native sidecar changed')
        with np.load(sf) as z:
            j=z['event_to_camera'];np.testing.assert_array_equal(z['camera_indices'][j],d['camera_index'])
            extras.append(torch.tensor(np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),z['discarded_grid8'][j]],1),dtype=torch.float32))
        with np.load(ef) as z:
            for k in ['body_packet','body_receipt','decision_time']:np.testing.assert_array_equal(z[k],d[k])
            sensations.append(torch.tensor(z['values']));validities.append(torch.tensor(z['valid']))
    model,head,dn,base,lineage=assembly(a.source)
    cp=torch.load(Path(a.source)/'steering.pt',map_location='cpu',weights_only=True);ac=cp['optical']
    optical=RetinalResidualPort(ac['reference'],ac['scale'],len(model.motor.backbone.vision_ports));optical.load_state_dict(ac)
    for module in [model,head,dn,optical]:module.eval().requires_grad_(False)
    port=MultisensoryBodyFeedbackPort(reference,scale,len(model.motor.backbone.body_ports))
    optimizer=torch.optim.Adam(port.parameters(),lr=.0003)
    run=begin_run(a.output,dict(arguments=vars(a),seed=780514,program_sha256=sha256(__file__),
        cohort_sha256=sha256(a.split),fit_plan_sha256=sha256(a.plan),entries=entries,lineage=lineage,
        source_checkpoint_sha256=sha256(Path(a.source)/'steering.pt'),extension_contract_sha256=er['contract_sha256'],
        spatial_result_sha256=sha256(Path(a.spatial)/'result.json'),trainable_parameters=sum(p.numel() for p in port.parameters()),
        fixed_epochs=1,tbptt_events=32,loss_weights=[4,1,1],adam_lr=.0003,
        old_model_state_entries_frozen=51,all_old_adapters_and_readout_frozen=True,
        scope=__doc__,actor_deployed=False,vehicle_control_sent=False))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    holder=dict(episode=0,index=0,case='normal');maximum_delta=0.;updates=0;start=time.monotonic()
    def dn_hook(m,args,value):
        steer,power=value;i=holder['index'];e=holder['episode']
        return steer+dn(extras[e][i:i+1]),power
    def optical_hook(m,args,value):
        i=holder['index'];return value+optical(extras[holder['episode']][i:i+1,64:])
    def body_hook(m,args,value):
        nonlocal maximum_delta
        i=holder['index'];e=holder['episode'];case=holder['case']
        expected=torch.tensor(items[e][0]['body'][i:i+1],dtype=torch.float32)
        if not torch.equal(args[0],expected):raise ValueError('Native port joined to different current body event')
        if case=='native_neutral':return value
        x=sensations[e][i:i+1]
        if case=='temperature_neutral':
            x=x.clone();x[:,11:15]=reference[11:15]
        delta=port(x,validities[e][i:i+1]);maximum_delta=max(maximum_delta,float(delta.detach().abs().max()))
        return value+delta
    handles=[model.motor.command.register_forward_hook(dn_hook),model.visual.register_forward_hook(optical_hook),model.body_feedback.register_forward_hook(body_hook)]
    goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
    def rollout(episode,*,learn=False,case='normal',limit=None):
        nonlocal updates
        holder.update(episode=episode,case=case);d,c,_=items[episode]
        scenes=torch.tensor(d['scene'],dtype=torch.float32);labels=torch.tensor(d['labels'],dtype=torch.float32)
        first,last=np.flatnonzero(d['fit_mask'])[[0,-1]];model.reset_episode();pred=[];losses=[]
        for i in range(len(scenes) if limit is None else min(limit,len(scenes))):
            holder['index']=i;learning=bool(learn and first<=i<=last)
            with torch.set_grad_enabled(learning):
                fn=model.sensory_training_decision if learning else model.sensory_decision
                _,mu=fn(decision_time=float(d['decision_time'][i]),racing_goals=goals,
                    scene_features=scenes[i:i+1],scene_valid=views,body_context=context(d,c,i))
                pr=head(mu);pred.append(pr[0].detach().numpy().copy())
                if learning and d['fit_mask'][i]:losses.append(((pr[0]-labels[i]).square()*pr.new_tensor([4,1,1])).mean())
                if learn and ((i+1)%32==0 or i==len(scenes)-1):
                    if losses:
                        optimizer.zero_grad(set_to_none=True);torch.stack(losses).mean().backward()
                        torch.nn.utils.clip_grad_norm_(port.parameters(),1.)
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in port.parameters()):
                            raise ValueError('Missing/nonfinite new native-port gradient')
                        optimizer.step();updates+=1;losses.clear()
                    detach_recurrence(model)
            if i%1000==0:write_json(run/'progress.json',dict(episode=episode,event=i,total=len(scenes),
                phase='training' if learn else case,optimizer_updates=updates,elapsed_seconds=time.monotonic()-start))
        return np.asarray(pred)
    out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0);saved={}
    try:
        # Probe the real recurrent path before expensive whole-episode replay.
        # No labels or optimizer steps: test reachability of the six-muscle
        # readout gradient, then discard the diagnostic numerical state.
        holder.update(episode=2,case='normal');model.reset_episode()
        d,c,_=items[2];scenes=torch.tensor(d['scene'],dtype=torch.float32)
        with torch.enable_grad():
            for i in range(64):
                holder['index']=i
                _,mu=model.sensory_training_decision(decision_time=float(d['decision_time'][i]),
                    racing_goals=goals,scene_features=scenes[i:i+1],scene_valid=views,body_context=context(d,c,i))
            head(mu).sum().backward()
        grad=port.projection.weight.grad
        if grad is None or not torch.isfinite(grad).all() or not (grad.abs()>0).any():
            raise ValueError('No finite nonzero recurrent readout gradient to native sensory boundary')
        write_json(run/'gradient-smoke.json',dict(events=64,labels_used=0,optimizer_updates=0,
            gradient_max_abs=float(grad.abs().max()),gradient_nonzero_entries=int((grad!=0).sum()),
            old_weights_requires_grad=False,scope='Numerical gradient-path check, not physiology or driving'))
        optimizer.zero_grad(set_to_none=True)
        baseline=[]
        for i in range(3):
            pr=rollout(i,case='native_neutral');saved[f'baseline_{i}']=pr
            baseline.append(command_metrics(pr,items[i][0]['labels'],items[i][0]['fit_mask']))
            np.savez_compressed(run/'baseline-predictions.npz',**saved)
            write_json(run/'baseline-metrics.json',dict(completed_episodes=i+1,metrics=baseline,
                optimizer_updates=0,source_checkpoint_sha256=sha256(Path(a.source)/'steering.pt')))
        # Exact zero neutrality is checked on a continuous prefix before learning.
        # Full baseline and intervention episodes preserve all numerical states.
        prefix=rollout(2,limit=64)
        zero_error=float(abs(prefix-saved['baseline_2'][:64]).max())
        if maximum_delta!=0 or zero_error>2e-6:raise ValueError('Zero initialization changed source prefix')
        with torch.no_grad():
            generator=torch.Generator().manual_seed(780514)
            port.projection.weight.copy_(torch.randn(port.projection.weight.shape,generator=generator)*1e-4)
        stimulated=rollout(2,limit=64)
        stimulus_response=float(abs(stimulated-prefix).max())
        if not np.isfinite(stimulated).all() or stimulus_response<=1e-10:
            raise ValueError('Nonzero native sensory port did not reach the muscle/vehicle readout')
        with torch.no_grad():port.projection.weight.zero_()
        write_json(run/'nonzero-port-smoke.json',dict(scope='Engineering current stimulus, not physiology or driving',
            same_episode_prefix_events=64,source_prediction_max_difference=stimulus_response,
            stimulus_weight_std=1e-4,optimizer_updates=0,restored_zero_weights=True))
        for i in [0,1]:rollout(i,learn=True)
        if updates==0 or float(port.projection.weight.detach().abs().max())==0:
            raise ValueError('New sensory boundary did not actually learn')
        candidate=[]
        for i in range(3):
            pr=rollout(i);saved[f'candidate_{i}']=pr
            candidate.append(command_metrics(pr,items[i][0]['labels'],items[i][0]['fit_mask']))
        pr=rollout(2,case='temperature_neutral');saved['development_temperature_neutral']=pr
        temperature=command_metrics(pr,items[2][0]['labels'],items[2][0]['fit_mask'])
        # A second complete baseline replay verifies hooks/state resets and
        # immutable inherited modules, rather than assuming a zero-delta effect.
        neutral=rollout(2,case='native_neutral');saved['development_native_neutral']=neutral
        error=float(abs(neutral-saved['baseline_2']).max())
        if error>2e-6:raise ValueError('Neutral boundary no longer reproduces protected source')
        for k,v in model.state_dict().items():
            original=base['model'][k]
            if v.layout!=torch.strided:
                x,y=v.to_sparse_coo().coalesce(),original.to_sparse_coo().coalesce()
                same=torch.equal(x.indices(),y.indices()) and torch.equal(x.values(),y.values())
            else:same=torch.equal(v,original)
            if not same:raise ValueError('Protected model state changed: '+k)
        for module,key in [(head,'readout'),(dn,'adapter'),(optical,'optical')]:
            for k,v in module.state_dict().items():
                if not torch.equal(v,cp[key][k]):raise ValueError('Protected inherited component changed: '+key+'/'+k)
        checkpoint=run/'body-extension.pt'
        torch.save(dict(schema='flybrain.thermal_native_body_port.v1',body_port=port.state_dict(),
            source_fit_directory=a.source,source_checkpoint_sha256=sha256(Path(a.source)/'steering.pt'),
            cohort_sha256=sha256(a.split),extension_contract_sha256=er['contract_sha256'],
            trained=True,optimizer_updates=updates,actor_qualified=False),checkpoint)
        torch.load(checkpoint,map_location='cpu',weights_only=True)
        np.savez_compressed(run/'predictions.npz',**saved)
        out.update(status='completed_thermal_native_body_port_fit',optimizer_updates=updates,trained=True,
            baseline=baseline,candidate=candidate,development_temperature_neutral=temperature,
            development_native_neutral=command_metrics(neutral,items[2][0]['labels'],items[2][0]['fit_mask']),
            neutral_source_max_difference=error,maximum_body_port_abs_delta=maximum_delta,
            initial_zero_port_prefix_max_difference=zero_error,
            nonzero_body_port_readout_response=stimulus_response,
            original_model_state_entries_exact=51,old_optical_dn_and_all_readout_rows_exact=True,
            checkpoint_sha256=sha256(checkpoint),predictions_sha256=sha256(run/'predictions.npz'),
            termination_reason='One fixed TRAIN-only native sensory fit and complete continuous baseline/candidate/information-intervention replay; no new-condition held-out/live/physiology claim')
    except Exception as e:
        out.update(error=type(e).__name__,reason=str(e),optimizer_updates=updates,termination_reason='Explicit fit/invariant failure; no deployment')
        if saved:
            np.savez_compressed(run/'partial-predictions.npz',**saved)
            out['partial_predictions_sha256']=sha256(run/'partial-predictions.npz')
        torch.save(dict(body_port=port.state_dict(),optimizer_updates=updates,
            partial=True,fit_completed=False,actor_qualified=False),run/'partial-port-state.pt')
    finally:
        for h in handles:h.remove()
    out['elapsed_seconds']=time.monotonic()-start;write_json(run/'result.json',out);print(json.dumps(out))
    if out['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
