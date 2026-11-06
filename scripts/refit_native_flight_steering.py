"""Refit only six-muscle steering after the frozen native sensory adaptation.

One fixed TRAIN-only command-space fit; no adapter/circuit/temperature learning,
development selection, actuator access, or unseen-road qualification.
"""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.bounded_steering_fit import fit_bounded_steering
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.sac_unified_student import read
from scripts.fit_bounded_sac_steering import metrics, pre_mask
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.sac_steering_experiment import assembly
from scripts.train_qualified_sac_unified import load_episode, context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); plan=read(a.plan); torch.set_num_threads(1)
    assert plan['program_sha256']==sha256(__file__)
    fit_root=Path(plan['native_fit']); fit=read(fit_root/'result.json')
    assert fit['status']=='completed_thermal_native_body_port_fit'
    assert sha256(fit_root/'body-extension.pt')==plan['native_checkpoint_sha256']==fit['checkpoint_sha256']
    assert sha256(fit_root/'predictions.npz')==fit['predictions_sha256']
    audit_root=Path(plan['native_audit']); audit=read(audit_root/'result.json')
    assert audit['status']=='completed_independent_thermal_native_fit_audit'
    assert read(audit_root/'manifest.json')['config']['fit_result_sha256']==sha256(fit_root/'result.json')
    cfg=read(fit_root/'manifest.json')['config']; arguments=cfg['arguments']
    cp=torch.load(fit_root/'body-extension.pt',map_location='cpu',weights_only=True)
    source=Path(cp['source_fit_directory']); assert sha256(source/'steering.pt')==cp['source_checkpoint_sha256']
    parent=torch.load(source/'steering.pt',map_location='cpu',weights_only=True)
    entries=three_episode_cohort(arguments['split'])
    assert sha256(arguments['split'])==cp['cohort_sha256']
    items=[load_episode(e['data']) for e in entries]
    extension=Path(arguments['extension']); er=read(extension/'result.json')
    assert sha256(extension/'contract.json')==cp['extension_contract_sha256']==er['contract_sha256']
    spatial=Path(arguments['spatial']); sr=read(spatial/'result.json')
    model,head,dn,_,lineage=assembly(source)
    ac=parent['optical']; optical=RetinalResidualPort(ac['reference'],ac['scale'],len(model.motor.backbone.vision_ports)); optical.load_state_dict(ac)
    bc=cp['body_port']; port=MultisensoryBodyFeedbackPort(bc['reference'],bc['scale'],len(model.motor.backbone.body_ports)); port.load_state_dict(bc)
    for module in (model,head,dn,optical,port): module.eval().requires_grad_(False)
    extras=[]; native=[]; valid=[]
    for i,(entry,(d,c,provenance)) in enumerate(zip(entries,items)):
        assert provenance['arrays_sha256']==entry['arrays_sha256']
        sp=spatial/f'spatial-{i}.npz'; ep=extension/f'extension-{i}.npz'
        assert sha256(sp)==sr['files_sha256'][sp.name] and sha256(ep)==er['files_sha256'][ep.name]
        with np.load(sp,allow_pickle=False) as z:
            j=z['event_to_camera']; np.testing.assert_array_equal(z['camera_indices'][j],d['camera_index'])
            extras.append(torch.tensor(np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),z['discarded_grid8'][j]],1),dtype=torch.float32))
        with np.load(ep,allow_pickle=False) as z:
            for key in ('body_packet','body_receipt','decision_time'): np.testing.assert_array_equal(z[key],d[key])
            native.append(torch.tensor(z['values'])); valid.append(torch.tensor(z['valid']))
    sign=read(Path(plan['sign_source'])/'manifest.json')['config']['sign_evidence']
    assert sign['training_only'] and sign['agreement']>=.9
    pre=[pre_mask(d,sign['sign']) for d,_,_ in items]
    run=begin_run(a.output,dict(plan=plan,plan_sha256=sha256(a.plan),entries=entries,lineage=lineage,
        sign_evidence=sign,seed=None,program_sha256=sha256(__file__),
        sensory_and_circuit_weights_frozen=True,preweight=16,regularization=.0001,
        choice='Single fixed configuration inherited from previous TRAIN-only bounded-head choice; no new grid or development selection',
        actor_qualified=False,vehicle_control_sent=False))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    shutil.copyfile('src/flybrain_drive/bounded_steering_fit.py',run/'executed-fit.py')
    holder=dict(episode=0,index=0)
    def descending_hook(module,args,value):
        e,i=holder['episode'],holder['index']; steer,power=value
        return steer+dn(extras[e][i:i+1]),power
    def optical_hook(module,args,value):
        e,i=holder['episode'],holder['index']; return value+optical(extras[e][i:i+1,64:])
    def body_hook(module,args,value):
        e,i=holder['episode'],holder['index']
        assert torch.equal(args[0],torch.tensor(items[e][0]['body'][i:i+1],dtype=torch.float32))
        return value+port(native[e][i:i+1],valid[e][i:i+1])
    handles=[model.motor.command.register_forward_hook(descending_hook),model.visual.register_forward_hook(optical_hook),model.body_feedback.register_forward_hook(body_hook)]
    goals=torch.tensor(racing_goal(300,1)); views=torch.ones(1,4,dtype=torch.bool)
    started=time.monotonic(); saved={}; out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
    try:
        with np.load(fit_root/'predictions.npz',allow_pickle=False) as reference, torch.no_grad():
            for e,(d,c,_) in enumerate(items):
                holder['episode']=e; model.reset_episode(); muscles=[]; old=[]
                scene=torch.tensor(d['scene'],dtype=torch.float32)
                for i in range(len(scene)):
                    holder['index']=i
                    _,mu=model.sensory_decision(decision_time=float(d['decision_time'][i]),racing_goals=goals,
                        scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i))
                    muscles.append(mu[0].numpy().copy()); old.append(head(mu)[0].numpy().copy())
                    if i%1000==0: write_json(run/'progress.json',dict(phase='frozen_native_replay',episode=e,event=i,total=len(scene),elapsed_seconds=time.monotonic()-started))
                old=np.asarray(old); error=float(abs(old-reference[f'candidate_{e}']).max())
                if error>2e-6: raise ValueError('Frozen native full replay differs from protected original')
                saved[f'muscles_{e}']=np.asarray(muscles); saved[f'baseline_{e}']=old; saved[f'pre_{e}']=pre[e]
                np.savez_compressed(run/'frozen-trajectories.npz',**saved)
                write_json(run/'completed-replays.json',dict(completed=e+1,last_replay_max_error=error))
        # Only the two TRAIN records enter the fit. Development muscles may be
        # archived above but neither their labels nor their fit score select it.
        features=[((torch.tensor(saved[f'muscles_{e}'])-head.training_reference)/head.training_scale).numpy().astype(float) for e in range(3)]
        masks=[items[e][0]['fit_mask'] for e in (0,1)]
        x=np.concatenate([features[e][masks[e]] for e in (0,1)])
        y=np.concatenate([items[e][0]['labels'][masks[e],0] for e in (0,1)])
        weights=np.concatenate([np.where(pre[e][masks[e]],16.,1.) for e in (0,1)])
        before={k:v.clone() for k,v in head.state_dict().items()}
        fitted=fit_bounded_steering(x,y,weights,before['projection.weight'][0].numpy(),float(before['projection.bias'][0]),.0001)
        write_json(run/'train-fit.json',dict(objective_evaluations=fitted['objective_evaluations'],initial_loss=fitted['initial_loss'],final_loss=fitted['final_loss'],train_only=True,development_used_for_selection=False))
        with torch.no_grad():
            head.projection.weight[0].copy_(torch.tensor(fitted['beta'],dtype=torch.float32)); head.projection.bias[0]=fitted['bias']
        for key in ('projection.weight','projection.bias'): assert torch.equal(before[key][1:],head.state_dict()[key][1:])
        for key in ('training_reference','training_scale'): assert torch.equal(before[key],head.state_dict()[key])
        comparisons=[]
        for e,(d,_,_) in enumerate(items):
            predicted=head(torch.tensor(saved[f'muscles_{e}'])).detach().numpy()
            old=saved[f'baseline_{e}']; np.testing.assert_allclose(predicted[:,1:],old[:,1:],atol=2e-7,rtol=0)
            # Archive steering and untouched original pedal bytes together.
            values=old.copy(); values[:,0]=predicted[:,0]; saved[f'candidate_{e}']=values
            comparisons.append(dict(episode=e,baseline=metrics(old[:,0],d,pre[e]),candidate=metrics(values[:,0],d,pre[e]),pedals_exact=bool(np.array_equal(values[:,1:],old[:,1:]))))
        checkpoint=run/'native-steering.pt'
        torch.save(dict(schema='flybrain.native_flight_steering_refit.v1',native_checkpoint_sha256=plan['native_checkpoint_sha256'],
            native_fit_directory=str(fit_root),readout=head.state_dict(),body_port=cp['body_port'],
            source_checkpoint_sha256=cp['source_checkpoint_sha256'],actor_qualified=False,head_fit_train_only=True),checkpoint)
        torch.load(checkpoint,map_location='cpu',weights_only=True)
        np.savez_compressed(run/'predictions.npz',**saved)
        dev=comparisons[2]
        passed=all(dev['candidate'][key]<=.98*dev['baseline'][key] for key in ('all_rmse','pre_rmse'))
        out.update(status='completed_native_flight_steering_refit',comparisons=comparisons,
            development_two_percent_gate_passed=passed,objective_evaluations=fitted['objective_evaluations'],
            train_initial_loss=fitted['initial_loss'],train_final_loss=fitted['final_loss'],
            checkpoint_sha256=sha256(checkpoint),predictions_sha256=sha256(run/'predictions.npz'),
            frozen_trajectories_sha256=sha256(run/'frozen-trajectories.npz'),
            termination_reason='Frozen sensory/circuit replay verified; single TRAIN-only six-muscle steering fit completed; consumed development diagnostics, no live/generalization qualification')
    except Exception as error:
        out.update(error=type(error).__name__,reason=str(error),termination_reason='Explicit replay/fit/invariant failure; no deployment')
    finally:
        for h in handles:h.remove()
    out['elapsed_seconds']=time.monotonic()-started;write_json(run/'result.json',out)
    if out['status']=='failed': raise SystemExit(1)


if __name__=='__main__':main()
