"""Verify the held-input causal delay of a single pre-corner visual feature pulse."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.sac_unified_student import read
from scripts.sac_steering_experiment import assembly,snapshot,restore
from scripts.train_qualified_sac_unified import load_episode,context


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(1);root=Path(a.replay);r=read(root/'result.json');cfg=read(root/'manifest.json')['config']
    assert r['status']=='completed_frozen_precorner_visual_transfer'
    assert sha256(root/'episode-0.npz')==r['comparisons'][0]['predictions_sha256']
    plan=read(cfg['plan']['transfer_plan']);d,c,_=load_episode(cfg['entries'][0]['data'])
    with np.load(root/'episode-0.npz',allow_pickle=False) as z:
        receiver=z['window_0_receiver_indices'];mask=z['window_0_pre_mask'];i0=int(receiver[mask][0])
        donor=int(z['window_0_opposite_indices'][np.flatnonzero(mask)[0]]);expected=z['normal'].copy()
    t=d['decision_time'];ids=np.arange(i0,int(np.searchsorted(t,t[i0]+.2,side='right')))
    model,head,dn,_,_=assembly(plan['candidate']);cp=torch.load(Path(plan['candidate'])/'steering.pt',map_location='cpu',weights_only=True)
    ac=cp['optical'];optical=RetinalResidualPort(ac['reference'],ac['scale'],len(model.motor.backbone.vision_ports));optical.load_state_dict(ac)
    for m in (model,head,dn,optical):m.eval().requires_grad_(False)
    sp=Path(plan['spatial'])/'spatial-0.npz';sr=read(Path(plan['spatial'])/'result.json');assert sha256(sp)==sr['files_sha256'][sp.name]
    with np.load(sp,allow_pickle=False) as z:
        j=z['event_to_camera'];np.testing.assert_array_equal(z['camera_indices'][j],d['camera_index'])
        extra=torch.tensor(np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),z['discarded_grid8'][j]],1),dtype=torch.float32)
    holder=dict(extra=None)
    def descending(m,args,value):
        steer,power=value;return steer+dn(holder['extra']),power
    def spatial(m,args,value):return value+optical(holder['extra'][:,64:])
    handles=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(spatial)]
    scene=torch.tensor(d['scene'],dtype=torch.float32);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
    def step(i,j):
        holder['extra']=extra[j:j+1]
        _,mu=model.sensory_decision(decision_time=float(t[i]),racing_goals=goals,scene_features=scene[j:j+1],scene_valid=views,body_context=context(d,c,i))
        return head(mu)[0].numpy().copy()
    run=begin_run(a.output,dict(replay=str(root),replay_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),seed=None,
        receiver_event=i0,donor_event=donor,pulse_events=1,followup_seconds=.2,
        actor_qualified=False,vehicle_control_sent=False,scope='One known recorded window; held-input timing diagnostic, not visual direction/physical driving proof'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
    try:
        saved={};prefix_error=0.;model.reset_episode()
        with torch.no_grad():
            for i in range(i0):
                pr=step(i,i);prefix_error=max(prefix_error,float(abs(pr-expected[i]).max()))
            assert prefix_error<=2e-6;before=snapshot(model)
            for case in ('normal','pulse'):
                restore(model,before);values=[]
                for k,i in enumerate(ids):
                    values.append(step(int(i),donor if case=='pulse' and k==0 else int(i)))
                    if k==0:
                        saved[case+'_first_state']=model.state.numpy().copy()
                        saved[case+'_first_drive']=model.previous_drive.numpy().copy()
                saved[case]=np.asarray(values)
            assert np.array_equal(saved['normal_first_state'],saved['pulse_first_state'])
            assert np.array_equal(saved['normal'][0],saved['pulse'][0])
            drive_change=float(abs(saved['normal_first_drive']-saved['pulse_first_drive']).max());assert drive_change>0
            replay_error=float(abs(saved['normal']-expected[ids]).max());assert replay_error<=2e-6
            delta=abs(saved['normal'][:,0]-saved['pulse'][:,0]);changed=np.flatnonzero(delta>1e-7)
            assert len(changed) and changed[0]>0
            saved['receiver_indices']=ids;np.savez_compressed(run/'pulse.npz',**saved)
            out.update(status='completed_precorner_visual_pulse_delay_probe',prefix_events=i0,
                normal_replay_max_error=max(prefix_error,replay_error),immediate_state_and_output_equal=True,
                new_held_drive_max_difference=drive_change,first_changed_event_offset=int(changed[0]),
                first_changed_delay_seconds=float(t[ids[changed[0]]]-t[i0]),
                maximum_followup_steering_difference=float(delta.max()),predictions_sha256=sha256(run/'pulse.npz'),
                termination_reason='Same-state/body single visual pulse changes newly held drive, not past-interval state; later output response verified without time relabeling')
    except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit causal timing or replay failure')
    finally:
        for h in handles:h.remove()
    write_json(run/'result.json',out)
    if out['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
