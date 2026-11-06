"""Evaluate a previously selected steering head on new episodes without fitting."""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.sac_unified_student import MuscleVehicleReadout, read
from flybrain_drive.sustained_turn_diagnostic import steering_yaw_alignment
from scripts.fit_bounded_sac_steering import pre_mask, metrics
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.sac_steering_experiment import assembly
from scripts.train_qualified_sac_unified import load_episode, context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); plan = read(a.plan); torch.set_num_threads(1)
    if plan['program_sha256'] != sha256(__file__):
        raise ValueError('Sealed program differs')
    for key in ['source', 'candidate']:
        if sha256(Path(plan[key])/'steering.pt') != plan[key+'_sha256']:
            raise ValueError('Previously selected checkpoint changed')
    if sha256(plan['split']) != plan['split_sha256']:
        raise ValueError('New episode cohort changed')
    entries = three_episode_cohort(plan['split'])
    items = [load_episode(e['data']) for e in entries]
    source = torch.load(Path(plan['source'])/'steering.pt', map_location='cpu', weights_only=True)
    candidate = torch.load(Path(plan['candidate'])/'steering.pt', map_location='cpu', weights_only=True)
    for key in ['adapter', 'optical']:
        for k, v in source[key].items():
            if not torch.equal(v, candidate[key][k]):
                raise ValueError('Candidate changes inherited sensory trajectory')
    for k in ['projection.weight', 'projection.bias']:
        assert torch.equal(source['readout'][k][1:], candidate['readout'][k][1:])
    for k in ['training_reference', 'training_scale']:
        assert torch.equal(source['readout'][k], candidate['readout'][k])
    model, head, dn, _, lineage = assembly(plan['source'])
    new_head = MuscleVehicleReadout(6); new_head.load_state_dict(candidate['readout'])
    optical_state = source['optical']
    optical = RetinalResidualPort(optical_state['reference'], optical_state['scale'], len(model.motor.backbone.vision_ports))
    optical.load_state_dict(optical_state)
    for module in [model, head, new_head, dn, optical]:
        module.eval().requires_grad_(False)
    sr = read(Path(plan['spatial'])/'result.json'); extras = []
    for i, (entry, (d, _, provenance)) in enumerate(zip(entries, items)):
        if provenance['arrays_sha256'] != entry['arrays_sha256']:
            raise ValueError('New actor arrays changed')
        path = Path(plan['spatial'])/f'spatial-{i}.npz'
        if sha256(path) != sr['files_sha256'][path.name]:
            raise ValueError('Static spatial source changed')
        with np.load(path) as z:
            j = z['event_to_camera']; np.testing.assert_array_equal(z['camera_indices'][j], d['camera_index'])
            extras.append(torch.tensor(np.concatenate([d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64), z['discarded_grid8'][j]],1), dtype=torch.float32))
    state = dict(episode=0, index=0)
    def descending_hook(module, args, value):
        steer, power = value
        return steer+dn(extras[state['episode']][state['index']:state['index']+1]), power
    def optical_hook(module, args, value):
        return value+optical(extras[state['episode']][state['index']:state['index']+1,64:])
    handles = [model.motor.command.register_forward_hook(descending_hook), model.visual.register_forward_hook(optical_hook)]
    alignment, sign_receipt = steering_yaw_alignment([item[0] for item in items[:2]])
    run = begin_run(a.output, dict(plan=plan, plan_sha256=sha256(a.plan), entries=entries, lineage=lineage,
        program_sha256=sha256(__file__), optimizer_updates=0, seed=None, sign_receipt=sign_receipt,
        actor_qualified=False, vehicle_control_sent=False,
        scope='Previously selected head transfer to independent repeated Monza recordings; no new-track or live claim'))
    shutil.copyfile(__file__, run/'executed-entrypoint.py')
    goals = torch.tensor(racing_goal(300,1)); views = torch.ones(1,4,dtype=torch.bool)
    saved = {}; reports = []; started = time.monotonic()
    out = dict(status='failed', optimizer_updates=0, actor_qualified=False, vehicle_control_sent=False)
    try:
        with np.load(Path(plan['baseline_fit'])/'predictions.npz') as reference, torch.no_grad():
            for episode, (d,c,_) in enumerate(items):
                state['episode'] = episode; model.reset_episode(); old = []; new = []; muscles = []
                scenes = torch.tensor(d['scene'], dtype=torch.float32)
                for i in range(len(scenes)):
                    state['index'] = i
                    _, mu = model.sensory_decision(decision_time=float(d['decision_time'][i]), racing_goals=goals,
                        scene_features=scenes[i:i+1], scene_valid=views, body_context=context(d,c,i))
                    old.append(head(mu)[0].numpy().copy()); new.append(new_head(mu)[0].numpy().copy()); muscles.append(mu[0].numpy().copy())
                    if i % 1000 == 0:
                        write_json(run/'progress.json', dict(episode=episode,event=i,total=len(scenes),optimizer_updates=0,elapsed_seconds=time.monotonic()-started))
                old = np.asarray(old); new = np.asarray(new)
                reproduction = float(abs(old-reference[f'baseline_{episode}']).max())
                if reproduction > 2e-6:
                    raise ValueError('Original full-episode replay not reproduced')
                assert np.array_equal(old[:,1:], new[:,1:])
                pre = pre_mask(d, alignment)
                reports.append(dict(episode=episode, source=metrics(old[:,0],d,pre), candidate=metrics(new[:,0],d,pre),
                    original_replay_max_difference=reproduction, pedal_predictions_exact=True))
                saved[f'source_{episode}'] = old; saved[f'candidate_{episode}'] = new
                saved[f'muscles_{episode}'] = np.asarray(muscles); saved[f'pre_mask_{episode}'] = pre
                np.savez_compressed(run/'predictions.npz', **saved)
                write_json(run/'completed-episodes.json', dict(completed_episodes=episode+1,reports=reports))
        out.update(status='completed_frozen_steering_head_transfer',comparisons=reports,
            predictions_sha256=sha256(run/'predictions.npz'), neural_trajectories_shared_exact=True,
            termination_reason='Three continuous new recordings evaluated with previously fixed heads, zero fit or selection; not unseen-track/live qualification')
    except Exception as error:
        out.update(error=type(error).__name__,reason=str(error),termination_reason='Explicit replay/invariant failure; no deployment')
    finally:
        for h in handles: h.remove()
    out['elapsed_seconds'] = time.monotonic()-started
    write_json(run/'result.json',out)
    if out['status']=='failed': raise SystemExit(1)


if __name__ == '__main__': main()
