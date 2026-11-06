"""Frozen same-body visual interventions on the three newer Monza recordings.

Feature-boundary counterfactuals have no independent action ground truth. Original
camera/body clocks remain receiver clocks; injected features are explicitly offline
stimuli, never asserted to be fresh synchronized camera observations.
"""
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
from flybrain_drive.sustained_turn_diagnostic import sustained_onsets, steering_yaw_alignment
from scripts.fit_bounded_sac_steering import metrics, pre_mask
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.sac_steering_experiment import assembly, snapshot, restore
from scripts.train_qualified_sac_unified import context, load_episode


def floor_indices(times, wanted):
    times = np.asarray(times)
    wanted = np.asarray(wanted)
    if (times.ndim != 1 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(wanted).all()):
        raise ValueError('Finite strictly increasing recorded times required')
    indices = np.searchsorted(times, wanted, side='right') - 1
    if np.any(indices < 0) or np.any(wanted > times[-1]):
        raise ValueError('Counterfactual time outside source; do not clamp or invent frames')
    assert np.all(times[indices] <= wanted)
    return indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); plan = read(a.plan); torch.set_num_threads(1)
    if plan['program_sha256'] != sha256(__file__):
        raise ValueError('Sealed entrypoint changed')
    transfer = read(plan['transfer_plan'])
    if sha256(plan['transfer_plan']) != plan['transfer_plan_sha256']:
        raise ValueError('Cohort/checkpoint plan changed')
    for key in ('source', 'candidate'):
        if sha256(Path(transfer[key])/'steering.pt') != transfer[key+'_sha256']:
            raise ValueError('Protected frozen checkpoint changed')
    if sha256(transfer['split']) != transfer['split_sha256']:
        raise ValueError('Cohort changed')
    entries = three_episode_cohort(transfer['split'])
    items = [load_episode(e['data']) for e in entries]
    model, old_head, dn, _, lineage = assembly(transfer['source'])
    cp = torch.load(Path(transfer['candidate'])/'steering.pt', map_location='cpu', weights_only=True)
    parent = torch.load(Path(transfer['source'])/'steering.pt', map_location='cpu', weights_only=True)
    for key in ('adapter', 'optical'):
        assert set(cp[key]) == set(parent[key])
        assert all(torch.equal(v, cp[key][k]) for k, v in parent[key].items())
    new_head = MuscleVehicleReadout(6); new_head.load_state_dict(cp['readout'])
    for key in ('projection.weight', 'projection.bias'):
        assert torch.equal(parent['readout'][key][1:], cp['readout'][key][1:])
    for key in ('training_reference', 'training_scale'):
        assert torch.equal(parent['readout'][key], cp['readout'][key])
    ac = parent['optical']
    optical = RetinalResidualPort(ac['reference'], ac['scale'], len(model.motor.backbone.vision_ports))
    optical.load_state_dict(ac)
    native_port=None;native_values=[];native_valid=[];native_receipt=None
    if 'native' in plan:
        from flybrain_drive.audited_native_flight_candidate import load_audited_native_flight_candidate
        components,native_receipt=load_audited_native_flight_candidate(plan['native'])
        if native_receipt['source_checkpoint_sha256']!=transfer['source_sha256']:raise ValueError('Different original visual/flight assembly')
        model,new_head,dn,optical,native_port,_=components
        # The source head remains the original pre-native head, now acting on
        # the same learned native-port trajectory as the protected head.
        old_head.load_state_dict(parent['readout'])
        native_cfg=read(Path(plan['native']['fit'])/'manifest.json')['config']
        if native_cfg['entries']!=entries:raise ValueError('Native and visual cohorts differ')
        extension=Path(native_cfg['arguments']['extension']);er=read(extension/'result.json')
        for episode,(d,_,_) in enumerate(items):
            path=extension/f'extension-{episode}.npz'
            if sha256(path)!=er['files_sha256'][path.name]:raise ValueError('Native sidecar changed')
            with np.load(path,allow_pickle=False) as z:
                for key in ('body_packet','body_receipt','decision_time'):np.testing.assert_array_equal(z[key],d[key])
                native_values.append(torch.tensor(z['values']));native_valid.append(torch.tensor(z['valid']))
    for module in (model, old_head, new_head, dn, optical):
        module.eval().requires_grad_(False)
    spatial_root = Path(transfer['spatial']); spatial = read(spatial_root/'result.json')
    extras = []
    for episode, (entry, (d, _, provenance)) in enumerate(zip(entries, items)):
        assert provenance['arrays_sha256'] == entry['arrays_sha256']
        path = spatial_root/f'spatial-{episode}.npz'
        assert sha256(path) == spatial['files_sha256'][path.name]
        with np.load(path, allow_pickle=False) as z:
            join = z['event_to_camera']
            np.testing.assert_array_equal(z['camera_indices'][join], d['camera_index'])
            extras.append(torch.tensor(np.concatenate([
                d['scene'].reshape(-1,4,9,16)[:,:,0].reshape(-1,64),
                z['discarded_grid8'][join]], axis=1), dtype=torch.float32))
    # Preserve the established sign convention from the original TRAIN records.
    sign_evidence = read(Path(transfer['candidate'])/'manifest.json')['config']['sign_evidence']
    assert sign_evidence['training_only'] and sign_evidence['agreement'] >= .9
    alignment = sign_evidence['sign']
    run = begin_run(a.output, dict(plan=plan, plan_sha256=sha256(a.plan), entries=entries,
        lineage=lineage, sign_evidence=sign_evidence, optimizer_updates=0, seed=None,
        actor_qualified=False, vehicle_control_sent=False,
        body_and_clock='Original receiver body/current and per-view history/masks/receipts; no oracle repair',
        native_candidate=native_receipt,
        scope='Consumed repeated Monza recordings; feature interventions, not pixel replay or unseen-track qualification'))
    shutil.copyfile(__file__, run/'executed-entrypoint.py')
    goals = torch.tensor(racing_goal(300,1)); views = torch.ones(1,4,dtype=torch.bool)
    holder = dict(extra=None, spatial_neutral=False,episode=0,index=0)
    def descending_hook(module, args, value):
        steer, power = value
        return steer+dn(holder['extra']), power
    def optical_hook(module, args, value):
        return value if holder['spatial_neutral'] else value+optical(holder['extra'][:,64:])
    handles = [model.motor.command.register_forward_hook(descending_hook),
               model.visual.register_forward_hook(optical_hook)]
    if native_port is not None:
        def native_hook(module,args,value):
            e,i=holder['episode'],holder['index']
            expected=torch.tensor(items[e][0]['body'][i:i+1],dtype=torch.float32)
            if not torch.equal(args[0],expected):raise ValueError('Native port attached to another receiver body')
            return value+native_port(native_values[e][i:i+1],native_valid[e][i:i+1])
        handles.append(model.body_feedback.register_forward_hook(native_hook))
    started = time.monotonic(); reports = []
    result = dict(status='failed', optimizer_updates=0, actor_qualified=False, vehicle_control_sent=False)
    try:
        with np.load(Path(plan['reference'])/'predictions.npz', allow_pickle=False) as reference, torch.no_grad():
            if sha256(Path(plan['reference'])/'predictions.npz') != plan['reference_predictions_sha256']:
                raise ValueError('Previously evaluated original trajectories changed')
            for episode, (d,c,_) in enumerate(items):
                holder['episode']=episode
                t = d['decision_time']; scene = torch.tensor(d['scene'], dtype=torch.float32)
                events = sustained_onsets(d, alignment); starts = {w['start'] for w in events}
                windows = []
                for w in events:
                    options = [z for z in events if z['direction'] != w['direction']]
                    donor = min(options, key=lambda z:abs(d['body'][z['start'],16]-d['body'][w['start'],16])) if options else None
                    windows.append(dict(**w, donor_event=donor['event'] if donor else None,
                                        donor_time=donor['time'] if donor else None))
                pre = pre_mask(d, alignment); saved = {}; states = {}; old = []; normal = []; muscles = []
                if native_port is not None:
                    saved.update(native_values=native_values[episode].numpy().copy(),native_valid=native_valid[episode].numpy().copy())
                def step(i, j=None, spatial_neutral=False):
                    j = i if j is None else j
                    holder.update(extra=extras[episode][j:j+1], spatial_neutral=spatial_neutral,index=i)
                    _, mu = model.sensory_decision(decision_time=float(t[i]), racing_goals=goals,
                        scene_features=scene[j:j+1], scene_valid=views, body_context=context(d,c,i))
                    return old_head(mu)[0].numpy().copy(), new_head(mu)[0].numpy().copy(), mu[0].numpy().copy()
                model.reset_episode()
                for i in range(len(scene)):
                    if i in starts: states[i] = snapshot(model)
                    before = snapshot(model) if pre[i] else None
                    op, npred, mu = step(i); old.append(op); normal.append(npred); muscles.append(mu)
                    if before is not None:
                        containing = [w for w in windows if w['time']-.8 <= t[i] <= w['time']-.08 and w['donor_time'] is not None]
                        if containing:
                            w = containing[0]; j = int(floor_indices(t, t[i]-w['time']+w['donor_time']))
                            after = snapshot(model); restore(model, before)
                            _, one, _ = step(i,j); restore(model, after)
                            saved[f'instant_{i}'] = np.array([j, npred[0], one[0]], dtype=np.float64)
                    if i % 1000 == 0:
                        write_json(run/'progress.json', dict(phase='continuous_replay', episode=episode,
                            event=i,total=len(scene),elapsed_seconds=time.monotonic()-started))
                old = np.asarray(old); normal = np.asarray(normal); muscles = np.asarray(muscles)
                for name, values in [('source',old), ('candidate',normal)]:
                    key='baseline' if native_port is not None and name=='source' else name
                    if np.max(abs(values-reference[f'{key}_{episode}'])) > 2e-6:
                        raise ValueError('Full prefix replay differs from independent prior reference')
                assert np.array_equal(old[:,1:],normal[:,1:])
                saved.update(normal=normal, source=old, muscles=muscles, pre_mask=pre)
                window_reports = []
                for wi,w in enumerate(windows):
                    ids = np.arange(w['start'], w['event']+1)
                    mask = pre[ids] & (t[ids] >= w['time']-.8) & (t[ids] <= w['time']-.08)
                    paths = dict(normal=ids, delayed600=floor_indices(t,t[ids]-.6),
                                 frozen_start=np.full(len(ids),ids[0]), spatial_neutral=ids)
                    if w['donor_time'] is not None:
                        paths['opposite'] = floor_indices(t,t[ids]-w['time']+w['donor_time'])
                    preds = {}
                    for name, donor_indices in paths.items():
                        restore(model, states[w['start']]); pred = []
                        for i,j in zip(ids, donor_indices):
                            pred.append(step(int(i),int(j),name=='spatial_neutral')[1])
                        preds[name] = np.asarray(pred)
                        saved[f'window_{wi}_{name}'] = preds[name]
                        saved[f'window_{wi}_{name}_indices'] = donor_indices
                    replay_error = float(abs(preds['normal']-normal[ids]).max())
                    if replay_error > 2e-6: raise ValueError('Restored same-state normal branch differs')
                    changes = {name:float(np.mean(abs(preds[name][mask,0]-preds['normal'][mask,0])))
                               for name in paths if name!='normal'} if mask.any() else {}
                    signed = float(np.mean((preds['normal'][mask,0]-preds['opposite'][mask,0])*w['direction'])) if mask.any() and 'opposite' in paths else None
                    saved[f'window_{wi}_receiver_indices'] = ids; saved[f'window_{wi}_pre_mask'] = mask
                    window_reports.append(dict(**w,supervised_pre_count=int(mask.sum()),
                        replay_max_error=replay_error,mean_abs_steering_change=changes,
                        signed_normal_minus_opposite=signed,initial_state_and_body_history_equal=True,
                        counterfactual_action_ground_truth=False))
                path = run/f'episode-{episode}.npz'; np.savez_compressed(path,**saved)
                reports.append(dict(episode=episode,episode_id=c['episode_id'],model_events=len(scene),
                    source=metrics(old[:,0],d,pre),candidate=metrics(normal[:,0],d,pre),
                    windows=window_reports,predictions_sha256=sha256(path),pedal_predictions_exact=True))
                write_json(run/'completed-episodes.json',dict(comparisons=reports))
        result.update(status='completed_frozen_precorner_visual_transfer',comparisons=reports,
            termination_reason='Three full continuous prefixes, same-state/body interventions and frozen head comparison completed; no live/unseen-track qualification')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error),comparisons=reports,
                      termination_reason='Explicit diagnostic or invariant failure; original evidence retained')
    finally:
        for h in handles: h.remove()
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(run/'result.json',result)
    if result['status']=='failed': raise SystemExit(1)


if __name__ == '__main__': main()
