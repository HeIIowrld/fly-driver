"""Derive offline visual/body maneuver commands with a fixed historical readout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.visual_preview_speed import PreviewSpeedCommand
from scripts.prepare_flyvis_preview_folds import registered_cpu
from scripts.replay_registered_visual_yaw import predictor


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','visual','body-context','body-audit','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('train','development'),required=True)
    a=p.parse_args();candidate,visual,body,ba=map(Path,(a.candidate,a.visual,a.body_context,a.body_audit))
    root=candidate/'recording'; events=root/'events.jsonl'
    outer,vr,br,bar=[read(x/'result.json') for x in (candidate,visual,body,ba)]
    vm,bm,am=[read(x/'manifest.json')['config'] for x in (visual,body,ba)]
    if (outer['status']!='completed_preassigned_pace_recording' or outer['role']!=a.role
            or outer['fit_allowed'] is not False or vr['status']!='completed'
            or br['status']!='completed_high_rate_per_view_body' or bar['status']!='completed'
            or any(m['events_sha256']!=sha256(events) for m in (vm,bm,am))
            or sha256(visual/'responses.npz')!=vr['responses_sha256']
            or sha256(body/'per-view-body.npz')!=br['arrays_sha256']
            or am['arrays_sha256']!=br['arrays_sha256']
            or not vr.get('reference_replay_max_errors')
            or max(vr['reference_replay_max_errors'].values())>2e-6):
        raise ValueError('Audited role, original body, and reference-checked visual replay required')
    b=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    folds=b/'20261030-t1692-fold-setup'; result=read(folds/'result.json')
    cp=folds/'episode-04-train/vision_body.pt'
    fold=next(row for row in result['folds'] if row['heldout_history']=='episode-04-train')
    if result['status']!='completed_cpu_train_history_fold_setup' or sha256(cp)!=fold['models']['vision_body']['sha256']:
        raise ValueError('Original train-only fold checkpoint required')
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    if (saved['condition']!='vision_body' or saved['heldout_history']!='episode-04-train'
            or saved['fitted_history_ids']!=['episode-05-train','episode-06-train']
            or saved['actuator_policy'] is not False):
        raise ValueError('Original fixed preview training provenance required')
    with np.load(visual/'responses.npz',allow_pickle=False) as f:
        if f['additional_cell_types'].tolist()!=['L1','L2','Mi1','Tm3','Tm1','Tm9']:
            raise ValueError('Four-view original fourteen visual cell types required')
        clocks=f['decision_monotonic_seconds'].copy()
        activity=np.concatenate([f['additional_activity'],f['motion_activity']],axis=2)
    with np.load(body/'per-view-body.npz',allow_pickle=False) as f:
        body_values,body_valid=f['current_values'].copy(),f['current_valid'].copy()
        if not np.array_equal(clocks,f['decisions']):raise ValueError('Original clocks differ')
    with np.load(visual/'retina-registration.npz',allow_pickle=False) as f:
        if f['view_order'].tolist()!=['front','left','right','rear']:raise ValueError('Four views required')
        rays,origins=f['body_directions'].copy(),f['body_origins_m'].copy()
    with np.load(root/'observations.npz',allow_pickle=False) as f:
        if not np.array_equal(clocks,f['observation_time']):raise ValueError('Original observation clocks differ')
        view_valid=torch.as_tensor(f['obs_view_valid'].all(axis=2))
    episode=dict(activity=activity,clock=clocks,body=body_values,body_valid=body_valid,
                 rays=rays,origins=origins)
    torch.set_num_threads(1)
    features=registered_cpu(episode,saved['channel_mean'],saved['channel_scale'],batch_size=8)['intact']
    model=PreviewSpeedCommand(predictor(saved),curvature_gain_m=100.).eval()
    with torch.no_grad():
        outputs=[model(torch.as_tensor(features,dtype=torch.float32),
            torch.as_tensor(body_values,dtype=torch.float32),torch.as_tensor(body_valid,dtype=torch.bool),
            view_valid,torch.tensor(racing_goal(100,len(clocks))),intervention=condition)
            for condition in (None,'motion_channels')]
    commands=np.stack([o['command'].numpy() for o in outputs])
    if commands.shape!=(2,len(clocks),2) or not np.isfinite(commands).all():
        raise ValueError('Complete finite physical preview commands required')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),role=a.role,events_sha256=sha256(events),
        visual_sha256=vr['responses_sha256'],body_sha256=br['arrays_sha256'],preview_checkpoint_sha256=sha256(cp),
        camera_ceiling_kmh=100,curvature_gain_m=100.,scope='fixed engineering preview, diagnostic only'))
    np.savez_compressed(run/'commands.npz',commands=commands,decision_clock=clocks)
    d=np.abs(commands[0]-commands[1])
    out=dict(status='completed_visual_preview_commands_diagnostic',role=a.role,decisions=len(clocks),
        commands_sha256=sha256(run/'commands.npz'),motion_mean_abs_command_delta=d.mean(0).tolist(),
        predicted_speed_kmh_min_max=[float(commands[0,:,1].min()*3.6),
                                     float(commands[0,:,1].max()*3.6)],
        predicted_yaw_abs_max=float(np.abs(commands[0,:,0]).max()),
        scene_or_body_from_teacher_or_world_pose=False,source_model_actor_qualified=False,
        physics_or_preview_accuracy_verified=False,fit_allowed=False,optimizer_updates=0)
    write_json(run/'result.json',out);print(json.dumps(out))


if __name__=='__main__':main()
