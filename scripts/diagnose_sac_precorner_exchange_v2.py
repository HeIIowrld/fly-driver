"""Exact same-body/state visual-feature exchange before yaw-transition proxies.

Recorded features are counterfactual boundary stimuli, not synchronized new videos
or a physically simulated other corner. Teacher/body truth only selects/evaluates
windows; it never enters the new visual adapter or vehicle readout.
"""
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.visual_port_current_correction import VisualPortCurrentCorrection
from flybrain_drive.sustained_turn_diagnostic import steering_yaw_alignment,paired_sustained_onsets
from scripts.train_qualified_sac_unified import context
from scripts.sac_steering_experiment import dataset,assembly,snapshot,restore,paired_onsets,SOURCE_SHA,SOURCE

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit');p.add_argument('--episode',type=int,default=2);p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(1);entries,items,extras=dataset();d,c,_=items[a.episode]
 model,readout,adapter,_,_=assembly(a.fit);model.requires_grad_(False);readout.requires_grad_(False);adapter.requires_grad_(False)
 source_dn=copy.deepcopy(adapter)
 source_cp=torch.load(SOURCE/'spatial-port.pt',map_location='cpu',weights_only=True)
 source_dn.load_state_dict(source_cp['adapter'],strict=True);source_dn.requires_grad_(False)
 optical=None;correction=None
 if a.fit:
  cp=torch.load(Path(a.fit)/'steering.pt',map_location='cpu',weights_only=True)
  ac=cp.get('optical')
  cc=cp.get('port_correction')
  if cc is not None:
   correction=VisualPortCurrentCorrection(cc['reference'],cc['scale'])
   correction.load_state_dict(cc,strict=True);correction.requires_grad_(False)
  if ac is not None:
   optical=RetinalResidualPort(ac['reference'],ac['scale'],len(model.motor.backbone.vision_ports))
   optical.load_state_dict(ac,strict=True);optical.requires_grad_(False)
 alignment,sign_evidence=steering_yaw_alignment([item[0] for item in items[:2]])
 windows=paired_sustained_onsets(d,alignment);assert windows,'No eligible opposite-direction windows'
 run=begin_run(a.output,dict(arguments=vars(a),seed=780501,program_sha256=sha256(__file__),helper_sha256=sha256('scripts/sac_steering_experiment.py'),source_checkpoint_sha256=SOURCE_SHA,entries=entries,windows=windows,correction_present=correction is not None,source_path_definition='Disable only new four-port correction while retaining fixed parent optical/DN if correction is present; otherwise original optical/DN baseline ablation',sign_evidence=sign_evidence,stable_definition_sha256=sha256('src/flybrain_drive/sustained_turn_diagnostic.py'),
   onset='First |yaw|>=.15 with previous .6s mean |yaw|<.07, following .6s |median yaw|>=.15 and >=80% same yaw sign; separated2s; TRAIN-only steering/yaw sign alignment; diagnostic proxy only',
   window_seconds=[-1.2,0],evaluation_seconds=[-.8,-.08],body_low_yaw_gate=.07,
   intervention='Swap old scene576 and extra static320 with opposite-turn recorded features at floor donor-relative time; receiver clocks and complete body/history/masks stay exact; not physical exposure or FlyVis replay',
   actor_deployed=False,vehicle_control_sent=False))
 out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False);holder={'extra':None,'neutral':False,'source_path':False}
 def hook(m,args,output):
  steering,power=output;v=(source_dn if holder['source_path'] else adapter)(holder['extra'])
  if correction is not None and not holder['neutral'] and not holder['source_path']:v=correction(holder['extra'],v)
  if holder['neutral']:v=(adapter.maximum*adapter.tonic_logit.sigmoid()).expand_as(v)
  return steering+v,power
 h=model.motor.command.register_forward_hook(hook)
 def optical_hook(m,args,value):
  delta=optical(holder['extra'][:,64:])
  return value if holder['neutral'] or (holder['source_path'] and correction is None) else value+delta
 oh=model.visual.register_forward_hook(optical_hook) if optical is not None else None
 goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool)
 scene=torch.tensor(d['scene'],dtype=torch.float32);extra=torch.tensor(extras[a.episode],dtype=torch.float32)
 def step(i,j=None,neutral=False,source_path=False):
  j=i if j is None else j;holder['extra']=extra[j:j+1];holder['neutral']=neutral;holder['source_path']=source_path
  _,muscle=model.sensory_decision(decision_time=float(d['decision_time'][i]),racing_goals=goals,
     scene_features=scene[j:j+1],scene_valid=views,body_context=context(d,c,i))
  return readout(muscle)[0].numpy().copy(),muscle[0].numpy().copy()
 start=time.monotonic()
 try:
  states={};normal=[];muscles=[];starts={w['start'] for w in windows};model.reset_episode()
  for i in range(len(scene)):
   if i in starts:states[i]=snapshot(model)
   pr,mu=step(i);normal.append(pr);muscles.append(mu)
   if i%1000==0:write_json(run/'progress.json',dict(phase='continuous_prefix',event=i,total=len(scene)))
  normal=np.asarray(normal);muscles=np.asarray(muscles);reports=[];saved={'normal':normal,'muscles':muscles}
  original=(Path(a.fit)/'predictions.npz' if a.fit else Path('runs/visual-circuit-spatial-20261103/descending-residual-01/predictions.npz'))
  with np.load(original,allow_pickle=False) as z:expected=z['dev' if a.episode==2 else f'train_{a.episode}']
  # Training predictions are online parameter-changing trajectories; only dev is exact reference.
  reference_error=float(abs(normal-expected).max()) if a.episode==2 else None
  if reference_error is not None and reference_error>2e-6:raise ValueError('Saved fixed checkpoint replay differs')
  for wi,w in enumerate(windows):
   ids=np.arange(w['start'],w['event']+1);t=d['decision_time'];wanted=t[ids]-w['time']+w['donor_time']
   donor=np.searchsorted(t,wanted,side='right')-1
   if np.any(donor<0):raise ValueError('Donor window outside recorded episode')
   same_options=[z for z in windows if z['direction']==w['direction'] and z['event']!=w['event']]
   same=min(same_options,key=lambda z:abs(d['body'][z['start'],16]-d['body'][w['start'],16])) if same_options else None
   same_donor=(np.searchsorted(t,t[ids]-w['time']+same['time'],side='right')-1) if same else ids
   paths={}
   for case in ['normal','opposite_video','same_direction_video','new_path_neutral','source_path']:
    restore(model,states[w['start']]);pred=[]
    for n,i in enumerate(ids):pred.append(step(int(i),int(donor[n]) if case=='opposite_video' else int(same_donor[n]) if case=='same_direction_video' else None,case=='new_path_neutral',case=='source_path')[0])
    paths[case]=np.asarray(pred);saved[f'window_{wi}_{case}']=paths[case]
   error=float(abs(paths['normal']-normal[ids]).max())
   if error>2e-6:raise ValueError('Identical restored state/body/scene did not reproduce original window')
   mask=(t[ids]>=w['time']-.8)&(t[ids]<=w['time']-.08)&d['fit_mask'][ids]&(abs(d['body'][ids,4])<.07)
   delta=paths['normal'][:,0]-paths['opposite_video'][:,0]
   neutral=paths['normal'][:,0]-paths['new_path_neutral'][:,0]
   direction_control=paths['same_direction_video'][:,0]-paths['opposite_video'][:,0]
   saved[f'window_{wi}_indices']=ids;saved[f'window_{wi}_donor_indices']=donor;saved[f'window_{wi}_pre_mask']=mask;saved[f'window_{wi}_same_donor_indices']=same_donor
   report=dict(**w,same_direction_donor_event=same['event'] if same else None,
      mean_signed_same_minus_opposite=float(np.mean(direction_control[mask])*w['direction']) if mask.any() and same else None,supervised_pre_count=int(mask.sum()),normal_replay_max_error=error,
      mean_signed_normal_minus_opposite=float(np.mean(delta[mask])*w['direction']) if mask.any() else None,
      mean_abs_delta=float(np.mean(abs(delta[mask]))) if mask.any() else None,
      mean_signed_new_path_effect=float(np.mean(neutral[mask])*w['direction']) if mask.any() else None,
      counterfactual_has_no_action_ground_truth=True,body_and_body_history_equal=True,initial_state_equal=True)
   reports.append(report);write_json(run/'progress.json',dict(phase='paired_windows',window=wi+1,total=len(windows)))
  path=run/'predictions.npz';np.savez_compressed(path,**saved)
  usable=[r for r in reports if r['supervised_pre_count']]
  out.update(status='completed_same_body_precorner_visual_exchange',normal_replay_max_error=reference_error,windows=reports,
     usable_windows=len(usable),right_direction_windows=sum(r['mean_signed_normal_minus_opposite']>0 for r in usable),
     mean_signed_delta=float(np.mean([r['mean_signed_normal_minus_opposite'] for r in usable])),
     predictions_sha256=sha256(path),termination_reason='Complete original replay and all declared same-state/body counterfactual windows completed')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit diagnostic failure')
 finally:
  h.remove()
  if oh is not None:oh.remove()
 out['elapsed_seconds']=time.monotonic()-start;write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
