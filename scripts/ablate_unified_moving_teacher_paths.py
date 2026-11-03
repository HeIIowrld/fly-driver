"""Causal path ablations on archived moving teacher observations; no vehicle control."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from scripts.audit_causal_copy_body_context import first_observed_packets
from scripts.replay_process_shadow_live_original import body_from_saved
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--device',choices=('cpu','cuda'),required=True);a=ap.parse_args();planpath=B/'20261101-adaptive-contrast-plan/plan.json';plan=read(planpath)
 trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
 if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage')
 roots=[B/f'20261101-flyvis-varied-causal-{seed}' for seed in (879511,879512)];hashes=[]
 for root,seed in zip(roots,(879511,879512),strict=True):
  rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=rr['responses_sha256'] or cfg['texture_seed']!=seed:raise ValueError('synthetic source')
  hashes.append(sha256(f))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates=0,actor=False,real_data_used_for_fit=False,device=a.device))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  torch.set_num_threads(1);torch.manual_seed(863009)
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  spec=read(trdir/'manifest.json')['config']['plan'];source=read(Path(spec['source'])/'result.json')
  visual=EmbodiedVisualFlightCircuit(spec['source_spec'],torch.load(source['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu');visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
  mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];motor=PhysicalManeuverCircuit(mp['spec'],torch.load(mp['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
  temporal=OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,visual.backbone.scene_command.scale,len(visual.backbone.vision_ports));temporal.load_state_dict(torch.load(opdir/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model'])
  boundary=PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports));boundary.load_state_dict(torch.load(ctxdir/'context-boundary.pt',map_location='cpu',weights_only=True)['model'])
  model=UnifiedFlightSensorimotor(motor,visual.backbone,temporal,boundary,body_feedback=CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports)))
  model.motor.backbone.circuit.sparse_backend='numba_csr8_cpu' if a.device=='cpu' else 'cached_csr_gpu';model=model.to(a.device).eval().requires_grad_(False)
  fit=B/'20261101-speed-wheel-turn-control-fit';fr=read(fit/'result.json');f=fit/'speed-body-adapter.pt'
  if fr['status']!='completed_synthetic_speed_wheel_turn_control_training' or sha256(f)!=fr['checkpoint_sha256']:raise ValueError('body checkpoint')
  payload=torch.load(f,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False:raise ValueError('body qualification')
  model.body_feedback.load_state_dict(payload['model'],strict=True);parameter=model.temporal.opponent_projection.weight;parameter.requires_grad_(False);initial=parameter.detach().clone()
  goals=torch.tensor(racing_goal(100,1),device=a.device);view=torch.ones((1,4),dtype=torch.bool,device=a.device)
  def load_scene(seed,amplitude,root):
   r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
   if r['status']!='completed_causal_four_view_constant_yaw' or sha256(f)!=r['responses_sha256'] or cfg['texture_seed']!=seed or cfg['program_sha256']!=plan['generator_sha256'] or r['cases']!=[0.,-amplitude,amplitude]:raise ValueError('declared causal synthetic stimulus required')
   with np.load(f,allow_pickle=False) as z:ret=z['retina'].copy();mot=z['motion'].copy();times=z['times'].copy()
   if ret.shape!=(3,41,4,721) or mot.shape!=(3,41,4,8,721) or not np.isfinite(ret).all() or not np.isfinite(mot).all() or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12) or not np.all(ret[:,:11]==ret[:,:1]):raise ValueError('synthetic stimulus shape')
   full=np.concatenate([ret[:,:,:,None],mot],axis=3)
   with torch.no_grad():sc=model.scene_pool(torch.tensor(full.reshape(-1,4,9,721),dtype=torch.float32,device=a.device)).reshape(3,41,576)
   return sc,times,sha256(f)
  def rollout(scene,times,learn):
   cases=[]
   for case in (1,2):
    model.reset_episode();values=np.zeros(21);values[16]=5.;values[17:21]=5./.29;valid=np.ones(21,dtype=bool);outputs=[]
    for i,t in enumerate(times):
     now=1.+float(t);body=NavigationBodyObservation('synthetic-yaw',i,now-.01,now,values.copy(),valid.copy(),'synthetic-only');ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
     fn=model.sensory_training_decision if learn else model.sensory_decision
     intent,_=fn(decision_time=now,racing_goals=goals,scene_features=scene[case,i:i+1],scene_valid=view,body_context=ctx);outputs.append(intent[0])
    cases.append(torch.stack(outputs))
   return torch.stack(cases)
  if sha256('src/flybrain_drive/synthetic_road_visual.py')!=plan['road_source_sha256'] or sha256('scripts/replay_flyvis_synthetic_road_varied.py')!=plan['generator_sha256']:raise ValueError('registered generator changed')
  ckroot=B/'20261101-causal-yaw-multispeed-fit';crs=read(ckroot/'result.json');ck=ckroot/'opponent-projection.pt'
  if crs['status']!='completed_synthetic_multispeed_yaw_training' or sha256(ck)!=crs['checkpoint_sha256']:raise ValueError('visual checkpoint')
  candidate=torch.load(ck,map_location='cpu',weights_only=True)
  if candidate['actor_qualified'] is not False or candidate['source_body_sha256']!=sha256(f):raise ValueError('fixed body lineage')
  with torch.no_grad():model.temporal.opponent_projection.weight.copy_(candidate['weight'].to(a.device))
  model.spatial_contrast=RetinalSpatialContrastBoundary(len(model.motor.backbone.vision_ports)).to(a.device)
  weight=model.spatial_contrast.projection.weight;initial_weight=weight.detach().clone()
  optim=torch.optim.Adam([weight],lr=.003,eps=1e-12)
  def load_source(folder,seed,curvature,start):
   root=B/folder;rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];src=root/'responses.npz'
   if rr['status']!='completed_causal_four_view_synthetic_road' or sha256(src)!=rr['responses_sha256'] or cfg['road_generator_sha256']!=plan['road_source_sha256'] or cfg['texture_seed']!=seed:raise ValueError('road source provenance')
   if seed!=879701 and (cfg['program_sha256']!=plan['generator_sha256'] or cfg['curvature']!=curvature or cfg['bend_start_m']!=start):raise ValueError('registered varied road')
   if seed==879701 and (curvature!=.018 or start!=12):raise ValueError('original fixed road parameters')
   with np.load(src,allow_pickle=False) as z:ret=z['retina'].copy();mot=z['motion'].copy();times=z['times'].copy();bends=z['bend'].copy()
   if ret.shape!=(3,41,4,721) or mot.shape!=(3,41,4,8,721) or not np.isfinite(ret).all() or not np.isfinite(mot).all() or not np.array_equal(bends,[0,-1,1]) or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12):raise ValueError('road sample shape')
   image=np.concatenate([ret[:,:,:,None],mot],axis=3)
   with torch.no_grad():scene=model.scene_pool(torch.tensor(image.reshape(-1,4,9,721),dtype=torch.float32,device=a.device)).reshape(3,41,576)
   return scene,times,sha256(src)
  def rollout(scene,times,learn):
   cases=[]
   for case in range(3):
    model.reset_episode();outputs=[]
    for i,t in enumerate(times):
     now=1.+float(t);speed=0. if t<=.4 else 3.;values=np.zeros(21);values[16]=speed;values[17:21]=speed/.29;valid=np.ones(21,dtype=bool)
     body=NavigationBodyObservation('synthetic-road',i,now-.01,now,values,valid,'synthetic-only');ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
     fn=model.sensory_training_decision if learn else model.sensory_decision
     intent,_=fn(decision_time=now,racing_goals=goals,scene_features=scene[case,i:i+1],scene_valid=view,body_context=ctx);outputs.append(intent[0])
    cases.append(torch.stack(outputs))
   return torch.stack(cases)
  fit=B/'20261101-adaptive-pattern-fit';fr=read(fit/'result.json');candidate_file=fit/'adaptive-pattern-contrast.pt'
  if fr['status']!='completed_synthetic_adaptive_pattern_fit' or sha256(candidate_file)!=fr['checkpoint_sha256']:raise ValueError('candidate mismatch')
  candidate=torch.load(candidate_file,map_location=a.device,weights_only=True)
  if candidate['actor_qualified'] is not False or candidate['body_sha256']!=sha256(f) or candidate['temporal_sha256']!=sha256(ck):raise ValueError('candidate lineage')
  with torch.no_grad():weight.copy_(candidate['weight']);model.spatial_contrast.reference.copy_(candidate['reference']);model.spatial_contrast.max_contrast_rms.fill_(candidate['max_contrast_rms'])
  root=B/'20261031-process-body-owned-pause-live';original=read(root/'result.json');audit=read(B/'20261031-process-body-owned-pause-live-audit/result.json');recon=B/'20261101-real-pixels-flyvis-scene-reconstruction';rr=read(recon/'result.json');sf=recon/'published-scenes.npz'
  if original['status']!='completed_diagnostic' or original['training_admissible'] is not False or audit['status']!='completed_mmf_immutable_shadow_audit' or rr['status']!='completed' or rr['over_tolerance']!=0 or sha256(sf)!=rr['published_scenes_sha256']:raise ValueError('proven diagnostic scene reconstruction required')
  for name,key in [('body-ticker.json','ticker_sha256'),('native-copy-windows.json','native_sha256'),('bridge-events.json','bridge_sha256')]:
   if sha256(root/name)!=original[key]:raise ValueError('original raw source mismatch')
  rows=read(root/'bridge-events.json');ticks=read(root/'body-ticker.json');packets=first_observed_packets(read(root/'native-copy-windows.json'));by_packet={p['packet']:p for p in packets}
  published={x['capture_id']:x for x in rows if x.get('neural',{}).get('status')=='visual_published'}
  with np.load(sf,allow_pickle=False) as z:ids=z['capture_ids'].copy();features=z['scene'].copy()
  if len(ticks)!=365 or len(published)!=105 or features.shape!=(len(ids),576) or set(map(int,ids))!=set(published):raise ValueError('complete published sources required')
  scenes={int(c):features[i:i+1] for i,c in enumerate(ids)};identity=read('configs/ac-virtual-body-v1.json')['native_identity'];encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);episode=original['camera_request_id'];contract=sha256('configs/ac-virtual-navigation-body-v2.json')
  baseline_root=B/'20261101-unified-real-recording-cpu';br=read(baseline_root/'result.json');bf=baseline_root/'unified-replay.npz'
  if a.device!='cpu' or br['status']!='completed_unified_real_recording_offline_replay' or sha256(bf)!=br['output_sha256'] or br['source_reconstruction_sha256']!=sha256(sf) or br['candidate_sha256']!=sha256(candidate_file):raise ValueError('Exact CPU original reference required')
  with np.load(bf,allow_pickle=False) as z:reference=z['intents'].copy();reference_states=z['states'].copy();reference_times=z['decision'].copy()
  components={'visual_opponent':model.temporal.opponent_projection.weight,'visual_spatial':model.spatial_contrast.projection.weight,'visual_body_interaction':model.view_context.projection.weight,'added_body_feedback':model.body_feedback.projection.weight}
  originals={name:weight.detach().clone() for name,weight in components.items()};scenarios=['observed',*components.keys()];all_outputs={};all_states={};yaw=[]
  for scenario in scenarios:
   with torch.no_grad():
    for name,weight in components.items():weight.copy_(originals[name])
    if scenario!='observed':components[scenario].zero_()
   model.reset_episode();encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);outputs=[];states=[]
   for i,tick in enumerate(ticks):
    packet=by_packet[tick['packet']]
    if packet['first_end']!=tick['body_receipt'] or tick['decision']!=reference_times[i]:raise ValueError('original body receipt/decision mismatch')
    when=float(tick['decision']);body=encoder.encode(packet['sample'],episode_id=episode,decision_monotonic_seconds=when);visual=published[tick['visual_frame_index']]['visual']
    previous=tuple(body_from_saved(x,episode=episode,decision=visual['camera_receipt'],contract=contract) for x in visual['view_body'])
    ctx=per_view_temporal_body_context(previous,body,render_start_lowers=tuple(visual['render_start_lower']),camera_receipt_time=visual['camera_receipt'],decision_time=when)
    if scenario=='observed':yaw.extend(abs(x.values[4]) for x in previous if x is not None and x.valid[4])
    scene=torch.as_tensor(scenes[tick['visual_frame_index']],device=a.device);intent,_=model.sensory_decision(decision_time=when,racing_goals=goals,scene_features=scene,scene_valid=view,body_context=ctx)
    outputs.append(intent.detach().cpu().numpy()[0].copy());states.append(model.state.detach().cpu().numpy()[0].copy())
   all_outputs[scenario]=np.stack(outputs);all_states[scenario]=np.stack(states)
  base_error=float(np.max(np.abs(all_outputs['observed']-reference)));state_error=float(np.max(np.abs(all_states['observed']-reference_states)))
  if max(base_error,state_error)>2e-5:raise ValueError('Unmodified original reference mismatch')
  comparisons={}
  for name in scenarios[1:]:
   difference=all_outputs[name]-all_outputs['observed'];state_difference=np.abs(all_states[name]-all_states['observed'])
   comparisons[name]={'turn_max_abs':float(np.max(np.abs(difference[:,0]))),'turn_mean_abs':float(np.mean(np.abs(difference[:,0]))),'forward_max_abs':float(np.max(np.abs(difference[:,1]))),'forward_mean_abs':float(np.mean(np.abs(difference[:,1]))),'state_max_abs':float(state_difference.max()),'intervention':'projection weights set to zero; original scene, timestamps, body and masks held fixed'}
  out=run/'ablations.npz';np.savez_compressed(out,**{name+'_intent':all_outputs[name] for name in scenarios},**{name+'_state':all_states[name] for name in scenarios});result.update(status='completed_moving_teacher_shadow_path_ablations',decisions=len(ticks),baseline_intent_max_abs=base_error,baseline_state_max_abs=state_error,valid_prior_yaw_count=len(yaw),valid_prior_yaw_max_abs=max(yaw) if yaw else None,comparisons=comparisons,output_sha256=sha256(out),source_original_sha256=sha256(root/'result.json'),source_reconstruction_sha256=sha256(sf),training_admissible=False,optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False,physical_or_physiology_verified=False,interpretation='Offline path removals on one archived AI-teacher drive with current experimental unified circuit; no fit, live actuation, physiological or driving validity proof')

 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_moving_teacher_shadow_path_ablations':raise RuntimeError(result.get('reason','ablation failed'))
if __name__=='__main__':main()
