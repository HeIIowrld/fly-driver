"""Predeclared diagnostic gain sweep of one visual boundary in synthetic closed loop."""
import argparse,json,copy,gc,os,sys,time,yaml
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
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.stationary_camera_admission import validate_stationary_file_camera_admission
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder, NoFreshCSPFrame
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import repeated_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.immutable_camera_group import freeze_camera_group
from flybrain_drive.online_body_shadow import OnlineBodyShadow
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from flybrain_drive.unified_multirate_shadow import UnifiedMultirateShadow
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from flybrain_drive.flight_stream import held_input_batches
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat
from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.flight_airflow_feedback_adapter import AirflowFeedbackAdapter
from flybrain_drive.relative_airflow_sensor import relative_airflow,missing_airflow
from flybrain_drive.synthetic_crosswind_vehicle import advance_with_crosswind
from flybrain_drive.unified_airflow_speed import recurrent_speed_airflow_decision
from flybrain_drive.interval_yaw_visual_boundary import IntervalYawVisualBoundary
from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
from flybrain_drive.view_yaw_interval import ViewYawInterval
from flybrain_drive.synthetic_planar_vehicle import PlanarVehicleState, advance_planar_vehicle
from flybrain_drive.synthetic_planar_road_visual import SyntheticPlanarRoadVisual
from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();planpath=B/'20261101-adaptive-contrast-plan/plan.json';plan=read(planpath)
 trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
 if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage')
 roots=[B/f'20261101-flyvis-varied-causal-{seed}' for seed in (879511,879512)];hashes=[]
 for root,seed in zip(roots,(879511,879512),strict=True):
  rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=rr['responses_sha256'] or cfg['texture_seed']!=seed:raise ValueError('synthetic source')
  hashes.append(sha256(f))
 experiment_plan=B/'20261101-paired-bend-fit-plan/plan.json'; experiment=read(experiment_plan)
 if experiment['schema']!='flybrain.diagnostic.paired-bend-plan.v1' or experiment['real_data_used_for_fit'] is not False or experiment['actor_qualified'] is not False:raise ValueError('exact synthetic-only protocol required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),experiment_plan_sha256=sha256(experiment_plan),graph_sha256=sha256(tr['checkpoint']),optimizer_updates_planned=0,actor=False,real_data_used_for_fit=False))
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
  model.motor.backbone.circuit.sparse_backend='cached_csr_gpu';model=model.to('cuda').eval().requires_grad_(False)
  fit=B/'20261101-speed-wheel-turn-control-fit';fr=read(fit/'result.json');f=fit/'speed-body-adapter.pt'
  if fr['status']!='completed_synthetic_speed_wheel_turn_control_training' or sha256(f)!=fr['checkpoint_sha256']:raise ValueError('body checkpoint')
  payload=torch.load(f,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False:raise ValueError('body qualification')
  model.body_feedback.load_state_dict(payload['model'],strict=True);parameter=model.temporal.opponent_projection.weight;parameter.requires_grad_(False);initial=parameter.detach().clone()
  goals=torch.tensor(racing_goal(100,1),device='cuda');view=torch.ones((1,4),dtype=torch.bool,device='cuda')
  def load_scene(seed,amplitude,root):
   r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
   if r['status']!='completed_causal_four_view_constant_yaw' or sha256(f)!=r['responses_sha256'] or cfg['texture_seed']!=seed or cfg['program_sha256']!=plan['generator_sha256'] or r['cases']!=[0.,-amplitude,amplitude]:raise ValueError('declared causal synthetic stimulus required')
   with np.load(f,allow_pickle=False) as z:ret=z['retina'].copy();mot=z['motion'].copy();times=z['times'].copy()
   if ret.shape!=(3,41,4,721) or mot.shape!=(3,41,4,8,721) or not np.isfinite(ret).all() or not np.isfinite(mot).all() or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12) or not np.all(ret[:,:11]==ret[:,:1]):raise ValueError('synthetic stimulus shape')
   full=np.concatenate([ret[:,:,:,None],mot],axis=3)
   with torch.no_grad():sc=model.scene_pool(torch.tensor(full.reshape(-1,4,9,721),dtype=torch.float32,device='cpu')).reshape(3,41,576)
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
  with torch.no_grad():model.temporal.opponent_projection.weight.copy_(candidate['weight'].to('cuda'))
  model.spatial_contrast=RetinalSpatialContrastBoundary(len(model.motor.backbone.vision_ports)).to('cuda')
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
   with torch.no_grad():scene=model.scene_pool(torch.tensor(image.reshape(-1,4,9,721),dtype=torch.float32,device='cuda')).reshape(3,41,576)
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
  candidate=torch.load(candidate_file,map_location='cpu',weights_only=True)
  if candidate['actor_qualified'] is not False or candidate['body_sha256']!=sha256(f) or candidate['temporal_sha256']!=sha256(ck):raise ValueError('candidate lineage')
  with torch.no_grad():weight.copy_(candidate['weight']);model.spatial_contrast.reference.copy_(candidate['reference']);model.spatial_contrast.max_contrast_rms.fill_(candidate['max_contrast_rms'])
  def load_combined(path,expected):
   path=Path(path);r=read(path.parent/'result.json');cfg=read(path.parent/'manifest.json')['config']
   if r['status']!='completed_causal_synthetic_road_with_yaw' or sha256(path)!=expected or sha256(path)!=r['responses_sha256'] or cfg['road_generator_sha256']!=sha256('src/flybrain_drive/synthetic_yaw_road_visual.py'):raise ValueError('verified synthetic source required')
   with np.load(path,allow_pickle=False) as z:ret=z['retina'].copy();motion=z['motion'].copy();times=z['times'].copy();bends=z['bend'].copy();yaws=z['yaw_rate'].copy()
   if ret.shape!=(3,3,41,4,721) or motion.shape!=(3,3,41,4,8,721) or not np.array_equal(bends,[0,-1,1]) or not np.allclose(yaws,[0.,-.12,.12]) or not np.isfinite(ret).all() or not np.isfinite(motion).all():raise ValueError('complete source grid required')
   full=np.concatenate((ret[:,:,:,:,None],motion),axis=4)
   with torch.no_grad():sc=model.scene_pool(torch.tensor(full.reshape(-1,4,9,721),dtype=torch.float32,device='cuda')).reshape(3,3,41,576)
   return sc,times,bends,yaws
  modeldir=Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000');cache=Path('C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-data-windows-v1');os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  from flybrain_drive.retinotopic_readout import column_indices
  camera_path=Path('configs/ac-csp-exterior-front-v3-retina.json');camera=read(camera_path)
  with torch.device('cuda'):
   nc=yaml.safe_load((modeldir/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**nc).to('cuda');network.load_state_dict(torch.load(modeldir/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
   nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{kind}{direction}' for kind in ('T4','T5') for direction in 'abcd']);projector=FlyvisCameraProjector(camera,u,v);road=SyntheticPlanarRoadVisual(camera,curvature=.018,bend_start_m=12.)
  source=B/'20261101-synthetic-multispeed-feedback-1';data=source/'synthetic-training-inputs.npz';power=B/'20261101-synthetic-power-reference-closed-loop-1/power-reference-candidate.pt';newplan=B/'20261101-state-speed-training-plan/plan.json';npn=read(newplan)
  if sha256(data)!=npn['source_inputs_sha256'] or sha256(power)!=npn['fixed_power_candidate_sha256']:raise ValueError('Exact synthetic inputs and diagnostic power reference required')
  power_payload=torch.load(power,map_location='cpu',weights_only=True)
  if power_payload['actor_qualified'] is not False or power_payload['synthetic_only'] is not True:raise ValueError('Synthetic unqualified power candidate required')
  with torch.no_grad():model.motor.backbone.readout.power_reference.copy_(power_payload['power_reference'].to('cuda'))
  from flybrain_drive.flight_state_speed_command import SymmetricFlightStateSpeedCommand
  oldfile=B/'20261101-synthetic-state-speed-1/state-speed-candidate.pt';oldr=read(oldfile.parent/'result.json');newfile=B/'20261102-symmetric-speed-state-fit-1/symmetric-speed-candidate.pt';newr=read(newfile.parent/'result.json')
  if sha256(oldfile)!=oldr['candidate_sha256'] or sha256(newfile)!=newr['candidate_sha256']:raise ValueError('Exact synthetic candidates required')
  old=torch.load(oldfile,map_location='cpu',weights_only=True);new=torch.load(newfile,map_location='cpu',weights_only=True)
  if old['actor_qualified'] is not False or new['actor_qualified'] is not False or new['synthetic_only'] is not True:raise ValueError('Diagnostic candidates only')
  from flybrain_drive.flight_pedal_learning_v2 import bind_registered_reference
  binding=bind_registered_reference(Path(mp['cpu_reference']).read_bytes(),Path(mp['cpu_readout']).read_bytes());registered=binding.readout_state;rd=model.motor.backbone.readout
  if not torch.equal(rd.turn_log_gain.detach().cpu(),torch.as_tensor(registered['turn_log_gain']).cpu()) or not torch.equal(rd.steering_mix.detach().cpu(),torch.as_tensor(registered['steering_mix']).cpu()):raise ValueError('registered gain/mix mismatch')
  original_ref=rd.lateral_reference.detach().clone();registered_ref=torch.as_tensor(registered['lateral_reference'],device='cuda',dtype=original_ref.dtype);original_spatial=model.spatial_contrast.projection.weight.detach().clone()
  plan_file=Path('D:/CodexResearchRuns/flybrain-corner-wind-speed-plan-20261102-01/plan.json');sealed=read(plan_file.parent/'result.json');evaluation=read(plan_file)
  if sealed['status']!='sealed_synthetic_corner_wind_plan' or sealed['plan_sha256']!=sha256(plan_file) or evaluation['program_sha256']!=sha256(__file__) or evaluation['vehicle_sha256']!=sha256('src/flybrain_drive/synthetic_crosswind_vehicle.py') or evaluation['combination_sha256']!=sha256('src/flybrain_drive/unified_airflow_speed.py') or evaluation['unified_sha256']!=sha256('src/flybrain_drive/unified_flight_sensorimotor.py'):raise ValueError('Exact predeclared plan and source required')
  ckpath=Path('D:/CodexResearchRuns/flybrain-synthetic-airflow-fit-20261102-03/synthetic-airflow-feedback.pt')
  if sha256(ckpath)!=evaluation['airflow_checkpoint_sha256']:raise ValueError('Original fixed airflow checkpoint required')
  cp=torch.load(ckpath,map_location='cpu',weights_only=True)
  if cp['actor_qualified'] is not False or cp['source_verified'] is not False or cp['synthetic_only'] is not True:raise ValueError('Synthetic candidate required')
  with torch.no_grad():rd.lateral_reference.copy_(registered_ref)
  model.airflow_feedback=AirflowFeedbackAdapter(len(model.motor.backbone.body_ports)).to('cuda')
  with torch.no_grad():model.airflow_feedback.projection.weight.copy_(cp['weight'].to('cuda'))
  model.airflow_feedback.requires_grad_(False);model.speed_command_boundary=FlightStateSpeedCommand(6).to('cuda');model.speed_command_boundary.load_state_dict(old['model'],strict=True);model.speed_command_boundary.eval().requires_grad_(False)
  results={};arrays={};seed=evaluation['seed']
  for speed in evaluation['initial_speeds_m_s']:
   for bend in evaluation['bends']:
    for wind in evaluation['wind_left_m_s']:
     for mode in ('missing','trained'):
      model.reset_episode();vehicle=PlanarVehicleState(x_m=8.,speed_m_s=speed);last_steer=0.;prior=None;held=None;rows=[];termination='step_budget';error_at_target=None
      with torch.no_grad(),torch.device('cuda'):
       ns=network.steady_state(1.,.005,batch_size=4,value=.5)
       for i in range(evaluation['max_steps']):
        now=1.+i*.04;center=bend*.018*max(vehicle.x_m-12.,0.)**2
        if not 0<=vehicle.x_m<=20. or abs(vehicle.y_left_m-center)>4.:termination='renderer_or_track_boundary';break
        views=road.render(bend=bend,forward_m=vehicle.x_m,lateral_m=vehicle.y_left_m,yaw_rad=vehicle.yaw_left_rad,texture_seed=seed)
        retinal=np.stack([projector.sample(frame) for frame in views]);current=torch.as_tensor(retinal,device='cuda')
        if held is not None:ns=network.simulate(held[:,None,None,:].expand(-1,8,-1,-1),dt=.005,initial_state=ns,as_states=True)[-1]
        motion=ns.nodes.activity[:,indices];scene=model.scene_pool(torch.cat((current[:,None,:],motion),dim=1)[None]);held=current.detach().clone()
        values=np.zeros(21,dtype=np.float64);valid=np.zeros(21,dtype=bool);values[4]=vehicle.speed_m_s*np.tan(last_steer)/2.5;values[16]=vehicle.speed_m_s;values[17:21]=vehicle.speed_m_s/.29;valid[[4,16,17,18,19,20]]=True
        body=NavigationBodyObservation('synthetic-corner-wind',i,now,now,values.copy(),valid.copy(),'synthetic-only');history=(None,)*4 if prior is None else (NavigationBodyObservation('synthetic-corner-wind',i-1,now-.04,now,prior.copy(),valid.copy(),'synthetic-only'),)*4
        ctx=per_view_temporal_body_context(history,body,render_start_lowers=(now,)*4,camera_receipt_time=now,decision_time=now)
        if mode=='missing':air=missing_airflow(now)
        else:
         c,sine=np.cos(vehicle.yaw_left_rad),np.sin(vehicle.yaw_left_rad);body_axes=np.array([[c,sine,0.],[-sine,c,0.],[0.,0.,1.]])
         velocity=np.array([vehicle.speed_m_s*c,vehicle.speed_m_s*sine+wind*evaluation['drift_gain'],0.])
         air=relative_airflow(wind_world_m_s=np.array([0.,wind,0.]),vehicle_world_m_s=velocity,body_axes_world=body_axes,wind_receipt=now,body_receipt=now,decision=now,source_verified=True)
        intent,_=recurrent_speed_airflow_decision(model,decision_time=now,racing_goals=goals,scene_features=scene,scene_valid=view,body_context=ctx,airflow=air)
        raw=intent[0].cpu().numpy();controls=canonical_controls(raw);steer=-float(controls[0])*.35;accel=float(controls[1])*3.-float(controls[2])*6.
        before=vehicle;vehicle=advance_with_crosswind(vehicle,steer_rad=steer,acceleration_m_s2=accel,dt_s=.04,wind_left_m_s=wind,drift_gain=evaluation['drift_gain']);rows.append([now,before.x_m,before.y_left_m,before.yaw_left_rad,before.speed_m_s,float(raw[0]),float(raw[1]),steer,accel,vehicle.x_m,vehicle.y_left_m,vehicle.yaw_left_rad,vehicle.speed_m_s]);last_steer=steer;prior=values
        if before.x_m<evaluation['target_x_m']<=vehicle.x_m:
         fraction=(evaluation['target_x_m']-before.x_m)/(vehicle.x_m-before.x_m);target_y=before.y_left_m+fraction*(vehicle.y_left_m-before.y_left_m);target_center=bend*.018*max(evaluation['target_x_m']-12.,0.)**2;error_at_target=target_y-target_center;termination='target_x';break
        if i%20==19:write_json(run/'progress.json',dict(speed=speed,bend=bend,wind=wind,mode=mode,step=i+1,completed_conditions=len(results),total_conditions=16))
      key=f'{int(speed)}_{bend}_{int(wind)}_{mode}';arrays[key]=np.asarray(rows);results[key]=dict(steps=len(rows),termination=termination,error_at_target=error_at_target,final_speed=vehicle.speed_m_s,final_position=[vehicle.x_m,vehicle.y_left_m])
  comparisons=[]
  for speed in evaluation['initial_speeds_m_s']:
   for bend in evaluation['bends']:
    for wind in evaluation['wind_left_m_s']:
     a=results[f'{int(speed)}_{bend}_{int(wind)}_missing'];b=results[f'{int(speed)}_{bend}_{int(wind)}_trained'];valid=a['error_at_target'] is not None and b['error_at_target'] is not None
     comparisons.append(dict(initial_speed=speed,bend=bend,wind_left_m_s=wind,missing_error=a['error_at_target'],trained_error=b['error_at_target'],both_reached=valid,improved=valid and abs(b['error_at_target'])<abs(a['error_at_target'])))
  out=run/'trajectory.npz';np.savez_compressed(out,**arrays)
  result.update(status='completed_synthetic_corner_wind_speed',predeclared_pass=all(x['improved'] for x in comparisons),comparisons=comparisons,conditions=results,output_sha256=sha256(out),plan_sha256=sha256(plan_file),checkpoint_sha256=sha256(ckpath),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,interpretation='Synthetic corner and slip-free bicycle with heuristic lateral drift and model-driven acceleration; not AC physics, physiology or actual driving')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_synthetic_corner_wind_speed':raise SystemExit(1)
if __name__=='__main__':main()
