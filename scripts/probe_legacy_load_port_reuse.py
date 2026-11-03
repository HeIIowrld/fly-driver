"""Offline recorded FlyVis plus original body into one recurrent circuit; no actuation."""
import argparse,base64,hashlib,json,os,time
from dataclasses import replace
from pathlib import Path
import numpy as np,torch
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix
from scripts.control_ac_session import read_session_heartbeat
B=Path(os.environ['LOCALAPPDATA'])/'flybrain/runtime-runs'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();source=B/'20261102-isolated-retina-probe-1';vision=B/'20261102-isolated-stationary-flyvis-1';sr=read(source/'result.json');vr=read(vision/'result.json');events=read(source/'events.json')
 if sr['status']!='completed_stationary_owned_mmf_resume_diagnostic' or vr['status']!='completed_offline_flyvis_stationary_replay' or sha256(source/'events.json')!=sr['events_sha256'] or sha256(vision/'responses.npz')!=vr['response_sha256']:raise ValueError('original verified diagnostic sources required')
 vm=read(vision/'manifest.json')['config'];before=read(source/'before.json')['session'];current,_,_=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
 if vm['source_events_sha256']!=sr['events_sha256'] or before['generation']!=current['generation'] or before['car_id']!=current['car_id']:raise ValueError('source lineage or same current session missing')
 with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
 if identity['car_model']!=before['car_id']:raise ValueError('native and source vehicle mismatch')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(source/'events.json'),vision_sha256=sha256(vision/'responses.npz'),native_identity=identity,native_identity_observed_after_capture=True,training=False,actor=False,source_time_verified=False));result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,training=False,source_time_verified=False)
 try:
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  torch.set_num_threads(1);torch.manual_seed(863009);trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
  if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage differs')
  spec=read(trdir/'manifest.json')['config']['plan'];upstream=read(Path(spec['source'])/'result.json');visual=EmbodiedVisualFlightCircuit(spec['source_spec'],torch.load(upstream['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu');visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
  mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];motor=PhysicalManeuverCircuit(mp['spec'],torch.load(mp['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu');temporal=OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,visual.backbone.scene_command.scale,len(visual.backbone.vision_ports));temporal.load_state_dict(torch.load(opdir/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model']);boundary=PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports));boundary.load_state_dict(torch.load(ctxdir/'context-boundary.pt',map_location='cpu',weights_only=True)['model'])
  model=UnifiedFlightSensorimotor(motor,visual.backbone,temporal,boundary,body_feedback=CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports)));model.motor.backbone.circuit.sparse_backend='cached_csr_gpu';model=model.to('cuda').eval().requires_grad_(False)
  fit=B/'20261101-speed-wheel-turn-control-fit';fr=read(fit/'result.json');bodyfile=fit/'speed-body-adapter.pt'
  if fr['status']!='completed_synthetic_speed_wheel_turn_control_training' or sha256(bodyfile)!=fr['checkpoint_sha256']:raise ValueError('body adapter lineage differs')
  payload=torch.load(bodyfile,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False:raise ValueError('candidate qualification differs')
  model.body_feedback.load_state_dict(payload['model'],strict=True)
  with np.load(vision/'responses.npz',allow_pickle=False) as z:ret=z['retina'].copy();motion=z['motion'].copy();clocks=z['clock'].copy()
  if ret.shape!=(len(events),4,721) or motion.shape!=(len(events),8,8,721) or not np.array_equal(clocks,np.array([x['decision'] for x in events])):raise ValueError('ordered original visual data differs')
  scene_actual=np.concatenate((ret[:,:,None,:],motion[:,:4]),axis=2);scene_held=np.concatenate((np.broadcast_to(ret[:1],ret.shape)[:,:,None,:],motion[:,4:]),axis=2);goals=torch.as_tensor(racing_goal(100,1),device='cuda');valid=torch.ones((1,4),dtype=torch.bool,device='cuda');out={}
  legacy_dir=B/'20261030-body-feedback-brake-steering-anchored';legacy_file=legacy_dir/'body-feedback-adapter.pt';legacy_result=read(legacy_dir/'result.json');legacy=torch.load(legacy_file,map_location='cpu',weights_only=True)
  if legacy_result['status']!='completed_brake_steering_anchored_body_fit' or sha256(legacy_file)!=legacy_result['checkpoint_sha256'] or legacy['actor_qualified'] is not False:raise ValueError('verified old candidate required')
  if legacy['model']['indices'].tolist()!=[0,1,8,9,10,11,12,13,14,15,17,18,19,20] or model.body_feedback.indices.tolist()!=[0,1,8,9,10,11,12,13,14,15,16,17,18,19,20]:raise ValueError('exact feature ordering required')
  oldweights=legacy['model']['projection.weight'];baseweights=model.body_feedback.projection.weight.detach().clone()
  if oldweights.shape!=(201,28) or baseweights.shape!=(201,30) or torch.count_nonzero(baseweights[:,2:6]).item()!=0:raise ValueError('specific old and current weight layouts required')
  if sha256(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/actual-fit/final-two-DNg-scalars.json')!=legacy['source_pedal_checkpoint_sha256']:raise ValueError('previous physical graph binding required')
  conditions=[('actual',scene_actual),('held_first',scene_held),('current_shift',scene_actual),('legacy_load',scene_actual),('legacy_load_shift',scene_actual)]
  for condition,scene in conditions:
   with torch.no_grad():
    model.body_feedback.projection.weight.copy_(baseweights)
    if condition.startswith('legacy_load'):model.body_feedback.projection.weight[:,2:6].copy_(oldweights[:,2:6].to('cuda'))
   model.reset_episode();encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);intents=[];states=[];masks=[];start=time.perf_counter()
   with torch.no_grad():
    for i,event in enumerate(events):
     raw=base64.b64decode(event['native_prefix_base64'],validate=True)
     if hashlib.sha256(raw).hexdigest()!=event['native_prefix_sha256']:raise ValueError('native prefix mismatch')
     sample=decode_navigation_body_prefix(raw,event['body_received']);when=float(event['decision']);body=encoder.encode(sample,episode_id='original-stationary',decision_monotonic_seconds=when)
     if body.packet_id!=event['body_packet']:raise ValueError('native identity mismatch')
     if condition.endswith('_shift'):
      if not body.valid[8:12].all():raise ValueError('valid original loads required')
      values=body.values.copy();values[8:10]+=.025;values[10:12]-=.025
      if min(values[8:12])<0 or abs(sum(values[8:12])-1.)>1e-5:raise ValueError('bounded total-preserving hypothetical load required')
      body=replace(body,values=values)
     context=per_view_temporal_body_context((None,)*4,body,render_start_lowers=tuple(v['render_started_lower'] for v in event['views']),camera_receipt_time=event['camera_received'],decision_time=when)
     features=model.scene_pool(torch.as_tensor(scene[i:i+1].copy(),dtype=torch.float32,device='cuda'));intent,_=model.sensory_decision(decision_time=when,racing_goals=goals,scene_features=features,scene_valid=valid,body_context=context);intents.append(intent.cpu().numpy()[0].copy());states.append(model.state.cpu().numpy()[0].copy());masks.append(body.valid.copy())
   out[condition+'_intent']=np.stack(intents);out[condition+'_state']=np.stack(states);out[condition+'_body_valid']=np.stack(masks);out[condition+'_seconds']=np.array(time.perf_counter()-start)
  previous=B/'20261102-isolated-unified-replay-1';previous_result=read(previous/'result.json')
  if previous_result['status']!='completed_offline_unified_stationary_replay' or sha256(previous/'responses.npz')!=previous_result['response_sha256']:raise ValueError('verified previous replay required')
  with np.load(previous/'responses.npz',allow_pickle=False) as old:
   repeat_diff=max(float(np.max(np.abs(out[k]-old[k]))) for k in ('actual_intent','actual_state','held_first_intent','held_first_state'))
  if repeat_diff>2e-5:raise ValueError('baseline reproducibility tolerance exceeded: '+str(repeat_diff))
  effects={name:dict(intent_max=float(np.max(np.abs(out[a+'_intent']-out[b+'_intent']))),state_max=float(np.max(np.abs(out[a+'_state']-out[b+'_state'])))) for name,a,b in [('current_shift','actual','current_shift'),('transplant_baseline','actual','legacy_load'),('transplant_shift','legacy_load','legacy_load_shift')]}
  path=run/'responses.npz';np.savez_compressed(path,**out);delta=out['actual_intent']-out['held_first_intent'];result.update(status='completed_offline_unified_stationary_replay',frames=len(events),response_sha256=sha256(path),previous_replay_sha256=sha256(previous/'responses.npz'),replay_max_difference=repeat_diff,load_interventions=effects,legacy_checkpoint_sha256=sha256(legacy_file),load_shift_fraction=0.05,intent_difference_max=float(np.abs(delta).max()),state_difference_max=float(np.abs(out['actual_state']-out['held_first_state']).max()),body_valid_min=int(out['actual_body_valid'].sum(axis=1).min()),body_valid_max=int(out['actual_body_valid'].sum(axis=1).max()),historical_body_views='explicitly missing; no interpolation',native_identity_observed_after_capture=True,interpretation='Offline replay of stationary source; vision and body in one circuit; no control, physical synchronization or driving qualification')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result))
 if result['status']!='completed_offline_unified_stationary_replay':raise SystemExit(1)
if __name__=='__main__':main()
