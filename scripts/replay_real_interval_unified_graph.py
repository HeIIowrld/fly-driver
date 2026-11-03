"""Offline single-flight-graph replay of audited real low-speed four-view/body history."""
import argparse,base64,bisect,hashlib,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_geometry_fresh_hold import FreshEvolvingGeometryPort
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.interval_yaw_visual_boundary import IntervalYawVisualBoundary
from flybrain_drive.native_angular_history import NativeAngularHistory
from flybrain_drive.view_yaw_interval import prepare_view_yaw_interval
from flybrain_drive.virtual_body import VirtualBodyEncoder
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,NavigationBodyObservation,decode_navigation_body_prefix
from flybrain_drive.visual_self_motion import VisualSelfMotionPool

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
p=argparse.ArgumentParser(description=__doc__)
for n in ('probe','visual','host-audit','body-audit','output'):p.add_argument('--'+n,required=True)
a=p.parse_args();root=Path(a.probe);visualroot=Path(a.visual);ha=Path(a.host_audit);ba=Path(a.body_audit);base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
source,vr,h,b=map(read,(root/'result.json',visualroot/'result.json',ha/'result.json',ba/'result.json'));ehash=sha256(root/'events.jsonl')
if (source['status']!='completed_low_speed_pulse_diagnostic' or vr['status']!='completed_audited_low_speed_visual_replay'
 or h['status']!='completed_camera_native_host_audit' or b['status']!='completed_low_speed_raw_audit'
 or read(visualroot/'manifest.json')['config']['events_sha256']!=ehash
 or read(ha/'manifest.json')['config']['events_sha256']!=ehash or read(ba/'manifest.json')['config']['source_events_sha256']!=ehash
 or sha256(visualroot/'responses.npz')!=vr['responses_sha256'] or not torch.cuda.is_available()):raise ValueError('Exact audited original history and CUDA required')
events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines() if json.loads(x)['kind']=='native']
with np.load(visualroot/'responses.npz',allow_pickle=False) as f:clocks=f['clock'].copy();retina=f['retina'].copy();motion=f['motion'].copy()
if len(events)!=len(clocks) or retina.shape!=(len(events),4,721) or motion.shape!=(len(events),8,8,721):raise ValueError('Complete original vision/body history required')
start=next(i for i,e in enumerate(events) if e.get('camera') is not None)
if any(e.get('camera') is None for e in events[start:]) or np.any(np.diff(clocks[start:])>.08):raise ValueError('Explicit contiguous valid camera and <=80ms graph timing required')
training=base/'20261029-t1437-embodied-joint-training';opponent=base/'20261031-opponent-temporal-graph-trained';context=base/'20261031-historical-train00-temporal-context-onepass';fit=base/'20261030-body-feedback-brake-steering-anchored';geometry=base/'20261101-unified-synthetic-geometry-yaw-last3-fit-30';plan=read(base/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];tr,op,cr,br=map(read,(training/'result.json',opponent/'result.json',context/'result.json',fit/'result.json'))
if (sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opponent/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(context/'context-boundary.pt')!=cr['checkpoint_sha256'] or sha256(fit/'body-feedback-adapter.pt')!=br['checkpoint_sha256'] or sha256(geometry/'geometry-visual-boundary.pt')!='093345b9d78b516a3c55804714e1a446fef057e7d4271d5e331aa5d67aae896c'):raise ValueError('Pinned graph and adapter lineage required')
run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_events_sha256=ehash,visual_sha256=vr['responses_sha256'],body_audit_sha256=sha256(ba/'result.json'),host_audit_sha256=sha256(ha/'result.json'),identity_source_sha256=sha256(base/'20261101-engine-return-native-sample/result.json'),identity_scope='prior same-session evidence, not per-sample static identity',start_index=start,excluded_before_start=start,clock='host perf counter, not exposure or physical source time',geometry_reference='last3_of_first11',command='no external yaw target; neutral speed error from current body speed; diagnostic only', geometry_hold='same capture, evolving motion activity only while host receipt <=50ms; diagnostic hypothesis',optimizer_updates=0,actor_deployed=False))
identity_source=base/'20261101-engine-return-native-sample/result.json';identity=read(identity_source)['identity'];
if read(identity_source)['status']!='completed' or events[start]['camera']['generation']!=read(base/'20261101-steering-contract-preflight/result.json')['session']['generation']:raise ValueError('Prior session identity evidence required')
encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);bodies=[];native=[]
for i,e in enumerate(events):
 n=e['native'];raw=base64.b64decode(n['prefix_base64'],validate=True)
 if hashlib.sha256(raw).hexdigest()!=n['prefix_sha256']:raise ValueError('Native prefix changed')
 sample=decode_navigation_body_prefix(raw,float(clocks[i]));native.append(decode_physics_prefix(raw,float(clocks[i])))
 bodies.append(encoder.encode(sample,episode_id='diagnostic',decision_monotonic_seconds=float(clocks[i])))
torch.set_num_threads(1);torch.manual_seed(863009);spec=read(training/'manifest.json')['config']['plan'];src=read(Path(spec['source'])/'result.json')
visual=EmbodiedVisualFlightCircuit(spec['source_spec'],torch.load(src['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu');visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
physical=PhysicalManeuverCircuit(plan['spec'],torch.load(plan['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu');temporal=OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,visual.backbone.scene_command.scale,len(visual.backbone.vision_ports));temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model']);boundary=PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports));boundary.load_state_dict(torch.load(context/'context-boundary.pt',map_location='cpu',weights_only=True)['model']);model=UnifiedFlightSensorimotor(physical,visual.backbone,temporal,boundary)
model.body_feedback.load_state_dict(torch.load(fit/'body-feedback-adapter.pt',map_location='cpu',weights_only=True)['model']);model.motor.backbone.circuit.sparse_backend='cached_csr_gpu';model=model.to('cuda').eval().requires_grad_(False)
camera=read('configs/ac-csp-yaw-retina.json');uv=np.array([(q,r) for q in range(-15,16) for r in range(max(-15,-15-q),min(15,15-q)+1)]);projector=FlyvisCameraProjector(camera,uv[:,0],uv[:,1]);pool=VisualSelfMotionPool(camera,projector.directions_body)
model.view_context=FreshEvolvingGeometryPort(model.view_context,pool,len(model.motor.backbone.vision_ports)).to('cuda');weights=torch.load(geometry/'geometry-visual-boundary.pt',map_location='cpu',weights_only=True)['model']['weight'].to('cuda')
command=torch.tensor([[0.,0.]],device='cuda');goals=torch.tensor(racing_goal(100,1),device='cuda');viewmask=torch.ones(1,4,dtype=torch.bool,device='cuda');allout=[];readiness=[];missing_counts=[];geometry_yaw=[]
digest=sha256('configs/ac-virtual-body-v1.json');body_encoder=VirtualBodyEncoder('configs/ac-virtual-body-v1.json',native_identity=identity);model.interval_yaw_boundary=IntervalYawVisualBoundary(len(model.motor.backbone.vision_ports),body_contract_sha256=digest,device='cuda');torch.manual_seed(864023);probe_weight=torch.randn_like(model.interval_yaw_boundary.projection.weight)*.1;outputs={};states={};valid_counts={};reason_counts={}
for condition in ('legacy','neutral','missing','probe'):
 model.reset_episode();history=NativeAngularHistory(capacity=1024);out=[];state=[];counts=np.zeros(4,dtype=int);reasons={};previous_capture=None
 with torch.no_grad():
  model.view_context.projection.weight.copy_(weights);model.interval_yaw_boundary.projection.weight.copy_(probe_weight if condition in ('missing','probe') else torch.zeros_like(probe_weight))
  for j in range(start):
   if float(events[j]['native']['received_monotonic_seconds'])!=float(clocks[j]):raise ValueError('original first native receipt required')
   history.offer(native[j])
  for i in range(start,len(events)):
   event=events[i];cam=event['camera'];now=float(clocks[i]);current=bodies[i]
   if float(event['native']['received_monotonic_seconds'])!=now:raise ValueError('original first native receipt required')
   history.offer(native[i])
   if [v['view_id'] for v in cam['views']]!=['front','left','right','rear']:raise ValueError('original ordered views required')
   prior=[];bounds=[]
   for v in cam['views']:
    lower=float(v['render_started_lower']);bounds.append(lower);j=bisect.bisect_left(clocks,lower,0,i)-1
    if j<0 or cam['received']-clocks[j]>.05:prior.append(None)
    else:
     x=bodies[j];prior.append(NavigationBodyObservation(x.episode_id,x.packet_id,x.original_received_monotonic_seconds,float(cam['received']),x.values.copy(),x.valid.copy(),x.contract_sha256))
   ctx=per_view_temporal_body_context(tuple(prior),current,render_start_lowers=tuple(bounds),camera_receipt_time=float(cam['received']),decision_time=now)
   frame=np.concatenate((retina[i,:,None],motion[i,:4]),axis=1);scene=model.scene_pool(torch.as_tensor(frame[None],dtype=torch.float32,device='cuda'))
   model.view_context.observe(motion[i,:4],np.ones(4,dtype=bool),native[i],capture_id=int(cam['capture_id']),decision_time=now,camera_receipt_time=float(cam['received']))
   intervals=[None]*4;capture=int(cam['capture_id'])
   if previous_capture is not None and capture==previous_capture+1 and model.previous_render_lowers is not None:
    for v in range(4):
     try:
      intervals[v]=prepare_view_yaw_interval(history,body_encoder,previous_render_lower=model.previous_render_lowers[v],current_render_lower=bounds[v],camera_receipt=float(cam['received']),decision=now);counts[v]+=1
     except ValueError as error:reasons[str(error)]=reasons.get(str(error),0)+1
   if condition=='legacy':intent,_=model.sensory_decision(decision_time=now,racing_goals=goals,scene_features=scene,scene_valid=viewmask,body_context=ctx)
   else:intent,_=model.experimental_sensory_interval_decision(decision_time=now,racing_goals=goals,scene_features=scene,scene_valid=viewmask,body_context=ctx,interval_yaw=tuple(intervals) if condition!='missing' else (None,)*4)
   out.append(intent.cpu().numpy()[0].copy());state.append(model.state.cpu().numpy()[0].copy());previous_capture=capture
 outputs[condition]=np.stack(out);states[condition]=np.stack(state);valid_counts[condition]=counts.tolist();reason_counts[condition]=reasons;write_json(run/'progress.json',dict(completed_condition=condition))
refroot=base/'20261101-fast-body-fourview-graph-sensory-neutral';rr=read(refroot/'result.json');rf=refroot/'intents.npz'
if rr['status']!='completed_low_speed_unified_graph_diagnostic' or sha256(rf)!=rr['intents_sha256']:raise ValueError('exact historical baseline required')
with np.load(rf,allow_pickle=False) as z:reference=z['intents'][0].copy();original_clocks=z['clocks'].copy()
np.testing.assert_array_equal(original_clocks,clocks[start:]);prior_error=float(np.max(np.abs(reference-outputs['legacy'])));neutral_error=float(np.max(np.abs(outputs['neutral']-outputs['legacy'])));neutral_state=float(np.max(np.abs(states['neutral']-states['legacy'])));missing_error=float(np.max(np.abs(outputs['missing']-outputs['legacy'])));missing_state=float(np.max(np.abs(states['missing']-states['legacy'])))
if max(prior_error,neutral_error,neutral_state,missing_error,missing_state)>2e-5:raise ValueError('baseline neutral or missing mismatch')
out=run/'interval-graph.npz';np.savez_compressed(out,**{k+'_intents':v for k,v in outputs.items()},**{k+'_states':v for k,v in states.items()},clocks=clocks[start:]);result=dict(status='completed_real_interval_graph_offline_replay',decisions=len(reference),valid_by_view=valid_counts,unavailable_reasons=reason_counts,prior_intent_max_abs=prior_error,neutral_intent_max_abs=neutral_error,neutral_state_max_abs=neutral_state,missing_intent_max_abs=missing_error,missing_state_max_abs=missing_state,random_probe_intent_max_abs=float(np.max(np.abs(outputs['probe']-outputs['legacy']))),random_probe_state_max_abs=float(np.max(np.abs(states['probe']-states['legacy']))),output_sha256=sha256(out),physical_time_verified=False,actor_qualified=False,training_admissible=False,optimizer_updates=0,vehicle_control_sent=False,interpretation='Previously observed real low-speed frames and original host receipts, fixed graph, random untrained probe, no physiology or driving proof')
write_json(run/'result.json',result);print(json.dumps(result))
