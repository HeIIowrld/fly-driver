"""Prepare actor-only arrays from an independently qualified complete episode.

Replay original per-view capture events through frozen author FlyVis dynamics.
Absolute clocks are audit indices, never neural inputs. This script has no
simulator, lifecycle, vJoy, actuator, teacher-observation or map input API.
"""
import argparse,base64,copy,io,json,os,time,yaml
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.camera_body_alignment import source_render_witness
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import assemble_student,read
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix
from scripts.audit_synchronized_sac_demonstration import native_samples

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',required=True);p.add_argument('--qualification',required=True);p.add_argument('--identity',default='runs/teacher-synchronized-20261102/native-static-identity-inline.json');p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.recording);cert=Path(a.qualification);qr=read(cert/'result.json')
 if qr['status']!='qualified_bounded_camera_body_alignment' or not qr['training_admissible'] or qr['full_lap_label_coverage']!=1:
  raise ValueError('Complete independently qualified camera/body episode required before preprocessing')
 qm=read(cert/'manifest.json')['config']
 for name,key in [('result.json','source_result_sha256'),('camera/result.json','camera_result_sha256'),('teacher/result.json','teacher_result_sha256')]:
  if sha256(root/name)!=qm[key]:raise ValueError('Qualified source result changed')
 if sha256(cert/'pairings.json')!=qr['pairings_sha256']:raise ValueError('Qualified label pairings changed')
 if sha256(cert/'lap-proof/result.json')!=qr['lap_proof_sha256']:raise ValueError('Qualified full lap proof changed')
 pairings=read(cert/'pairings.json');max_needed_camera=max(p['camera_index'] for p in pairings if p['eligible'])
 lap_proof=read(cert/'lap-proof/result.json');full_start=lap_proof['crossings'][0]['host_time'];full_end=lap_proof['crossings'][-1]['host_time']
 cfg=read('configs/flight-stream-batched-v1-20261029.json');ccpath=Path('configs/ac-csp-exterior-front-v3-retina.json');cc=read(ccpath)
 modeldir=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root'])
 if sha256(modeldir/'best_chkpt')!=cfg['author_weights_sha256']:raise ValueError('Frozen author FlyVis weights changed')
 run=begin_run(a.output,dict(arguments=vars(a),seed=780204,program_sha256=sha256(__file__),
  native_identity_sha256=sha256(a.identity),student_fit_plan_sha256=sha256('runs/teacher-synchronized-20261102/student-fit-plan-v2.json'),qualification_sha256=sha256(cert/'result.json'),pairings_sha256=qr['pairings_sha256'],
  camera_configuration_sha256=sha256(ccpath),author_weights_sha256=cfg['author_weights_sha256'],
  representation='Frozen measured visual topology/dynamics with author-trained parameters; engineering camera lattice',
  actor_fields=['four-view pooled intensity/T4/T5 576','body21 plus original masks and per-view contexts'],
  forbidden_actor_fields=['world pose','absolute heading','map','teacher observation','racing line','absolute timestamp'],
  optimizer_updates=0,actuator_messages_sent=0,exact_exposure_timestamp_verified=False))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False)
 try:
  identity_meta=read(a.identity);identity=identity_meta['identity']
  if identity_meta['window_handle']!=read(root/'camera/ownership-plan.json')['window_handle']:raise ValueError('Static identity witness came from another engine window')
  if identity['car_model']!='bmw_z4_gt3' or identity['track']!='monza':raise ValueError('Verified source identity differs')
  torch.set_num_threads(1)
  if not torch.cuda.is_available():raise RuntimeError('Working existing GPU runtime required; do not bypass Windows policy')
  model,lineage=assemble_student();pool=copy.deepcopy(model.scene_pool).to('cuda').eval().requires_grad_(False);del model
  os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  with torch.device('cuda'):
   netcfg=yaml.safe_load((modeldir/'_meta.yaml').read_text(encoding='utf8'))['config']['network']
   network=flyvis.Network(**netcfg).to('cuda');network.load_state_dict(torch.load(modeldir/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
   nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors]
   indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{d}' for k in ('T4','T5') for d in 'abcd'])
   projector=FlyvisCameraProjector(cc,u,v)
   state=network.steady_state(1.,.005,batch_size=4,value=.5)
   held=torch.full((4,len(u)),.5,device='cuda')
  camera=read(root/'camera/result.json')
  if camera['camera_archive']['bundle_sha256']!=qr['camera_bundle_sha256'] or camera['native_body_sha256']!=qr['native_body_sha256']:raise ValueError('Qualified raw hashes changed')
  body,body_times=native_samples(root/'camera/native-body.jsonl',qr['native_body_sha256'])
  episode_id=read(root/'camera/ownership-plan.json')['request_id']
  body_cfg=read('configs/ac-virtual-navigation-body-v2.json');
  contract_hash=sha256('configs/ac-virtual-navigation-body-v2.json')
  # Each native packet is encoded at its own first receipt, in original order.
  encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity)
  observations=[]
  for b in body:
   sample=decode_navigation_body_prefix(b['prefix'],b['receipt'])
   observations.append(encoder.encode(sample,episode_id=episode_id,decision_monotonic_seconds=b['receipt']))
  scenes=[];rows=[];previous=None;renders=[];history=[];frame_admitted=[];skipped_visual_frames=[]
  with CameraArchive(root/'camera/vision') as archive:
   clock=archive.json('clock-initial.json')
   stream=[json.loads(x) for x in archive.read('stream.jsonl').decode('utf8').splitlines()]
   for i,row in enumerate(stream[:max_needed_camera+1]):
    frame=archive.json(Path(row['file']).name);byname={v['view_id']:v for v in frame['views']};records={v['view_id']:v for v in row['views']}
    hs=[int(np.searchsorted(body_times,records[n]['render_started_lower'],side='left')-1) for n in VIEW_ORDER]
    admitted=True
    for name,j in zip(VIEW_ORDER,hs,strict=True):
     try:
      source_render_witness(byname[name])
      upper=byname[name]['render_started_clock']+clock['offset_upper_seconds']+clock.get('host_clock_quantization_seconds',0.)
      if j<0 or not 0<=upper-body[j]['first_copy_start']<=.025:raise ValueError('No original preceding body bracket')
     except ValueError:admitted=False
    rows.append(row);renders.append([records[n]['render_started_lower'] for n in VIEW_ORDER]);history.append(hs);frame_admitted.append(admitted)
    if not admitted:
     # Invalid warmup pixels never enter even the frozen visual recurrence.
     # These placeholders cannot be selected by the actor data join below.
     scenes.append(np.zeros(576,dtype=np.float32));skipped_visual_frames.append(i);continue
    events=sorted((records[name]['render_started_lower'],j,name) for j,name in enumerate(VIEW_ORDER))
    for stamp,j,name in events:
     if previous is not None:
      elapsed=stamp-previous
      if elapsed<=0:raise ValueError('Original per-view capture times do not advance')
      for dt,count in held_input_batches(elapsed):
       state=network.simulate(held[:,None,None,:].expand(-1,count,-1,-1),dt=dt,initial_state=state,as_states=True)[-1]
     rgb=np.asarray(Image.open(io.BytesIO(archive.read(byname[name]['file']))).convert('RGB')).copy()
     held[j]=torch.as_tensor(projector.sample(rgb).copy(),device='cuda');previous=stamp
    scenes.append(pool(torch.cat([held[:,None,:],state.nodes.activity[:,indices]],dim=1)[None]).cpu().numpy()[0])
    if i%100==0:write_json(run/'progress.json',dict(frames=i,total=max_needed_camera+1,state_reset_count=1,invalid_pixels_used=0))
  events=[json.loads(x) for x in (root/'teacher/events.jsonl').read_text(encoding='utf8').splitlines()]
  requests=[e for e in events if e['kind']=='teacher_request'];pairings=read(cert/'pairings.json')
  if len(requests)!=len(pairings):raise ValueError('Teacher pairing count differs')
  # Start at the first continuously eligible decision. Never silently reset an
  # episode around missing data or splice across an ineligible internal gap.
  selected=np.flatnonzero([p['eligible'] for p in pairings]);start,end=int(selected[0]),int(selected[-1])+1
  if not all(p['eligible'] for p in pairings[start:end]):raise ValueError('Internal camera/body gap; no episode splice allowed')
  received=np.asarray([row['received'] for row in rows])
  event_by_time={float(requests[k]['decision_started_host_time']):k for k in range(start,end)}
  first_time,last_time=min(event_by_time),max(event_by_time)
  last_body_tick=None
  for t in body_times:
   t=float(t)
   if first_time<=t<=last_time and (last_body_tick is None or t-last_body_tick>=.01):
    event_by_time.setdefault(t,None);last_body_tick=t
  output={k:[] for k in ['scene','body','body_valid','view_body','view_body_valid','decision_time',
   'body_receipt','view_receipts','render_lowers','camera_receipt','labels','camera_index',
   'body_packet','view_packets','fit_mask','label_valid','teacher_request_index']}
  for decision,request_index in sorted(event_by_time.items()):
   i=int(np.searchsorted(received,decision,side='right')-1)
   j=int(np.searchsorted(body_times,decision,side='right')-1)
   if (min(i,j)<0 or not frame_admitted[i] or decision-body[j]['receipt']>.05
       or not all(0<=decision-lower<=.15 for lower in renders[i])):
    raise ValueError('Original native body tick lacks a qualified causal four-view scene; no splice or reset')
   obs=observations[j];hs=history[i]
   if min(hs)<0:raise ValueError('Missing independently qualified prior view context')
   hobs=[observations[k] for k in hs]
   fit=False;labels=[0.,0.,0.]
   if request_index is not None:
    request=requests[request_index];pair=pairings[request_index]
    if pair['camera_index']!=i or pair['body_index']!=j:raise ValueError('Original certificate label join changed')
    fit=full_start<=request['host_time']<=full_end
    if fit and request['unlimited']!=request['limited']:raise ValueError('Only unmodified unrestricted teacher controls may supervise a full lap')
    control=request['request']['requested_control'];labels=[control['steering']/.6,control['throttle'],control['brake']]
   values=dict(scene=scenes[i],body=obs.values,body_valid=obs.valid,
    view_body=[o.values for o in hobs],view_body_valid=[o.valid for o in hobs],
    decision_time=decision,body_receipt=obs.original_received_monotonic_seconds,
    view_receipts=[o.original_received_monotonic_seconds for o in hobs],render_lowers=renders[i],
    camera_receipt=rows[i]['received'],labels=labels,camera_index=i,body_packet=obs.packet_id,
    view_packets=[o.packet_id for o in hobs],fit_mask=fit,label_valid=request_index is not None,
    teacher_request_index=request_index if request_index is not None else -1)
   for key,value in values.items():output[key].append(value)
  output={k:np.asarray(v) for k,v in output.items()}
  if not np.all((np.diff(output['decision_time'])>0)&(np.diff(output['decision_time'])<=.08)):raise ValueError('Original decision cadence outside existing model envelope')
  if np.max(output['decision_time'][1:]-output['body_receipt'][:-1])>.05:raise ValueError('Original held body age still exceeds50ms')
  target=run/'actor-only.npz';np.savez_compressed(target,**output)
  write_json(run/'contract.json',dict(episode_id=episode_id,body_contract_sha256=contract_hash,lineage=lineage,
   source_start_request=start,source_end_request=end,source_continuous_replay=True,flyvis_episode_resets=1,
   flyvis_prior='Author steady .5 at first qualified visual frame; unqualified warmup pixels never enter recurrence',
   unqualified_pixel_inputs=0,skipped_visual_frames=skipped_visual_frames,
   time_domain='Original source QPC audit clocks, excluded from network input',
   body_tick_policy='Original native first-copy receipts >=10ms apart, union unchanged teacher decision timestamps; native-only ticks have explicit absent-label mask',
   source_qualification_sha256=sha256(cert/'result.json'),bounded_alignment_only=True))
  result.update(status='completed_qualified_actor_preprocessing',arrays_sha256=sha256(target),
   contract_sha256=sha256(run/'contract.json'),model_body_teacher_events=len(output['labels']),frames=len(scenes),
   unqualified_pixel_inputs=0,skipped_visual_frames=skipped_visual_frames,
   body_masks_valid_counts=output['body_valid'].sum(0).tolist(),
   qualified_full_lap_supervised_decisions=int(output['fit_mask'].sum()),
   native_only_unlabelled_ticks=int((~output['label_valid']).sum()),
   warmup_teacher_labels_used_only_for_burn_in=int((output['label_valid']&~output['fit_mask']).sum()),
   maximum_held_body_age_ms=float(np.max(output['decision_time'][1:]-output['body_receipt'][:-1])*1000),
   teacher_label_ranges=[output['labels'].min(0).tolist(),output['labels'].max(0).tolist()],
   physiological_mapping_verified=False,student_autonomy_verified=False)
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 write_json(run/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
