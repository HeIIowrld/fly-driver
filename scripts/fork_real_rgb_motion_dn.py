"""Original real RGB prefix, identical full FlyVis forks, recipient capture clocks.

Offline retimed image intervention only. No physical counterfactual action,
new sensor authority, neural physiology, fitting or vehicle control is claimed.
"""
import copy,gc,hashlib,io,json,os,time,yaml
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.motion_per_view_flyvis_stream import MotionPerViewFlyvisStream
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import read
from scripts.motion_spatial_experiment import load_motion_experiment

ROOT=Path('runs/spatial-motion-dn-learning-20261104')
def state_hash(value):
 h=hashlib.sha256();count=0
 def visit(v,path):
  nonlocal count
  h.update(path.encode())
  if isinstance(v,torch.Tensor):
   a=v.detach().cpu().contiguous().numpy();h.update(str((str(v.dtype),a.shape)).encode());h.update(a.tobytes());count+=1
  elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes());count+=1
  elif hasattr(v,'items'):
   for k,x in sorted(v.items(),key=lambda kv:str(kv[0])):visit(x,path+'/'+str(k))
  elif isinstance(v,(list,tuple)):
   for k,x in enumerate(v):visit(x,path+'/'+str(k))
  elif v is None or isinstance(v,(int,float,bool,str,np.number)):h.update(repr(v).encode())
  else:raise TypeError('Unhashed visual state type '+str(type(v)))
 visit(value,'state');assert count>0;return h.hexdigest(),count

@torch.no_grad()
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');base=read(p['base_port_plan']);jp=read(base['parent_fit_plan']);candidate=read(jp['candidate_plan']);mf=Path(candidate['motion']['fit']);cfg=read(mf/'manifest.json')['config'];components,_,entries,items,extras,_,_,_,_,features,maps,_,_,_=load_motion_experiment(cfg['plan'],'fine_motion');d,c,provenance=items[0];tasks=[t for t in p['physical_tasks'] if t['episode']==0 and t['phase']=='early'];recipient=next(t for t in tasks if t['onset']['direction']>0);left=next(t for t in tasks if t['onset']['direction']<0);original_begin=recipient['begin'];begin=int(np.flatnonzero(d['camera_index'][original_begin:]!=d['camera_index'][original_begin])[0]+original_begin);end=begin+32;shift=begin-original_begin;assert end<recipient['onset']['event'] and left['begin']+shift+32<left['onset']['event'];camera_ids=np.unique(d['camera_index'][begin:end]).astype(int);event_map=np.searchsorted(camera_ids,d['camera_index'][begin:end]);first={int(j):int(np.flatnonzero(d['camera_index'][begin:end]==j)[0]) for j in camera_ids};actor=Path(entries[0]['data']);am=read(actor/'manifest.json')['config'];recording=Path(am['arguments']['recording']);certificate=Path(am['arguments']['qualification']);qr=read(certificate/'result.json');assert qr['training_admissible'];geometry_file=Path('runs/thermal-driving-20261103/spatial-recovery-01/pool-geometry.npz')
 assert sha256(geometry_file)==read(p['reserved_plan'])['geometry_sha256'];assert sha256('configs/ac-csp-exterior-front-v3-retina.json')==read(p['reserved_plan'])['camera_sha256']
 with np.load(geometry_file) as z:geometry={k:z[k].copy() for k in ['old','inverse','grid8_narrow']}
 pool=components[0].scene_pool.to('cuda').eval().requires_grad_(False);np.testing.assert_array_equal(pool.weight.cpu().numpy(),geometry['old']);vc=read('configs/flight-stream-batched-v1-20261029.json');md=Path(vc['author_model_dir']);assert sha256(md/'best_chkpt')==vc['author_weights_sha256'];os.environ['FLYVIS_ROOT_DIR']=str(Path(vc['cache_root']).resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(vc['cache_root']);import flyvis
 with torch.device('cuda'):
  nc=yaml.safe_load((md/'_meta.yaml').read_text(encoding='utf-8'))['config']['network'];network=flyvis.Network(**nc).to('cuda');network.load_state_dict(torch.load(md/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False);nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{kind}{direction}' for kind in ['T4','T5'] for direction in 'abcd']);projector=FlyvisCameraProjector(read('configs/ac-csp-exterior-front-v3-retina.json'),u,v);prior=network.steady_state(1.,.005,batch_size=4,value=.5)
 out=begin_run(ROOT/'real-rgb-fork-01',dict(program_sha256=sha256(__file__),source_actor_sha256=provenance['arrays_sha256'],qualification_sha256=sha256(certificate/'result.json'),author_weights_sha256=vc['author_weights_sha256'],geometry_sha256=sha256(geometry_file),source_recording=str(recording),recipient=recipient,left_donor=left,aligned_begin=begin,end=end,alignment_shift=shift,seed=None,optimizer_updates=0,actor_qualified=False,training_admissible=False,vehicle_control_sent=False,scope=__doc__));(out/'executed-entrypoint.py').write_bytes(Path(__file__).read_bytes());start=time.perf_counter();result=dict(status='failed',actor_qualified=False,training_admissible=False,vehicle_control_sent=False,optimizer_updates=0);arrays={k:[] for k in ['scene','static','motion','coarse']};branches=[];max_prefix=0.;max_identity=0.
 try:
  with CameraArchive(recording/'camera/vision') as archive:
   assert archive.bundle_sha256==qr['camera_bundle_sha256'];rows=[json.loads(x) for x in archive.read('stream.jsonl').decode('utf-8').splitlines()];skipped=set(c['skipped_visual_frames']);stream=MotionPerViewFlyvisStream(network,prior,pool,projector,indices,geometry,episode_id=c['episode_id'],generation=rows[0]['generation'],device='cuda')
   def pixels(row):
    values={}
    for record in row['views']:
     raw=archive.read(Path(record['file']).name);assert hashlib.sha256(raw).hexdigest()==record['file_sha256'];rgb=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')).copy();assert hashlib.sha256(rgb.tobytes()).hexdigest()==record['rgb_sha256'];values[record['view_id']]=rgb
    return values
   def process(s,recipient_row,pixel_row):
    clocks=tuple(next(v['render_started_lower'] for v in recipient_row['views'] if v['view_id']==name) for name in VIEW_ORDER)
    return s.process(pixels=pixels(pixel_row),render_start_lowers=clocks,camera_received=recipient_row['received'],capture_id=int(recipient_row['capture_id']),generation=recipient_row['generation'])
   for i,row in enumerate(rows[:int(camera_ids[0])]):
    if i in skipped:continue
    obs=process(stream,row,row)
    matched=np.flatnonzero(d['camera_index']==i)
    if len(matched):max_prefix=max(max_prefix,float(abs(obs.scene-d['scene'][matched[0]]).max()));assert max_prefix<2e-6
    if i%100==0:write_json(out/'progress.json',dict(source_prefix_frame=i,total=int(camera_ids[0]),maximum_scene_difference=max_prefix,elapsed_seconds=time.perf_counter()-start))
   prefix_hash,tensor_count=state_hash(stream.state);held_hash=hashlib.sha256(stream.held.cpu().numpy().tobytes()).hexdigest();definitions=[('identity_right',begin),('retimed_left',left['begin']+shift),('identity_repeat',begin)]
   for name,donor_begin in definitions:
    fork=copy.copy(stream);fork.state=copy.deepcopy(stream.state);fork.held=stream.held.clone();fork.prefix=None;fork.poisoned=False;initial_hash,_=state_hash(fork.state);assert initial_hash==prefix_hash;assert hashlib.sha256(fork.held.cpu().numpy().tobytes()).hexdigest()==held_hash;values={k:[] for k in arrays};mapping=[]
    for j in camera_ids:
     offset=first[int(j)];donor_frame=int(d['camera_index'][donor_begin+offset]);assert int(j) not in skipped and donor_frame not in skipped;obs=process(fork,rows[int(j)],rows[donor_frame]);mapping.append(dict(recipient_frame=int(j),donor_frame=donor_frame,recipient_offset=offset,rgb_sha256=list(obs.rgb_sha256)))
     for key,value in [('scene',obs.scene),('static',obs.static),('motion',obs.motion_features),('coarse',obs.coarse_motion_reference)]:values[key].append(value.copy())
     if name.startswith('identity'):
      i=begin+offset;mi=int(maps[0][i]);max_identity=max(max_identity,float(abs(obs.scene-d['scene'][i]).max()),float(abs(obs.static-extras[0][i].numpy()).max()),float(abs(obs.motion_features-features[0][mi].numpy()).max()))
    for key in arrays:arrays[key].append(np.asarray(values[key]))
    assert state_hash(stream.state)[0]==prefix_hash;branches.append(dict(name=name,donor_begin=donor_begin,initial_full_visual_state_sha256=initial_hash,initial_held_retina_sha256=held_hash,mapping=mapping))
   repeat=max(float(abs(arrays[key][0]-arrays[key][2]).max()) for key in arrays);assert repeat<2e-6 and max_identity<2e-6;np.savez_compressed(out/'features.npz',**{k:np.asarray(v) for k,v in arrays.items()},event_to_camera=event_map,camera_indices=camera_ids,decision_indices=np.arange(begin,end));write_json(out/'branches.json',branches);result.update(status='completed_original_rgb_identical_flyvis_forks',aligned_begin=begin,end=end,original_student_prefix_begin=original_begin,episode=0,source_prefix_camera_frames=int(camera_ids[0]),branch_camera_groups=len(camera_ids),source_prefix_scene_maximum_difference=max_prefix,identity_cached_feature_maximum_difference=max_identity,identity_repeat_maximum_difference=repeat,full_visual_state_tensor_count=tensor_count,prefix_full_visual_state_sha256=prefix_hash,features_sha256=sha256(out/'features.npz'),branches_sha256=sha256(out/'branches.json'),upstream_flyvis_state_rebranched=True,physical_counterfactual_verified=False,termination_reason='Actual pixels and complete original prefix replayed; full recurrent state and held retina copied, exact recipient clocks reused, identity controls verified. Left pixels retimed offline, not a physical vehicle trajectory.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 try:
  torch.cuda.synchronize()
  if 'fork' in locals():del fork
  if 'stream' in locals():del stream
  del network,prior,pool,components;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();result['explicit_gpu_drain_and_release_completed']=True
 except Exception as error:result.update(status='failed',gpu_cleanup_error=str(error))
 result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(result)
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
