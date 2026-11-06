"""Full original-pixel parity for the opt-in live motion stream, without actuation."""
import argparse,io,json,os,shutil,sys,time,yaml
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.motion_per_view_flyvis_stream import MotionPerViewFlyvisStream
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import assemble_student,read
from scripts.train_qualified_sac_unified import load_episode


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('actor-data','motion-recovery','spatial','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    actor=Path(a.actor_data);d,c,provenance=load_episode(actor);cfg=read(actor/'manifest.json')['config'];recording=Path(cfg['arguments']['recording']);qualification=Path(cfg['arguments']['qualification']);qr=read(qualification/'result.json')
    recovery=Path(a.motion_recovery);rr=read(recovery/'result.json');rm=read(recovery/'manifest.json')['config'];geometry_file=Path(a.spatial)/'pool-geometry.npz'
    if not qr['training_admissible'] or c['source_qualification_sha256']!=sha256(qualification/'result.json'):raise ValueError('Qualified original episode required')
    if rr['status']!='completed_qualified_motion_spatial_recovery' or rm['actor_arrays_sha256']!=provenance['arrays_sha256'] or sha256(recovery/'motion-spatial.npz')!=rr['features_sha256']:raise ValueError('Unchanged original motion recovery required')
    with np.load(geometry_file) as z:geometry={k:z[k].copy() for k in ('old','inverse','grid8_narrow')}
    with np.load(recovery/'motion-spatial.npz') as z:ids=z['camera_indices'].copy();expected=z['discarded_motion_grid8'].copy();coarse=z['original_motion_pool'].copy();mapping=z['event_to_camera'].copy()
    np.testing.assert_array_equal(ids[mapping],d['camera_index'])
    model,_=assemble_student();pool=model.scene_pool.to('cuda').eval().requires_grad_(False)
    if not np.array_equal(pool.weight.cpu().numpy(),geometry['old']):raise ValueError('Original scene geometry differs')
    visual_cfg=read('configs/flight-stream-batched-v1-20261029.json');md=Path(visual_cfg['author_model_dir']);cache=Path(visual_cfg['cache_root'])
    if sha256(md/'best_chkpt')!=visual_cfg['author_weights_sha256']:raise ValueError('Author visual weights changed')
    os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
    import flyvis
    with torch.device('cuda'):
        nc=yaml.safe_load((md/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**nc).to('cuda');network.load_state_dict(torch.load(md/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
        nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{j}' for k in ('T4','T5') for j in 'abcd']);projector=FlyvisCameraProjector(read('configs/ac-csp-exterior-front-v3-retina.json'),u,v);prior=network.steady_state(1.,.005,batch_size=4,value=.5)
    sources={path:sha256(path) for path in ('src/flybrain_drive/per_view_flyvis_stream.py','src/flybrain_drive/motion_per_view_flyvis_stream.py','src/flybrain_drive/held_visual_prefix.py')}
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),source_sha256=sources,actor_arrays_sha256=provenance['arrays_sha256'],motion_recovery_result_sha256=sha256(recovery/'result.json'),geometry_sha256=sha256(geometry_file),author_visual_weights_sha256=visual_cfg['author_weights_sha256'],optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,seed=None,exit_strategy='Flush terminal evidence then explicit process exit; avoid ROCm interpreter finalizer hang. Camera archive context closes before result. No child/actuator/native source.'));shutil.copyfile(__file__,out/'executed-entrypoint.py')
    result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False);first={int(i):int(np.flatnonzero(d['camera_index']==i)[0]) for i in ids};position={int(i):j for j,i in enumerate(ids)};skipped=set(c['skipped_visual_frames']);scenes=[];statics=[];features=[];references=[];processed=0;maximum_scene=maximum_motion=maximum_coarse=0.;start=time.perf_counter()
    try:
        with CameraArchive(recording/'camera/vision') as archive:
            if archive.bundle_sha256!=qr['camera_bundle_sha256']:raise ValueError('Camera source bundle differs')
            rows=[json.loads(line) for line in archive.read('stream.jsonl').decode('utf8').splitlines()];stream=None
            for i,row in enumerate(rows[:int(ids[-1])+1]):
                if i in skipped:continue
                frame=archive.json(Path(row['file']).name);views={x['view_id']:x for x in frame['views']};records={x['view_id']:x for x in row['views']};clocks=tuple(records[n]['render_started_lower'] for n in VIEW_ORDER)
                if stream is None:stream=MotionPerViewFlyvisStream(network,prior,pool,projector,indices,geometry,episode_id=c['episode_id'],generation=row['generation'],device='cuda')
                pixels={n:np.asarray(Image.open(io.BytesIO(archive.read(views[n]['file']))).convert('RGB')).copy() for n in VIEW_ORDER}
                obs=stream.process(pixels=pixels,render_start_lowers=clocks,camera_received=row['received'],capture_id=int(row['capture_id']),generation=row['generation']);processed+=1
                if i in position:
                    j=position[i];maximum_scene=max(maximum_scene,float(abs(obs.scene-d['scene'][first[i]]).max()));maximum_motion=max(maximum_motion,float(abs(obs.motion_features-expected[j]).max()));maximum_coarse=max(maximum_coarse,float(abs(obs.coarse_motion_reference-coarse[j]).max()))
                    if max(maximum_scene,maximum_motion,maximum_coarse)>2e-6:raise ValueError('Full live stream differs from original training recurrence/features')
                    scenes.append(obs.scene.copy());statics.append(obs.static.copy());features.append(obs.motion_features.copy());references.append(obs.coarse_motion_reference.copy())
                stream.prepare_held_prefix()
                if processed%100==0:write_json(out/'progress.json',dict(processed_groups=processed,matched_groups=len(scenes),total=len(ids),maximum_scene=maximum_scene,maximum_motion=maximum_motion,elapsed_seconds=time.perf_counter()-start))
        if len(scenes)!=len(ids):raise ValueError('Full episode camera coverage required')
        np.savez_compressed(out/'stream-features.npz',camera_indices=ids,event_to_camera=mapping,scene=np.asarray(scenes),static=np.asarray(statics),motion=np.asarray(features),coarse=np.asarray(references))
        result.update(status='completed_motion_stream_source_parity',full_episode=True,held_prefix=True,state_resets=1,processed_groups=processed,matched_groups=len(scenes),model_events=len(d['scene']),scene_max_difference=maximum_scene,motion_max_difference=maximum_motion,coarse_max_difference=maximum_coarse,features_sha256=sha256(out/'stream-features.npz'),termination_reason='Original full-pixel per-view recurrence including held prefix matches original coarse scene and recovered fine motion; no student actuation or new-corner qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),processed_groups=processed,matched_groups=len(scenes))
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(json.dumps(result),flush=True)
    # Exit while the local visual tensors are still owned here. Returning from
    # main can itself enter the affected ROCm tensor finalizer before the
    # entrypoint gets control. All archive files/evidence are already closed.
    sys.stdout.flush();sys.stderr.flush();os._exit(1 if result['status']=='failed' else 0)


if __name__=='__main__':
    try:main();code=0
    except SystemExit as error:code=error.code if isinstance(error.code,int) else 1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
