"""Replay author visual dynamics and retain modeled T4/T5 pre-pooling space.

Engineering null-space readout, not measured neural data or actor deployment.
"""
import argparse,io,json,os,shutil,time,yaml
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import assemble_student,read
from flybrain_drive.spatial_pool_residual import discarded_retinal_component
from scripts.train_qualified_sac_unified import load_episode


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actor-data',required=True);p.add_argument('--spatial',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    actor=Path(a.actor_data);d,contract,provenance=load_episode(actor);cfg=read(actor/'manifest.json')['config'];recording=Path(cfg['arguments']['recording']);certificate=Path(cfg['arguments']['qualification']);qr=read(certificate/'result.json')
    if not qr['training_admissible'] or contract['source_qualification_sha256']!=sha256(certificate/'result.json'):raise ValueError('Qualified original stream required')
    spatial=Path(a.spatial);sr=read(spatial/'result.json');geometry_file=spatial/'pool-geometry.npz'
    if sr['status']!='completed_qualified_static_spatial_recovery' or sha256(geometry_file)!=sr['geometry_sha256']:raise ValueError('Verified fixed pooling geometry required')
    with np.load(geometry_file) as z:geometry={k:z[k].copy() for k in ['old','inverse','grid8_narrow','uv']}
    model,lineage=assemble_student();pool=model.scene_pool.to('cuda').eval().requires_grad_(False)
    if not np.array_equal(pool.weight.cpu().numpy(),geometry['old']):raise ValueError('Original motion pool changed')
    visual_cfg=read('configs/flight-stream-batched-v1-20261029.json');modeldir=Path(visual_cfg['author_model_dir']);cache=Path(visual_cfg['cache_root'])
    if sha256(modeldir/'best_chkpt')!=visual_cfg['author_weights_sha256']:raise ValueError('Frozen author visual weights changed')
    os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
    import flyvis
    with torch.device('cuda'):
        network_cfg=yaml.safe_load((modeldir/'_meta.yaml').read_text(encoding='utf-8'))['config']['network'];network=flyvis.Network(**network_cfg).to('cuda');network.load_state_dict(torch.load(modeldir/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
        nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors];names=[f'{k}{j}' for k in ['T4','T5'] for j in 'abcd'];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,names);projector=FlyvisCameraProjector(read('configs/ac-csp-exterior-front-v3-retina.json'),u,v)
        if not np.allclose(projector.uv,geometry['uv'],atol=1e-8,rtol=0):raise ValueError('Original registered retina changed')
        state=network.steady_state(1.,.005,batch_size=4,value=.5);held=torch.full((4,len(u)),.5,device='cuda')
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),actor_result_sha256=sha256(actor/'result.json'),actor_arrays_sha256=provenance['arrays_sha256'],qualification_sha256=sha256(certificate/'result.json'),geometry_sha256=sha256(geometry_file),author_visual_weights_sha256=visual_cfg['author_weights_sha256'],lineage=lineage,seed=None,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope='Original qualified per-view times and author dynamics; modeled T4/T5 activity tanh before fixed4x4 pooling. Retain null-space8x8 features; no measured neuronal tuning, feature usefulness or new actor qualification.'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');needed=np.unique(d['camera_index']).astype(int);needed_set=set(map(int,needed));first={int(i):int(np.flatnonzero(d['camera_index']==i)[0]) for i in needed};skipped=set(contract['skipped_visual_frames']);previous=None;features=[];coarse=[];ids=[];energy=[];lost_energy=[];sum_x=np.zeros((4,8,len(u)),float);sum_res=np.zeros_like(sum_x);total_energy=np.zeros((4,8),float);total_lost=np.zeros((4,8),float);maximum_scene_error=0.;maximum_null_error=0.;start=time.perf_counter();result=dict(status='failed',actor_qualified=False,optimizer_updates=0,vehicle_control_sent=False)
    try:
        with CameraArchive(recording/'camera/vision') as archive:
            if archive.bundle_sha256!=qr['camera_bundle_sha256']:raise ValueError('Original camera bundle changed')
            rows=[json.loads(line) for line in archive.read('stream.jsonl').decode('utf-8').splitlines()]
            for i,row in enumerate(rows[:int(needed[-1])+1]):
                if i in skipped:continue
                frame=archive.json(Path(row['file']).name);views={x['view_id']:x for x in frame['views']};records={x['view_id']:x for x in row['views']};events=sorted((records[name]['render_started_lower'],j,name) for j,name in enumerate(VIEW_ORDER))
                for stamp,j,name in events:
                    if previous is not None:
                        if stamp<=previous:raise ValueError('Original per-view clocks reversed')
                        for dt,count in held_input_batches(stamp-previous):state=network.simulate(held[:,None,None,:].expand(-1,count,-1,-1),dt=dt,initial_state=state,as_states=True)[-1]
                    rgb=np.asarray(Image.open(io.BytesIO(archive.read(views[name]['file']))).convert('RGB')).copy();held[j]=torch.as_tensor(projector.sample(rgb).copy(),device='cuda');previous=stamp
                if i not in needed_set:continue
                raw=torch.cat([held[:,None,:],state.nodes.activity[:,indices]],dim=1)[None];scene=pool(raw).cpu().numpy()[0];expected=d['scene'][first[i]];error=float(abs(scene-expected).max());maximum_scene_error=max(maximum_scene_error,error)
                if error>2e-6:raise ValueError('Original full visual recurrence not reconstructed')
                x=torch.tanh(raw[0,:,1:]).cpu().numpy().astype(float);residual=discarded_retinal_component(x,geometry['old'],geometry['inverse']);null_error=float(abs(residual@geometry['old'].T).max());maximum_null_error=max(maximum_null_error,null_error)
                if null_error>1e-6:raise ValueError('New residual changes old motion pool')
                fine=(residual@geometry['grid8_narrow'].T).reshape(-1).astype(np.float32);features.append(fine);coarse.append((x@geometry['old'].T).reshape(-1).astype(np.float32));ids.append(i)
                e=np.sum(x*x,axis=-1);lost=np.sum(residual*residual,axis=-1);energy.append(e);lost_energy.append(lost);sum_x+=x;sum_res+=residual;total_energy+=e;total_lost+=lost
                if len(ids)%100==0:write_json(run/'progress.json',dict(camera_groups=len(ids),total=len(needed),source_frame=i,maximum_scene_error=maximum_scene_error,maximum_null_error=maximum_null_error,elapsed_seconds=time.perf_counter()-start))
        np.testing.assert_array_equal(ids,needed);n=len(ids);mean=sum_x/n;mean_res=sum_res/n;variation=total_energy-n*np.sum(mean*mean,axis=-1);variation_lost=total_lost-n*np.sum(mean_res*mean_res,axis=-1)
        if np.min(variation)<-1e-7 or np.min(variation_lost)<-1e-7:raise ValueError('Negative temporal variation energy beyond numerical tolerance')
        variation=np.maximum(variation,0);variation_lost=np.maximum(variation_lost,0);valid_energy=variation>1e-12;fraction=np.divide(variation_lost,variation,out=np.zeros_like(variation),where=valid_energy)
        if np.max(fraction)>1+1e-6:raise ValueError('Orthogonal motion residual exceeds total temporal variation')
        np.savez_compressed(run/'motion-spatial.npz',camera_indices=np.asarray(ids),event_to_camera=np.searchsorted(np.asarray(ids),d['camera_index']),discarded_motion_grid8=np.asarray(features),original_motion_pool=np.asarray(coarse),per_frame_energy=np.asarray(energy),per_frame_lost_energy=np.asarray(lost_energy),mean_motion_tanh=mean,mean_residual=mean_res,temporal_variation_energy=variation,temporal_variation_lost_energy=variation_lost,temporal_fraction= fraction,temporal_fraction_valid=valid_energy)
        result.update(status='completed_qualified_motion_spatial_recovery',camera_groups=n,model_events=len(d['labels']),modeled_cell_types=names,view_order=list(VIEW_ORDER),feature_count=2048,original_scene_max_difference=maximum_scene_error,old_motion_pool_null_max_difference=maximum_null_error,temporal_variation_fraction_lost=fraction.tolist(),temporal_variation_fraction_valid=valid_energy.tolist(),weighted_temporal_variation_fraction_lost=float(variation_lost.sum()/variation.sum()),features_sha256=sha256(run/'motion-spatial.npz'),state_resets=1,unqualified_pixel_inputs=0,prepool_motion_recovered=True,connected_to_actor=False,termination_reason='Modeled motion recurrence matches original qualified scene and pre-pooling residual is invisible to original pool; engineering spatial readout only, not physiological validation, causal driving improvement or new training.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),camera_groups=len(ids))
    result['elapsed_seconds']=time.perf_counter()-start;write_json(run/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
