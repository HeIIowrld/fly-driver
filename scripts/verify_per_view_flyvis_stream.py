"""Compare the online-capable per-view visual stream against qualified training pixels."""
import argparse,copy,io,json,os,shutil,time,yaml
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.per_view_flyvis_stream import PerViewFlyvisStream
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import assemble_student,read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',required=True);p.add_argument('--spatial',required=True);p.add_argument('--episode',type=int,choices=[0,1,2],default=2)
    p.add_argument('--held-prefix',action='store_true');p.add_argument('--groups',type=int,default=64);p.add_argument('--output',required=True);a=p.parse_args()
    if not 8<=a.groups<=1600:raise ValueError('Finite8..1600 source groups required')
    torch.set_num_threads(1)
    entry=three_episode_cohort(a.split)[a.episode];d,c,provenance=load_episode(entry['data']);assert provenance['arrays_sha256']==entry['arrays_sha256']
    source_cfg=read(Path(entry['data'])/'manifest.json')['config'];recording=Path(source_cfg['arguments']['recording']);cert=read(Path(source_cfg['arguments']['qualification'])/'result.json')
    config=read('configs/flight-stream-batched-v1-20261029.json');modeldir=Path(config['author_model_dir']);cache=Path(config['cache_root'])
    assert sha256(modeldir/'best_chkpt')==config['author_weights_sha256']
    spatial=Path(a.spatial);sr=read(spatial/'result.json');sp=spatial/f'spatial-{a.episode}.npz'
    assert sha256(sp)==sr['files_sha256'][sp.name]
    with np.load(sp) as z:spatial_ids=z['camera_indices'].copy();expected_static=z['discarded_grid8'].copy()
    with np.load(spatial/'pool-geometry.npz') as z:geometry={k:z[k].copy() for k in ['old','inverse','grid8_narrow']}
    model,_=assemble_student();pool=copy.deepcopy(model.scene_pool)
    np.testing.assert_array_equal(pool.weight.numpy(),geometry['old']);del model
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),visual_stream_sha256=sha256('src/flybrain_drive/per_view_flyvis_stream.py'),
        entry=entry,author_weights_sha256=config['author_weights_sha256'],qualified_camera_bundle_sha256=cert['camera_bundle_sha256'],
        optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,seed=None,
        scope='Original source pixels and per-view time schedule; frozen author network parity. Processing speed diagnostic, not live whole-system deadline proof'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');shutil.copyfile('src/flybrain_drive/per_view_flyvis_stream.py',run/'executed-visual-stream.py')
    result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0);scenes=[];statics=[];ids=[];timings=[];prefix_times=[];errors=[];residual_errors=[];began=time.monotonic()
    try:
        if not torch.cuda.is_available():raise RuntimeError('Existing working GPU author environment required')
        pool=pool.to('cuda').eval().requires_grad_(False)
        os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
        import flyvis
        with torch.device('cuda'):
            netcfg=yaml.safe_load((modeldir/'_meta.yaml').read_text(encoding='utf8'))['config']['network']
            network=flyvis.Network(**netcfg).to('cuda');network.load_state_dict(torch.load(modeldir/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
            nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors]
            indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{j}' for k in ('T4','T5') for j in 'abcd'])
            projector=FlyvisCameraProjector(read('configs/ac-csp-exterior-front-v3-retina.json'),u,v)
            state=network.steady_state(1.,.005,batch_size=4,value=.5)
        visual=PerViewFlyvisStream(network,state,pool,projector,indices,geometry,episode_id=entry['episode_id'],generation='qualified-recording',device='cuda')
        skipped=set(c['skipped_visual_frames']);event_for_camera={int(camera):int(np.flatnonzero(d['camera_index']==camera)[0]) for camera in np.unique(d['camera_index'])}
        with CameraArchive(recording/'camera/vision') as archive:
            assert archive.bundle_sha256==cert['camera_bundle_sha256']
            stream=[json.loads(x) for x in archive.read('stream.jsonl').decode('utf8').splitlines()]
            for i,row in enumerate(stream[:a.groups]):
                if i in skipped:continue
                frame=archive.json(Path(row['file']).name);byname={v['view_id']:v for v in frame['views']};records={v['view_id']:v for v in row['views']}
                pixels={name:np.asarray(Image.open(io.BytesIO(archive.read(byname[name]['file']))).convert('RGB')).copy() for name in VIEW_ORDER}
                started=time.perf_counter()
                obs=visual.process(pixels=pixels,render_start_lowers=tuple(records[name]['render_started_lower'] for name in VIEW_ORDER),camera_received=row['received'],capture_id=i,generation='qualified-recording')
                torch.cuda.synchronize();timings.append(time.perf_counter()-started)
                if i in event_for_camera:
                    j=event_for_camera[i];error=float(abs(obs.scene-d['scene'][j]).max());errors.append(error)
                    if error>2e-6:raise ValueError('Per-view stream differs from qualified actor scene')
                    k=int(np.searchsorted(spatial_ids,i));assert spatial_ids[k]==i
                    error=float(abs(obs.static[64:]-expected_static[k]).max());residual_errors.append(error)
                    if error>2e-6:raise ValueError('Static residual differs from original qualified sidecar')
                if a.held_prefix:
                    prefix_start=time.perf_counter();visual.prepare_held_prefix();torch.cuda.synchronize();prefix_times.append(time.perf_counter()-prefix_start)
                scenes.append(obs.scene);statics.append(obs.static);ids.append(i)
                if len(ids)%16==0:write_json(run/'progress.json',dict(source_camera_index=i,admitted_groups=len(ids),elapsed_seconds=time.monotonic()-began))
        if not errors:raise ValueError('No qualified actor scene comparison')
        result.update(status='completed_per_view_flyvis_source_parity',admitted_groups=len(ids),actor_scene_comparisons=len(errors),scene_max_difference=max(errors),static_residual_max_difference=max(residual_errors),
            processing_seconds_percentiles=np.percentile(timings,[50,95,99,100]).tolist(),held_prefix=a.held_prefix,prefix_seconds_percentiles=np.percentile(prefix_times,[50,95,99,100]).tolist() if prefix_times else [],state_resets=1,skipped_initial_unqualified_frames=sorted(skipped),
            termination_reason='Original per-view pixel schedule reproduced frozen author scene and discarded spatial features; no live source or end-to-end control qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),completed_groups=len(ids),termination_reason='Explicit source parity or runtime failure; source retained, no deployment')
    np.savez_compressed(run/'features.npz',scene=np.asarray(scenes),static=np.asarray(statics),camera_indices=np.asarray(ids),processing_seconds=np.asarray(timings),prefix_seconds=np.asarray(prefix_times))
    result['features_sha256']=sha256(run/'features.npz');write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
