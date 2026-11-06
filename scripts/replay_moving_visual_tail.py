"""Retained moving pixels/clocks: visual phase timing and unchanged-feature parity."""
import argparse,copy,io,json,os,shutil,time,yaml
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_runtime_gc import defer_automatic_cyclic_gc
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.per_view_flyvis_stream import PerViewFlyvisStream
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import assemble_student,read
from flybrain_drive.visual_phase_probe import VisualPhaseProbe

SPATIAL=Path('runs/thermal-driving-20261103/spatial-recovery-01')

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1)
    source=Path(a.recording);source_result=read(source/'result.json');events=read(source/'events.json')
    assert sha256(source/'events.json')==source_result['events_sha256']
    selected=[e for e in events if 'visual_seconds' in e]
    assert selected and all(b['capture_id']>a['capture_id'] for a,b in zip(selected,selected[1:]))
    frames=[]
    with CameraArchive(source/'vision') as archive:
        assert archive.bundle_sha256==source_result['camera_archive']['bundle_sha256']
        rows={row['capture_id']:row for row in map(json.loads,archive.read('stream.jsonl').decode().splitlines())}
        for event in selected:
            i=event['capture_id'];row=rows[i];frame=archive.json(Path(row['file']).name)
            byname={v['view_id']:v for v in frame['views']};records={v['view_id']:v for v in row['views']}
            assert row['received']==event['camera_received']
            clocks=tuple(records[name]['render_started_lower'] for name in VIEW_ORDER)
            assert clocks==tuple(event['render_start_lowers'])
            pixels={name:np.asarray(Image.open(io.BytesIO(archive.read(byname[name]['file']))).convert('RGB')).copy() for name in VIEW_ORDER}
            frames.append((event,dict(pixels=pixels,render_start_lowers=clocks,camera_received=row['received'],capture_id=i,generation='retained-moving')))
    model,_=assemble_student()
    cfg=read('configs/flight-stream-batched-v1-20261029.json');md=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root']);assert sha256(md/'best_chkpt')==cfg['author_weights_sha256']
    os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
    import flyvis
    camera_config=read('configs/ac-csp-exterior-front-v3-retina.json');pool=copy.deepcopy(model.scene_pool).to('cuda').eval().requires_grad_(False)
    with np.load(SPATIAL/'pool-geometry.npz') as z:geometry={k:z[k].copy() for k in ['old','inverse','grid8_narrow']}
    with torch.device('cuda'):
        nc=yaml.safe_load((md/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**nc).to('cuda')
        network.load_state_dict(torch.load(md/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
        nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors]
        indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{j}' for k in ('T4','T5') for j in 'abcd'])
        projector=FlyvisCameraProjector(camera_config,u,v);prior=network.steady_state(1.,.005,batch_size=4,value=.5)
    # Warm discarded computations only. The actual stream starts from its
    # original author .5 steady prior, not from these synthetic pixels/states.
    warm_visual=PerViewFlyvisStream(network,prior,pool,projector,indices,geometry,episode_id='backend-warmup',generation='warmup',device='cuda')
    for i in range(3):
        warm_visual.process(pixels={n:np.full((180,320,3),128,dtype=np.uint8) for n in VIEW_ORDER},render_start_lowers=tuple(1+i*.04+j*.001 for j in range(4)),camera_received=1+i*.04+.01,capture_id=i,generation='warmup')
        warm_visual.prepare_held_prefix();torch.cuda.synchronize()
    del warm_visual
    with torch.device('cuda'):prior=network.steady_state(1.,.005,batch_size=4,value=.5)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),source_result_sha256=sha256(source/'result.json'),
        source_events_sha256=sha256(source/'events.json'),camera_bundle_sha256=source_result['camera_archive']['bundle_sha256'],
        visual_stream_sha256=sha256('src/flybrain_drive/per_view_flyvis_stream.py'),probe_sha256=sha256('src/flybrain_drive/visual_phase_probe.py'),
        seed=None,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,
        scope='Offline retained original pixels and clocks; no live cadence or isolated GPU kernel-time claim'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    passes=[];features={};profiles=[]
    try:
        for mode in ['reference','observed_gc_enabled','observed_gc_deferred']:
            with torch.device('cuda'):state=network.steady_state(1.,.005,batch_size=4,value=.5)
            stream=PerViewFlyvisStream(network,state,pool,projector,indices,geometry,episode_id='retained-moving',generation='retained-moving',device='cuda')
            probe=VisualPhaseProbe(stream,synchronize=torch.cuda.synchronize,event_factory=lambda:torch.cuda.Event(enable_timing=True))
            scenes=[];statics=[];identities=[];times=[];prefix_times=[]
            with defer_automatic_cyclic_gc() if mode=='observed_gc_deferred' else nullcontext() as gc_receipt:
                for event,args in frames:
                    record=dict(mode=mode,capture_id=event['capture_id'],process={});profiles.append(record)
                    started=time.perf_counter()
                    if mode=='reference':obs=stream.process(**args);torch.cuda.synchronize()
                    else:obs=probe.run('process',record['process'],**args)
                    times.append(time.perf_counter()-started)
                    scenes.append(obs.scene);statics.append(obs.static);identities.append(obs.rgb_sha256)
                    if 'prefix_seconds' in event:
                        started=time.perf_counter();record['prefix']={}
                        if mode=='reference':stream.prepare_held_prefix();torch.cuda.synchronize()
                        else:probe.run('prepare_held_prefix',record['prefix'])
                        prefix_times.append(time.perf_counter()-started)
            features[mode+'_scene']=np.array(scenes);features[mode+'_static']=np.array(statics)
            if mode=='reference':reference_identity=identities
            else:
                assert identities==reference_identity
                for field in ['scene','static']:
                    np.testing.assert_allclose(features[mode+'_'+field],features['reference_'+field],atol=2e-6,rtol=0)
            passes.append(dict(mode=mode,groups=len(scenes),capture_ids=[e['capture_id'] for e,_ in frames],
                processing_seconds_percentiles=np.percentile(times,[50,95,99,100]).tolist(),
                prefix_seconds_percentiles=np.percentile(prefix_times,[50,95,99,100]).tolist(),gc_scope=gc_receipt,
                scene_max_difference=float(abs(features[mode+'_scene']-features['reference_scene']).max()),
                static_max_difference=float(abs(features[mode+'_static']-features['reference_static']).max())))
            write_json(run/'progress.json',passes)
        result.update(status='completed_retained_moving_visual_phase_replay',passes=passes,
            includes_original_unpublished_frame=any(e.get('status')!='scene_published' for e,_ in frames),
            termination_reason='Same retained pixels and original per-view clocks, fresh episode only per replay pass; no live timing or autonomous control qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    write_json(run/'phase-records.json',profiles)
    np.savez_compressed(run/'features.npz',**features)
    result.update(phase_records_sha256=sha256(run/'phase-records.json'),features_sha256=sha256(run/'features.npz'))
    write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
