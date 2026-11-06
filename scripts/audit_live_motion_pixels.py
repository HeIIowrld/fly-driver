"""Reconstruct actual selected live pixels independently of the stream subclass."""
import argparse,io,json,os,shutil,sys,yaml
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import assemble_student,read
from flybrain_drive.spatial_pool_residual import discarded_retinal_component


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    root=Path(a.shadow);r=read(root/'result.json');events=read(root/'events.json');selected=[e for e in events if e.get('status')=='scene_published']
    if r['status']!='completed_teacher_moving_native_flight_shadow' or not r['motion_features_enabled'] or sha256(root/'events.json')!=r['events_sha256'] or sha256(root/'published-visual-inputs.npz')!=r['published_visual_inputs_sha256']:raise ValueError('Completed unchanged live motion evidence required')
    with np.load(root/'published-visual-inputs.npz') as z:original={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(original['capture_ids'],[e['capture_id'] for e in selected])
    geometry_file=Path('runs/thermal-driving-20261103/spatial-recovery-01/pool-geometry.npz')
    with np.load(geometry_file) as z:geometry={k:z[k].copy() for k in ('old','inverse','grid8_narrow')}
    model,_=assemble_student();pool=model.scene_pool.to('cuda').eval().requires_grad_(False);np.testing.assert_array_equal(pool.weight.cpu().numpy(),geometry['old'])
    cfg=read('configs/flight-stream-batched-v1-20261029.json');md=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root'])
    if sha256(md/'best_chkpt')!=cfg['author_weights_sha256']:raise ValueError('Author weights changed')
    os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
    import flyvis
    with torch.device('cuda'):
        nc=yaml.safe_load((md/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**nc).to('cuda');network.load_state_dict(torch.load(md/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
        nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{j}' for k in ('T4','T5') for j in 'abcd']);projector=FlyvisCameraProjector(read('configs/ac-csp-exterior-front-v3-retina.json'),u,v);state=network.steady_state(1.,.005,batch_size=4,value=.5);held=torch.full((4,len(u)),.5,device='cuda')
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),shadow_result_sha256=sha256(root/'result.json'),published_visual_inputs_sha256=r['published_visual_inputs_sha256'],author_visual_weights_sha256=cfg['author_weights_sha256'],geometry_sha256=sha256(geometry_file),seed=None,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,exit_strategy='Closed archive/evidence then explicit in-main exit before affected ROCm local tensor finalization; GPU-only tool, no child/control sources'));shutil.copyfile(__file__,out/'executed-entrypoint.py');maximum={k:0. for k in ('scene','static','motion','coarse')};reconstructed={k:[] for k in maximum};previous=None;result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
    try:
        with CameraArchive(root/'vision') as archive:
            if archive.bundle_sha256!=r['camera_archive']['bundle_sha256']:raise ValueError('Actual camera archive changed')
            rows={int(x['capture_id']):x for x in [json.loads(line) for line in archive.read('stream.jsonl').decode('utf8').splitlines()]}
            for i,event in enumerate(selected):
                row=rows[event['capture_id']];records={x['view_id']:x for x in row['views']};frame=archive.json(Path(row['file']).name);views={x['view_id']:x for x in frame['views']};clocks=tuple(records[n]['render_started_lower'] for n in VIEW_ORDER)
                if list(clocks)!=event['render_start_lowers'] or row['received']!=event['camera_received']:raise ValueError('Actual selected source clocks differ')
                sampled=np.asarray([projector.sample(np.asarray(Image.open(io.BytesIO(archive.read(views[n]['file']))).convert('RGB')).copy()).copy() for n in VIEW_ORDER])
                for stamp,j in sorted((t,j) for j,t in enumerate(clocks)):
                    if previous is not None:
                        if stamp<=previous:raise ValueError('Original visual clocks reversed')
                        for dt,count in held_input_batches(stamp-previous):state=network.simulate(held[:,None,None,:].expand(-1,count,-1,-1),dt=dt,initial_state=state,as_states=True)[-1]
                    held[j]=torch.as_tensor(sampled[j],device='cuda');previous=stamp
                raw=torch.cat([held[:,None,:],state.nodes.activity[:,indices]],dim=1)[None];scene=pool(raw).cpu().numpy()[0];motion=torch.tanh(raw[0,:,1:]).cpu().numpy().astype(float);residual=discarded_retinal_component(motion,geometry['old'],geometry['inverse']);static_residual=discarded_retinal_component(np.tanh(sampled),geometry['old'],geometry['inverse'])
                values=dict(scene=scene,static=np.concatenate([scene.reshape(4,9,16)[:,0].reshape(-1),(static_residual@geometry['grid8_narrow'].T).reshape(-1).astype(np.float32)]),motion=(residual@geometry['grid8_narrow'].T).reshape(-1).astype(np.float32),coarse=(motion@geometry['old'].T).reshape(-1).astype(np.float32))
                for key,value in values.items():
                    maximum[key]=max(maximum[key],float(abs(value-original[key][i]).max()));reconstructed[key].append(value)
                    if maximum[key]>2e-6:raise ValueError('Actual selected pixel recurrence differs '+key)
        np.savez_compressed(out/'reconstructed-features.npz',capture_ids=original['capture_ids'],**{k:np.asarray(v) for k,v in reconstructed.items()})
        result.update(status='completed_independent_live_motion_pixel_audit',groups_verified=len(selected),maximum_differences=maximum,features_sha256=sha256(out/'reconstructed-features.npz'),actual_selected_pixels_reconstructed=True,continuous_original_per_view_state=True,state_resets=1,termination_reason='Actual selected pixel bytes and original per-view clocks independently reconstruct old scene/static and new fine/coarse motion; no student actuation, action correctness or new-corner qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),groups_verified=len(reconstructed['scene']),maximum_differences=maximum)
    write_json(out/'result.json',result);print(json.dumps(result),flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(1 if result['status']=='failed' else 0)


if __name__=='__main__':main()
