"""Replay audited native-only recording through FlyVis and causal stationary visual port (diagnostic)."""
import argparse, base64, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flyvis_camera import FlyvisCameraProjector, VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool
from flybrain_drive.flight_geometry_feedback import CausalGeometryVisualPort
from flybrain_drive.ac_native_physics import decode_physics_prefix
from scripts.audit_ac_csp_sensory import replay_recording
from scripts.replay_ac_camera_motion_probe import diagnostic_steps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('probe','model-dir','cache-root','output'): p.add_argument('--'+n,required=True)
    a=p.parse_args(); root=Path(a.probe); model=Path(a.model_dir)
    source=json.loads((root/'result.json').read_text(encoding='utf8'))
    archive={x['file']:x['sha256'] for x in json.loads((root/'archive.json').read_text(encoding='utf8'))}
    if source['status']!='completed' or source.get('sdk_omitted') is not True or not torch.cuda.is_available(): raise ValueError('Audited native-only recording and CUDA required')
    audited=replay_recording(root,allow_native_only=True)
    if not audited['exact_replay_verified']: raise ValueError('Exact raw replay required')
    events=[json.loads(x) for x in (root/'client/events.jsonl').read_text(encoding='utf8').splitlines()]
    os.environ['FLYVIS_ROOT_DIR']=a.cache_root; install_datamate_cache_compat(a.cache_root)
    import flyvis
    torch.manual_seed(863009); torch.set_num_threads(1); torch.set_default_device('cuda')
    config=yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))
    network=flyvis.Network(**config['config']['network']).to('cuda')
    weights=torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network']
    network.load_state_dict(weights,strict=True); network.eval().requires_grad_(False)
    nodes=network.connectome.nodes; idx=network.stimulus.input_index[0]
    u,v=nodes.u[:][idx],nodes.v[:][idx]
    camera=json.loads(Path('configs/ac-csp-yaw-retina.json').read_text(encoding='utf8'))
    projector=FlyvisCameraProjector(camera,u,v); pool=VisualSelfMotionPool(camera,projector.directions_body)
    types,all_u,all_v=nodes.type[:].astype(str),nodes.u[:],nodes.v[:]
    channels=[]
    for name in ('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'):
        layer=np.flatnonzero(types==name); mapping={(int(all_u[i]),int(all_v[i])):i for i in layer}
        channels.append([mapping[(int(q),int(r))] for q,r in zip(u,v,strict=True)])
    channels=np.asarray(channels); images=[]; clocks=[]
    for i,row in enumerate(events):
        meta_path=root/'camera'/f'frame-{i:04d}.json'; raw=meta_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=archive[f'camera/frame-{i:04d}.json']: raise ValueError('Metadata hash mismatch')
        meta=json.loads(raw); views={v['view_id']:v for v in row['image_sources']}
        if set(views)!=set(VIEW_ORDER) or meta['index']!=i: raise ValueError('Original view identity required')
        clocks.append(float(row['decision_monotonic_seconds']))
        sampled=[]
        for name in VIEW_ORDER:
            entry=views[name]; path=root/'camera'/entry['file']; payload=path.read_bytes(); digest=hashlib.sha256(payload).hexdigest()
            if digest!=entry['sha256'] or digest!=archive['camera/'+entry['file']]: raise ValueError('Image hash mismatch')
            with Image.open(path) as img: sampled.append(projector.sample(np.asarray(img.convert('RGB'))))
        images.append(np.stack(sampled))
    clocks=np.asarray(clocks); images=np.asarray(images)
    if not np.all(np.diff(clocks)>0): raise ValueError('Original CSP clock must advance')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_result_sha256=sha256(root/'result.json'),
        source_archive_sha256=sha256(root/'archive.json'),checkpoint_sha256=sha256(model/'best_chkpt'),
        camera_config_sha256=sha256('configs/ac-csp-yaw-retina.json'),seed=863009,
        clock='host decision receipt clock, physical capture unverified', counterfactual='first valid image held; invalid frames withheld',
        training_allowed=False,actor_deployed=False,optimizer_updates=0))
    with torch.no_grad():
        state=network.steady_state(1.,.005,batch_size=8,value=.5)
        held=None; first_valid=None; activity=[]; features=[]; ticks=0
        for i,clock in enumerate(clocks):
            if i:
                for dt in diagnostic_steps(float(clock-clocks[i-1])):
                    state=network.simulate(held[:,None,None,:],dt=dt,initial_state=state,as_states=True)[-1];ticks+=1
            valid=all(v>0 for v in events[i]['summary']['view_valid_counts'])
            if valid and first_valid is None: first_valid=images[i].copy()
            if valid:
                held=torch.as_tensor(np.concatenate([images[i],first_valid],axis=0),device='cuda')
            elif held is None:
                held=torch.full((8,images.shape[-1]),.5,device='cuda')
            x=state.nodes.activity[:,channels].cpu().numpy();activity.append(x)
            features.append(np.stack([pool.features(x[:4]),pool.features(x[4:])]))
    activity=np.asarray(activity);features=np.asarray(features)
    port=CausalGeometryVisualPort(torch.nn.Identity(),pool,2).to('cuda')
    rows=[]
    for i,event in enumerate(events):
        native=event['native']; raw=base64.b64decode(native['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=native['prefix_sha256']: raise ValueError('Native prefix hash mismatch')
        body=decode_physics_prefix(raw,native['received_monotonic_seconds'])
        valid=np.asarray([v>0 for v in event['summary']['view_valid_counts']],dtype=bool)
        ready=port.observe(activity[i,:4],valid,body,capture_id=i,decision_time=float(clocks[i]),camera_receipt_time=float(event['host_image_receipt']))
        rows.append(dict(index=i,ready=ready,reason=port.reason,reference_ready=port.reference.value is not None,reference_rows=len(port.reference.rows),geometry_yaw=float(port.current[0,2]) if ready else None))
    write_json(run/'port-rows.json',rows)
    np.savez_compressed(run/'responses.npz',clock=clocks,retina=images,motion=activity,features=features)
    diff=features[:,0]-features[:,1]
    write_json(run/'result.json',dict(status='completed_offline_visual_probe',groups=len(clocks),ticks=ticks,
        host_decision_intervals=np.diff(clocks).tolist(),feature_abs_max=np.abs(diff).max(axis=0).tolist(),
        final_feature_delta=diff[-1].tolist(), motion_abs_max=float(np.abs(activity[:,:4]-activity[:,4:]).max()),
        port_rows_sha256=sha256(run/'port-rows.json'), port_ready_count=sum(x['ready'] for x in rows),
        first_ready_index=next((x['index'] for x in rows if x['ready']),None),
        reasons={k:sum(r['reason']==k for r in rows) for k in sorted(set(r['reason'] for r in rows))},
        responses_sha256=sha256(run/'responses.npz'), stationary_physically_verified=False,
        measured_exposure_verified=False,training_allowed=False,vehicle_control_sent=False))
    print(json.dumps(json.loads((run/'result.json').read_text())))

if __name__=='__main__': main()
