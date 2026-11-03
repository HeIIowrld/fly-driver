"""Synthetic coherent four-view yaw reversal through original FlyVis, no training."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector, VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.visual_self_motion import SphereCameraStimulus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--texture-seed',type=int,default=863101)
    p.add_argument('--yaw-amplitude',type=float,default=.23)
    a = p.parse_args()
    camera_file = Path('configs/ac-csp-exterior-front-v3-retina.json')
    camera = json.loads(camera_file.read_text(encoding='utf8'))
    model = Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000')
    cache = Path('C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-data-windows-v1')
    if not np.isfinite(a.yaw_amplitude) or not 0 < a.yaw_amplitude <= 1:
        raise ValueError('Positive bounded synthetic yaw amplitude required')
    angles = [0.,-a.yaw_amplitude,a.yaw_amplitude]
    origins = [camera['views'][name]['origin_body_m'] for name in VIEW_ORDER]
    run = begin_run(a.output,dict(program_sha256=sha256(__file__),
        camera_sha256=sha256(camera_file),checkpoint_sha256=sha256(model/'best_chkpt'),
        synthetic_sphere_radius_m=20.,yaw_rad_s=angles,texture_seed=a.texture_seed,
        reverse_after_motion_seconds=.6, trajectory="stationary then yaw, continuous opposite yaw",
        frame_dt=.04,frame_count=41,stationary_seconds=.4,internal_dt=.005,
        optimizer_updates=0,actor_deployed=False))
    result = dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,
        actor_deployed=False,physiological_or_closed_loop_verified=False)
    try:
        torch.set_num_threads(1); torch.manual_seed(863009)
        if not torch.cuda.is_available(): raise RuntimeError('GPU required')
        os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve()); os.environ['MPLBACKEND']='Agg'
        install_datamate_cache_compat(cache)
        import flyvis
        torch.set_default_device('cuda')
        config=yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))['config']['network']
        network=flyvis.Network(**config).to('cuda')
        weights=torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network']
        network.load_state_dict(weights,strict=True); network.eval().requires_grad_(False)
        nodes=network.connectome.nodes
        receptors=network.stimulus.input_index[0]
        u,v=nodes.u[:][receptors],nodes.v[:][receptors]
        indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,
            [f'{kind}{direction}' for kind in ('T4','T5') for direction in 'abcd'])
        projector=FlyvisCameraProjector(camera,u,v)
        sphere=SphereCameraStimulus(camera,origins,radius=20.)
        retina_all,motion_all=[],[]
        torch.cuda.reset_peak_memory_stats()
        for case,yaw in enumerate(angles):
            frames=[]
            for i in range(41):
                t=max(0.,i*.04-.4)
                reversing_time=min(t,.6)-max(0.,t-.6)
                rgb=sphere.render([0.,0.,yaw],[0.,0.,0.],reversing_time,texture_seed=a.texture_seed)
                frames.append(np.stack([projector.sample(view) for view in rgb]))
            retina=np.stack(frames)
            movie=torch.tensor(np.transpose(np.repeat(retina,8,axis=0),(1,0,2))[:,:,None,:],device='cuda')
            with torch.no_grad():
                activity=network.simulate(movie,dt=.005,initial_state='auto')
                motion=activity[:,7::8,indices].permute(1,0,2,3).cpu().numpy()
            if motion.shape != (41,4,8,721) or not np.isfinite(motion).all():
                raise ValueError('Complete finite original four-view motion required')
            retina_all.append(retina); motion_all.append(motion)
            write_json(run/'progress.json',dict(completed=case+1,total=3))
            del activity,motion,movie
        path=run/'responses.npz'
        np.savez_compressed(path,retina=np.stack(retina_all),motion=np.stack(motion_all),
            yaw_rad_s=np.array(angles),times=np.arange(41)*.04)
        result.update(status='completed_coherent_four_view_reversing_yaw',responses_sha256=sha256(path),
            cases=angles,frames=41,per_case_independent_view_neural_states=True,
            source_geometry='synthetic textured sphere, declared camera mounting, continuous yaw reversal',
            body_or_ground_truth_actor_input=False,peak_gpu_bytes=int(torch.cuda.max_memory_allocated()))
    except Exception as e:
        result.update(error=type(e).__name__,reason=str(e))
    finally:
        write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed': raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
