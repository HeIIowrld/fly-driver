"""Same exact visual initialization/parity as moving shadow; no simulator access."""
import copy,os,yaml
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.sac_unified_student import read
from flybrain_drive.graph import sha256
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate,require_motion_stream_parity
from flybrain_drive.motion_per_view_flyvis_stream import MotionPerViewFlyvisStream
from flybrain_drive.per_view_flyvis_stream import PerViewFlyvisStream

@torch.no_grad()
def prepare(candidate_plan_path):
    candidate=read(candidate_plan_path)
    components,motion_port,receipt=load_readonly_motion_candidate(candidate)
    model=components[0]
    SPATIAL=Path('runs/thermal-driving-20261103/spatial-recovery-01')
    proof=require_motion_stream_parity(candidate,receipt,SPATIAL/'pool-geometry.npz')
    visual_class=MotionPerViewFlyvisStream
    if motion_port is None:raise ValueError('Current actor requires motion stream')
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
    warm_visual=visual_class(network,prior,pool,projector,indices,geometry,episode_id='backend-warmup',generation='warmup',device='cuda')
    for i in range(3):
        warm_visual.process(pixels={n:np.full((180,320,3),128,dtype=np.uint8) for n in VIEW_ORDER},render_start_lowers=tuple(1+i*.04+j*.001 for j in range(4)),camera_received=1+i*.04+.01,capture_id=i,generation='warmup')
        warm_visual.prepare_held_prefix();torch.cuda.synchronize()
    del warm_visual
    with torch.device('cuda'):prior=network.steady_state(1.,.005,batch_size=4,value=.5)
    def build(episode_id,generation):
        return visual_class(network,prior,pool,projector,indices,geometry,episode_id=episode_id,generation=generation,device='cuda')
    return build,camera_config,dict(candidate=receipt,motion_proof=proof)
