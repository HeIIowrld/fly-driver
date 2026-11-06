"""Causal author FlyVis stream with the original per-view capture event schedule.

A camera group is four differently timed samples, not four simultaneous frames.
Absolute clocks only schedule integration; actor outputs are pooled features.
"""
import hashlib
import math
from dataclasses import dataclass
import numpy as np
import torch
from .flyvis_camera import VIEW_ORDER
from .flight_stream import held_input_batches
from .held_visual_prefix import HeldVisualPrefix
from .sensor_contract import readonly_array
from .spatial_pool_residual import discarded_retinal_component


@dataclass(frozen=True)
class VisualFlightObservation:
    episode_id: str
    generation: str
    capture_id: int
    camera_received: float
    render_start_lowers: tuple
    scene: np.ndarray
    static: np.ndarray
    rgb_sha256: tuple
    physical_exposure_time_verified: bool = False


class PerViewFlyvisStream:
    def __init__(self, network, state, pool, projector, indices, geometry, *, episode_id, generation, device):
        if not episode_id or not generation:
            raise ValueError('Explicit episode and producer generation required')
        self.network, self.state, self.pool, self.projector = network,state,pool,projector
        self.indices, self.geometry, self.device = indices,geometry,device
        self.episode_id, self.generation = episode_id,generation
        self.held = torch.full((4,len(projector.uv)),.5,device=device)
        self.previous = None; self.capture = None; self.poisoned = False; self.prefix = None

    @torch.no_grad()
    def process(self, *, pixels, render_start_lowers, camera_received, capture_id, generation):
        if self.poisoned:
            raise RuntimeError('Rejected visual stream requires explicit new episode construction')
        try:
            if (generation!=self.generation or type(capture_id) is not int or capture_id<0
                    or self.capture is not None and capture_id<=self.capture
                    or set(pixels)!=set(VIEW_ORDER) or len(render_start_lowers)!=4
                    or not math.isfinite(camera_received)):
                raise ValueError('Advancing original four-view producer identity required')
            clocks=tuple(map(float,render_start_lowers));events=sorted((t,j,name) for j,(t,name) in enumerate(zip(clocks,VIEW_ORDER)))
            if (not np.isfinite(clocks).all() or min(clocks)<0 or max(clocks)>camera_received
                    or any(b[0]<=a[0] for a,b in zip(events,events[1:]))
                    or self.previous is not None and events[0][0]<=self.previous):
                raise ValueError('Strict original per-view capture clocks required')
            # Validate/project all views before any recurrent step. A later view
            # is installed only at its own source event, never before that time.
            sampled=[];digests=[]
            for name in VIEW_ORDER:
                rgb=pixels[name]
                if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(180,320,3) or not np.any(rgb):
                    raise ValueError('Four original nonblank RGB320x180 frames required')
                digests.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                sampled.append(self.projector.sample(rgb).copy())
            sampled=np.asarray(sampled)
            for stamp,j,name in events:
                if self.previous is not None:
                    if self.prefix is not None:
                        self.state=self.prefix.commit(stamp-self.previous);self.prefix=None
                    else:
                        for dt,count in held_input_batches(stamp-self.previous):
                            self.state=self.network.simulate(self.held[:,None,None,:].expand(-1,count,-1,-1),dt=dt,initial_state=self.state,as_states=True)[-1]
                self.held[j]=torch.as_tensor(sampled[j],device=self.device)
                self.previous=stamp
            scene=self.pool(torch.cat([self.held[:,None,:],self.state.nodes.activity[:,self.indices]],dim=1)[None]).cpu().numpy()[0]
            retina=np.tanh(sampled)
            residual=discarded_retinal_component(retina,self.geometry['old'],self.geometry['inverse'])
            discarded=(residual@self.geometry['grid8_narrow'].T).reshape(-1).astype(np.float32)
            static=np.concatenate([scene.reshape(4,9,16)[:,0].reshape(-1),discarded])
            self.capture=capture_id
            return VisualFlightObservation(self.episode_id,self.generation,capture_id,float(camera_received),clocks,
                readonly_array(scene,np.float32),readonly_array(static,np.float32),tuple(digests))
        except Exception:
            self.poisoned=True
            raise


    @torch.no_grad()
    def prepare_held_prefix(self):
        if self.poisoned or self.previous is None or self.prefix is not None:
            raise ValueError('One speculative prefix after a committed visual group required')
        self.prefix=HeldVisualPrefix(self.network,self.state,self.held,steps=16)
