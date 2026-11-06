"""Causal high-resolution motion history; engineering filters, not measured cells."""
from dataclasses import dataclass
import hashlib,math
import numpy as np
from .sensor_contract import readonly_array


@dataclass(frozen=True)
class FineMotionHistoryState:
    fast:np.ndarray
    slow:np.ndarray
    held:np.ndarray
    decision_time:float
    camera_receipt:float
    render_lowers:tuple
    features_sha256:str


class FineMotionHistory:
    def __init__(self,reference,scale,*,fast_seconds=.05,slow_seconds=.2):
        reference=np.asarray(reference,dtype=np.float32);scale=np.asarray(scale,dtype=np.float32)
        if reference.shape!=(2048,) or scale.shape!=(2048,) or not np.isfinite(reference).all() or not np.isfinite(scale).all() or np.any(scale<=0) or not np.isfinite([fast_seconds,slow_seconds]).all() or not 0<fast_seconds<slow_seconds:raise ValueError('Finite motion2048 normalization and ordered time constants required')
        self.reference=readonly_array(reference,np.float32);self.scale=readonly_array(scale,np.float32);self.fast_seconds=float(fast_seconds);self.slow_seconds=float(slow_seconds);self.episode=None;self.state=None;self.poisoned=False

    def begin_episode(self,episode_id):
        if not isinstance(episode_id,str) or not episode_id:raise ValueError('Explicit nonempty episode required')
        self.episode=episode_id;self.state=None;self.poisoned=False

    def step(self,features,*,episode_id,decision_time,camera_receipt,render_lowers):
        if self.poisoned or self.episode is None:raise RuntimeError('Explicit new history episode required')
        try:
            x=np.asarray(features,dtype=np.float32);clocks=tuple(map(float,render_lowers));t=float(decision_time);c=float(camera_receipt)
            if episode_id!=self.episode or x.shape!=(2048,) or not np.isfinite(x).all() or len(clocks)!=4 or not np.isfinite([t,c,*clocks]).all() or min(clocks)<0 or not max(clocks)<=c<=t or t-min(clocks)>.15:raise ValueError('Original finite same-episode motion and current view clocks required')
            digest=hashlib.sha256(x.tobytes()).hexdigest();normalized=np.clip((x-self.reference)/self.scale,-5,5);old=self.state
            if old is None:fast=normalized.copy();slow=normalized.copy()
            else:
                elapsed=t-old.decision_time
                if not 0<elapsed<=.08 or c<old.camera_receipt:raise ValueError('Advancing original decision and nondecreasing camera receipt required')
                if c==old.camera_receipt:
                    if clocks!=old.render_lowers or digest!=old.features_sha256:raise ValueError('Held camera identity changed bytes or clocks')
                elif min(clocks)<=max(old.render_lowers):raise ValueError('New camera view clocks do not advance')
                # Integrate only the feature available during the elapsed
                # interval. The current observation is installed afterwards.
                af=math.exp(-elapsed/self.fast_seconds);al=math.exp(-elapsed/self.slow_seconds);fast=(af*old.fast+(1-af)*old.held).astype(np.float32);slow=(al*old.slow+(1-al)*old.held).astype(np.float32)
            self.state=FineMotionHistoryState(readonly_array(fast,np.float32),readonly_array(slow,np.float32),readonly_array(normalized,np.float32),t,c,clocks,digest)
            return dict(innovation=readonly_array(normalized-slow,np.float32),fast_minus_slow=readonly_array(fast-slow,np.float32),features_sha256=digest)
        except Exception:
            self.poisoned=True
            raise
