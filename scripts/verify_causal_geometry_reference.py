"""Compare causal stationary-reference implementation with prior synthetic training inputs."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_geometry_feedback import CausalGeometryVisualPort
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


class ZeroBase(nn.Module):
    def forward(self,traces,body,valid):
        return torch.zeros(1,2)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    paths=[base/'20261101-flyvis-coherent-four-view-yaw',
           *(base/f'20261101-flyvis-coherent-yaw-texture-{seed}' for seed in [863102,863103,863104,863105,863106,863107])]
    camera_file=Path('configs/ac-csp-exterior-front-v3-retina.json')
    registration=base/'20261031-pace-train-flyvis-replay/retina-registration.npz'
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),camera_sha256=sha256(camera_file),
        registration_sha256=sha256(registration),
        inputs=[dict(path=str(p),sha256=sha256(p/'responses.npz')) for p in paths],
        native_samples='synthetic encoded fixtures; not simulator observations',optimizer_updates=0))
    with np.load(registration,allow_pickle=False) as f:
        pool=VisualSelfMotionPool.from_registration(read(camera_file),{k:f[k].copy() for k in f.files})
    reports=[]
    for path in paths:
        proof=read(path/'result.json')
        if proof['status'] != 'completed_coherent_four_view_yaw' or sha256(path/'responses.npz') != proof['responses_sha256']:
            raise ValueError('Original fixed synthetic source required')
        with np.load(path/'responses.npz',allow_pickle=False) as f:
            motion,times=f['motion'].copy(),f['times'].copy()
        original=pool.features(motion)
        expected=original-original[:,:11].mean(1,keepdims=True)
        expected[:,:11]=0
        expected[:,:,[0,1,3,4,5]]=0
        actual=np.zeros_like(expected,dtype=np.float32)
        for case in range(3):
            port=CausalGeometryVisualPort(ZeroBase(),pool,2)
            for i,t in enumerate(times):
                raw=bytearray(308);struct.pack_into('<i',raw,0,i)
                # Fixture's own sensor: stationary for reference, moving afterwards.
                struct.pack_into('<f',raw,28,0. if i<11 else 36.)
                native=decode_physics_prefix(bytes(raw),100.+float(t))
                ready=port.observe(motion[case,i],np.ones(4,dtype=bool),native,
                    capture_id=i,decision_time=100.+float(t)+.001, camera_receipt_time=100.+float(t))
                if ready != (i>=11):
                    raise ValueError('Causal readiness differs from prior training contract')
                if ready:actual[case,i]=port.current.cpu().numpy()[0]
        error=float(np.max(np.abs(actual-expected)))
        if error > 1e-6:raise ValueError('Online causal features differ from precomputed training features')
        reports.append(dict(path=str(path),cases=3,decisions_per_case=41,max_abs_error=error))
    result=dict(status='completed_synthetic_causal_reference_equivalence',reports=reports,
        original_training_feature_equivalence=True,synthetic_native_samples=True,
        physical_exposure_or_real_stationarity_verified=False,optimizer_updates=0,
        vehicle_control_sent=False,actor_deployed=False)
    write_json(run/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
