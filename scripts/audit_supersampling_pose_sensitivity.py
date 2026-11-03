"""Compare historical and supersampled raster sensitivity on recorded poses."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.synthetic_antialiased_planar_road_visual import AntialiasedSyntheticPlanarRoadVisual

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();b=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');left=b/'20261102-paired-direction-loop-1';right=b/'20261102-straight-balanced-varied-1';lr=json.loads((left/'result.json').read_text());rr=json.loads((right/'result.json').read_text());
 if sha(left/'trajectory.npz')!=lr['output_sha256'] or sha(right/'trajectory.npz')!=rr['output_sha256']:raise ValueError('verified original histories required')
 cfg=Path('configs/ac-csp-exterior-front-v3-retina.json');camera=json.loads(cfg.read_text());coords=[(q,r) for q in range(-15,16) for r in range(max(-15,-15-q),min(15,15-q)+1)];u,v=np.asarray(coords).T;projector=FlyvisCameraProjector(camera,u,v);run=begin_run(a.output,dict(program_sha256=sha(__file__),renderer_sha256=sha('src/flybrain_drive/synthetic_antialiased_planar_road_visual.py'),camera_sha256=sha(cfg),left_sha256=lr['output_sha256'],right_sha256=rr['output_sha256'],seed=882617,indices=list(range(60,72)),samples_per_axis=[1,2,4],actor=False))
 with np.load(left/'trajectory.npz') as z,np.load(right/'trajectory.npz') as t:one=z['original_882617_012_0'].copy();two=t['candidate_882617_012_0'].copy()
 results={}
 for factor in (1,2,4):
  road=AntialiasedSyntheticPlanarRoadVisual(camera,curvature=.012,bend_start_m=12.,samples_per_axis=factor);rows=[]
  for i in range(60,72):
   def render(r):return road.render(bend=0,forward_m=float(r[1]),lateral_m=float(r[2]),yaw_rad=float(r[3]),texture_seed=882617)
   x,y=render(one[i]),render(two[i]);dx=np.abs(x.astype(np.int16)-y.astype(np.int16));rx=np.stack([projector.sample(frame) for frame in x]);ry=np.stack([projector.sample(frame) for frame in y]);d=np.abs(rx-ry);rows.append(dict(index=i,changed_pixels=int(np.count_nonzero(np.any(x!=y,axis=-1))),max_channel_difference=int(dx.max()),changed_receptors=int(np.count_nonzero(d)),max_retina_difference=float(d.max()),retina_l1=float(d.sum())))
  results[str(factor)]=rows;write_json(run/'progress.json',dict(completed_factor=factor))
 write_json(run/'rows.json',results);result=dict(status='completed_synthetic_supersampling_pose_audit',rows_sha256=sha(run/'rows.json'),actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0,interpretation='Fixed previously observed synthetic pose pairs; supersampling can smooth edges but is not real camera or biological validation');write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
