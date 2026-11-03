"""Bound current muscle-mix readout using fixed recorded circuit states."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();syn=B/'20261101-synthetic-candidate-spatial-path-trace';live=B/'20261101-unified-moving-teacher-path-ablations';sr=read(syn/'result.json');lr=read(live/'result.json');sf=syn/'traces.npz';lf=live/'ablations.npz'
 if sr['status']!='completed_synthetic_spatial_path_candidate_trace' or lr['status']!='completed_moving_teacher_shadow_path_ablations' or lr['training_admissible'] is not False or sha256(sf)!=sr['output_sha256'] or sha256(lf)!=lr['output_sha256']:raise ValueError('verified prior diagnostic traces required')
 mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];motor=PhysicalManeuverCircuit(mp['spec'],torch.load(mp['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu').eval();motor.requires_grad_(False);readout=motor.backbone.readout
 with np.load(sf,allow_pickle=False) as z:syn_muscles=z['muscles'][0].copy();syn_intent=z['intent'][0].copy()
 with np.load(lf,allow_pickle=False) as z:states=z['observed_state'].copy();real_intent=z['observed_intent'].copy()
 with torch.no_grad():recomputed,real_muscles=motor.backbone._read_state_tensor(torch.as_tensor(states));real_muscles=real_muscles.cpu().numpy();recomputed=recomputed.cpu().numpy()
 error=float(np.max(np.abs(recomputed-real_intent)))
 if error>2e-5:raise ValueError('recorded original state/readout mismatch')
 ref=readout.lateral_reference.cpu().numpy();gain=float(readout.turn_log_gain.exp().cpu());mix=readout.steering_mix.softmax(0).cpu().numpy()
 reports={};arrays={}
 for label,muscles,intents in [('synthetic',syn_muscles,syn_intent),('recorded',real_muscles,real_intent)]:
  lat=np.stack((muscles[...,1]-muscles[...,0],muscles[...,3]-muscles[...,2]),axis=-1); centered=lat-ref;reconstructed=np.tanh(gain*(centered@mix));err=float(np.max(np.abs(reconstructed-intents[...,0])))
  if err>2e-5 or not np.isfinite(centered).all():raise ValueError('independent readout reconstruction mismatch')
  extrema=np.tanh(gain*centered);both_negative=np.all(centered<0,axis=-1);both_positive=np.all(centered>0,axis=-1);reports[label]={'samples':int(both_negative.size),'both_channels_negative':int(both_negative.sum()),'both_channels_positive':int(both_positive.sum()),'opposite_or_zero':int((~(both_negative|both_positive)).sum()),'readout_reconstruction_max_abs':err,'turn_min_extreme':float(extrema.min()),'turn_max_extreme':float(extrema.max())};arrays[label+'_muscles']=muscles;arrays[label+'_turn_extrema']=extrema
  if label=='synthetic':
   e=extrema[:,-5:,:].mean(axis=1);reports[label]['last5_extreme_turn_by_bend']=e.tolist();reports[label]['last5_bend_difference_each_extreme']=(e[1]-e[0]).tolist()
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),synthetic_sha256=sr['output_sha256'],recorded_sha256=lr['output_sha256'],motor_checkpoint_sha256=sha256(mp['checkpoint']),optimizer_updates=0,training_admissible=False));np.savez_compressed(run/'readout-bounds.npz',**arrays)
 result={'status':'completed_fixed_reference_muscle_mix_bounds','reports':reports,'recorded_replay_max_abs':error,'mix':mix.tolist(),'turn_gain':gain,'reference':ref.tolist(),'arrays_sha256':sha256(run/'readout-bounds.npz'),'training_admissible':False,'optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False,'interpretation':'For fixed per-decision muscle states, reference and positive gain, any softmax mix lies between two extreme readouts. Does not cover changes to circuit state, negative gain, reference, physiology, or driving.'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
