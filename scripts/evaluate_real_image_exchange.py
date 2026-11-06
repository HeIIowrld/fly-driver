"""Offline real pre-corner image exchange at identical body/candidate prefixes.

Images are deliberately reassigned to the recipient clock. This is a consumed
same-course intervention, not a physically valid sensor replay or action truth.
Future teacher steering only labels donor bends and never enters actor inputs.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment
from scripts.sac_steering_experiment import onsets,restore,snapshot
from scripts.train_qualified_sac_unified import context
from scripts.intervene_motion_spatial_same_body import state_digest

class ImageExchangeExperiment(JointDNDecoderExperiment):
 def inputs(self):
  original=super().inputs();donor=self.holder.get('donor')
  if donor is None:return original
  episode,index=donor;j=int(self.maps[episode][index]);return self.extras[episode][index:index+1],self.features[episode][j:j+1],original[2],original[3]
 def step(self,i,learn):
  donor=self.holder.get('donor')
  if donor is None:return super().step(i,learn)
  if learn:raise ValueError('Image exchange has no optimizer path')
  self.holder['index']=i;d,c,_=self.items[self.holder['episode']];de,di=donor;scene=torch.tensor(self.items[de][0]['scene'][di:di+1]);ctx=context(d,c,i)
  with torch.no_grad():_,mu=self.model.sensory_decision(decision_time=float(d['decision_time'][i]),racing_goals=self.goals,scene_features=scene,scene_valid=self.views,body_context=ctx);return self.head(mu)[0]

def select_donors(d):
 donors=[]
 for onset in onsets(d):
  begin=int(np.searchsorted(d['decision_time'],onset['time']-.8));end=begin+32
  if end>onset['event'] or float(abs(d['body'][begin:end,4]).mean())>=.07:continue
  donors.append(dict(episode=2,begin=begin,end=end,onset=onset,mean_speed_m_s=float(d['body'][begin:end,16].mean()),mean_absolute_yaw_rad_s=float(abs(d['body'][begin:end,4]).mean()),fit_fraction=float(d['fit_mask'][begin:end].mean()),unique_camera_count=len(np.unique(d['camera_index'][begin:end])),last_frame_before_onset_seconds=float(onset['time']-d['decision_time'][end-1])))
 left=[v for v in donors if v['onset']['direction']<0];right=[v for v in donors if v['onset']['direction']>0]
 if not left or not right:raise ValueError('Both pre-yaw development donor bend directions required')
 pair=min([(l,r) for l in left for r in right],key=lambda p:(abs(p[0]['mean_speed_m_s']-p[1]['mean_speed_m_s']),p[0]['begin'],p[1]['begin']))
 return dict(left=pair[0],right=pair[1],selection='Minimum donor mean speed difference, teacher-sign-labelled pre-yaw windows; no model output used',mean_speed_difference_m_s=abs(pair[0]['mean_speed_m_s']-pair[1]['mean_speed_m_s']))

def main():
 torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');reports=[];saved={};out=None;donor_receipt=None
 for rank in [1,4]:
  fit=root/f'rank{rank}-fit-01';assert sha256(fit/'joint-dn-decoder.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'joint-dn-decoder.pt',map_location='cpu',weights_only=True);e=ImageExchangeExperiment(plan,rank,cp['readout']);e.calibration.load_state_dict(cp['current_calibration']);evaluation=root/f'rank{rank}-evaluation-01';assert sha256(evaluation/'candidate-prefix-states.pt')==read(evaluation/'result.json')['prefix_states_sha256'];prefixes=torch.load(evaluation/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);d=e.items[2][0];donors=[]
  try:
   receipt=select_donors(d);assert receipt==read(root/'real-image-exchange-plan-01.json')['donors'];pair=(receipt['left'],receipt['right'])
   if donor_receipt is None:donor_receipt=receipt;out=begin_run(root/'real-image-exchange-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(root/'fit-plan-01.json'),donors=receipt,scope=__doc__,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False))
   else:assert receipt==donor_receipt
   tasks=[v for v in read(Path(plan['yaw_probes'])/'result.json')['cases'] if v['role']=='development']
   for task in tasks:
    t=task['task'];begin,end=t['begin'],t['end'];assert end-begin==32;source=2;bodyhash=hashlib.sha256(d['body'][begin:end].tobytes()+d['body_valid'][begin:end].tobytes()+e.native[source][begin:end].numpy().tobytes()+e.valid[source][begin:end].numpy().tobytes()).hexdigest();prefixhash=state_digest(prefixes[source,begin]);row=dict(rank=rank,source=task,body_input_sha256=bodyhash,prefix_sha256=prefixhash,arms={});restore(e.model,prefixes[source,begin]);e.holder.update(mode='real',episode=source,calibration_off=False,motion_off=False);identity=[]
    for i in range(begin,end):
     e.holder['donor']=(source,i);identity.append(e.step(i,False).numpy())
    with np.load(evaluation/'predictions.npz') as z:expected=z[f'prediction_{source}'][begin:end].copy()
    identity_error=float(abs(np.asarray(identity)-expected).max());assert identity_error<2e-6;row['identity_donor_maximum_difference']=identity_error;saved[f'rank{rank}_case{task["case"]}_identity']=np.asarray(identity)
    for arm in ['fine','calibration_off_after_prefix']:
     branches=[]
     for donor in pair:
      restore(e.model,prefixes[source,begin]);assert state_digest(snapshot(e.model))==prefixhash;e.holder.update(mode='real',episode=source,calibration_off=arm!='fine',motion_off=False);sequence=[]
      for offset,i in enumerate(range(begin,end)):
       e.holder['donor']=(2,donor['begin']+offset);sequence.append(e.step(i,False).numpy())
      branches.append(sequence)
     p=np.asarray(branches);saved[f'rank{rank}_case{task["case"]}_{arm}']=p;row['arms'][arm]=dict(mean_left_image_steering=float(p[0,:,0].mean()),mean_right_image_steering=float(p[1,:,0].mean()),signed_teacher_bend_margin=float(-np.mean(p[0,:,0]-p[1,:,0])/2),mean_action_difference=(p[0]-p[1]).mean(0).tolist())
    reports.append(row)
   e.holder.pop('donor',None);e.verify_fixed()
  finally:e.close()
 assert out is not None;np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_real_pre_corner_image_exchange',reports=reports,donors=donor_receipt,predictions_sha256=sha256(out/'predictions.npz'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False,scope=__doc__))
if __name__=='__main__':main()
