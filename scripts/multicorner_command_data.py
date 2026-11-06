"""Protocol and original fixed joint candidate for multi-corner command studies."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment
from scripts.sac_steering_experiment import onsets

ROOT=Path('runs/multicorner-command-20261104')

def tasks_for_episode(data,episode):
 tasks=[]
 for corner,onset in enumerate(onsets(data)):
  for phase,offset in [('early',-1.),('late',-.4),('turning',.2)]:
   begin=int(np.searchsorted(data['decision_time'],onset['time']+offset));end=begin+32
   if end>len(data['scene']) or not data['body_valid'][begin:end][:,[4,16]].all() or (data['body'][begin:end,16]<.02).any():raise ValueError('Valid nonzero speed/yaw throughout predeclared command window required')
   tasks.append(dict(episode=episode,corner=corner,phase=phase,begin=begin,end=end,onset=onset,role='TRAIN' if episode<2 and phase!='late' else 'TRAIN_diagnostic' if episode<2 else 'development'))
 if len(tasks)!=15:raise ValueError('Expected five original corner onsets, three phases each')
 return tasks

def experiment(plan):
 parent=Path(plan['parent_checkpoint']);assert sha256(parent)==plan['parent_checkpoint_sha256'];cp=torch.load(parent,map_location='cpu',weights_only=True);e=JointDNDecoderExperiment(read(plan['parent_fit_plan']),4,cp['readout']);e.calibration.load_state_dict(cp['current_calibration'],strict=True);e.head.requires_grad_(False);e.calibration.requires_grad_(False);e.verify_fixed();return e

def collections(plan):
 rows=[]
 for episode in range(3):
  path=ROOT/f'episode{episode}-probes-01';r=read(path/'result.json');assert r['status']=='completed_multi_corner_command_collection' and sha256(path/'probes.npz')==r['probes_sha256'] and sha256(path/'neutral-full.npz')==r['neutral_full_sha256'];rows.append((path,r))
 return rows
