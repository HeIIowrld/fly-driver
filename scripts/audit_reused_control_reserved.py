"""Independent arithmetic for the reused control evaluated on the new texture."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/dual-reference-learning-20261105')

def main():
 folder=ROOT/'control-reserved-02';r=read(folder/'result.json');assert sha256(folder/'predictions.npz')==r['predictions_sha256'] and sha256(folder/'reserved-groups.json')==r['reserved_groups_sha256'];groups=read(folder/'reserved-groups.json');directions={k:0 for k in r['reserved_direction_counts']};target=0
 with np.load(folder/'predictions.npz') as z:
  for row in groups:
   for arm in directions:
    x=z[f"group{row['group']}_{arm}_prediction"];margin=float(-(x[1,14:,0]-x[0,14:,0]).mean()/2);assert margin==row['arms'][arm]['signed_margin'];directions[arm]+=int(margin>.0001);target+=int(arm=='fine' and margin>=.002)
 assert directions==r['reserved_direction_counts'] and target==r['reserved_target_pairs'];write_json(ROOT/'control-reserved-audit-01.json',dict(status='completed_independent_reused_control_reserved_audit',direction_counts=directions,target_pairs=target,groups=len(groups),old_full_control_replay_reused=True,vehicle_control_sent=False))

if __name__=='__main__':main()
