"""Compare the reused original-guard training trajectory with the new dual guard."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/dual-reference-learning-20261105')

def signature(row):
 t=row['task'];return row['kind'],row['epoch'],tuple((k,t.get(k)) for k in ['episode','begin','end','phase','source_slot','pair','temperature','group','left_donor','right_donor'])

def main():
 p=read(ROOT/'plan-01.json');folders=[Path(p['control_fit']),ROOT/'dual_guard-fit-01'];rows=[];arrays=[]
 for folder in folders:
  r=read(folder/'result.json');assert sha256(folder/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'] and sha256(folder/'training-records.json')==r['training_records_sha256'];rows.append(read(folder/'training-records.json'))
  with np.load(folder/'optimizer-trajectory.npz') as z:arrays.append({k:z[k].copy() for k in ['before','after','gradients']})
 assert [signature(x) for x in rows[0]]==[signature(x) for x in rows[1]] and len(rows[0])==395;a,b=arrays;assert np.array_equal(a['before'][0],b['before'][0]);delta=np.max(abs(a['gradients']-b['gradients']),axis=1);where=np.flatnonzero(delta>0);first=None if not len(where) else int(where[0]);loss=np.array([abs(x['parts'][2]-y['parts'][2]) if x['kind']=='real' else 0. for x,y in zip(*rows)]);loss_where=np.flatnonzero(loss>1e-12)
 if first is not None:assert rows[0][first]['kind']=='real' and np.array_equal(a['before'][:first+1],b['before'][:first+1])
 assert np.count_nonzero(a['after'][:,-42:])==np.count_nonzero(b['after'][:,-42:])==0
 write_json(ROOT/'matched-training-audit-01.json',dict(status='completed_reused_control_dual_guard_matching_audit',control_refit=False,initial_all_parameter_difference=0.,all_395_task_schedules_equal=True,first_clipped_gradient_difference_update=first,first_clipped_gradient_difference=None if first is None else float(delta[first]),first_protection_value_difference_update=None if not len(loss_where) else int(loss_where[0]),maximum_protection_value_difference=float(loss.max()),maximum_final_parameter_difference=float(abs(a['after'][-1]-b['after'][-1]).max()),direct_gain_fixed_zero=True,actor_qualified=False,vehicle_control_sent=False))

if __name__=='__main__':main()
