"""Verify matched schedules and that the first training difference is a fixture adjoint."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/direct-scene-gain-learning-20261105')
def signature(row):
 t=row['task'];return row['kind'],row['epoch'],tuple((k,t.get(k)) for k in ['episode','begin','end','phase','source_slot','pair','temperature','group','left_donor','right_donor'])
def main():
 rows={};values={}
 for arm in ['frozen_direct','learn_direct']:
  path=ROOT/f'{arm}-fit-01';r=read(path/'result.json');assert sha256(path/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'] and sha256(path/'training-records.json')==r['training_records_sha256'];rows[arm]=read(path/'training-records.json')
  with np.load(path/'optimizer-trajectory.npz') as z:values[arm]={k:z[k].copy() for k in ['before','after','gradients']}
 a,b=values['frozen_direct'],values['learn_direct'];assert [signature(r) for r in rows['frozen_direct']]==[signature(r) for r in rows['learn_direct']];assert np.array_equal(a['before'][0],b['before'][0]);different=np.max(abs(a['gradients']-b['gradients']),axis=1);first=int(np.flatnonzero(different>0)[0]);assert np.array_equal(a['before'][:first+1],b['before'][:first+1]) and np.array_equal(a['after'][:first],b['after'][:first]);loss0=rows['frozen_direct'][first]['loss'];loss1=rows['learn_direct'][first]['loss'];assert abs(loss0-loss1)<1e-12
 cut=a['before'].shape[1]-42
 assert np.array_equal(a['gradients'][first,:cut],b['gradients'][first,:cut]) and np.array_equal(a['after'][first,:cut],b['after'][first,:cut]);assert np.max(abs(a['after'][first,cut:]-b['after'][first,cut:]))>0 and np.count_nonzero(a['after'][:,cut:])==0
 write_json(ROOT/'matched-training-audit-01.json',dict(status='completed_matched_direct_visual_gain_training_audit',initial_all_parameter_difference=0.,all_task_schedules_equal=True,first_gradient_difference_update=first,first_gradient_difference_kind=rows['frozen_direct'][first]['kind'],first_objective_difference=abs(loss0-loss1),first_existing_four_parameter_gradients_and_updates_exact=True,first_new_visual_gain_gradient_max_difference=float(different[first]),first_new_visual_gain_parameter_max_difference=float(abs(a['after'][first,cut:]-b['after'][first,cut:]).max()),frozen_control_gain_maximum=0.,actor_qualified=False,vehicle_control_sent=False));print(first,rows['frozen_direct'][first]['kind'],float(different[first]))
if __name__=='__main__':main()
