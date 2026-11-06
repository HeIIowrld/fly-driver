"""Verify matched schedules and that the first training difference is a fixture adjoint."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/fixture-adjoint-learning-20261104')
def signature(row):
 t=row['task'];return row['kind'],row['epoch'],tuple((k,t.get(k)) for k in ['episode','begin','end','phase','source_slot','pair','temperature','group','left_donor','right_donor'])
def main():
 rows={};values={}
 for arm in ['stop_all','input_adjoint']:
  path=ROOT/f'{arm}-fit-01';r=read(path/'result.json');assert sha256(path/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'] and sha256(path/'training-records.json')==r['training_records_sha256'];rows[arm]=read(path/'training-records.json')
  with np.load(path/'optimizer-trajectory.npz') as z:values[arm]={k:z[k].copy() for k in ['before','after','gradients']}
 a,b=values['stop_all'],values['input_adjoint'];assert [signature(r) for r in rows['stop_all']]==[signature(r) for r in rows['input_adjoint']];assert np.array_equal(a['before'][0],b['before'][0]);different=np.max(abs(a['gradients']-b['gradients']),axis=1);first=int(np.flatnonzero(different>0)[0]);assert rows['stop_all'][first]['kind'] in ['synthetic','image_exchange'];assert np.array_equal(a['before'][:first+1],b['before'][:first+1]) and np.array_equal(a['after'][:first],b['after'][:first]);loss0=rows['stop_all'][first]['loss'];loss1=rows['input_adjoint'][first]['loss'];assert abs(loss0-loss1)<1e-12
 assert np.array_equal(a['gradients'][first,8192:],b['gradients'][first,8192:]) and np.count_nonzero(a['gradients'][first,8192:])==0 and np.array_equal(a['after'][first,8192:],b['after'][first,8192:]);assert np.max(abs(a['after'][first,:8192]-b['after'][first,:8192]))>0
 write_json(ROOT/'matched-training-audit-01.json',dict(status='completed_matched_fixture_adjoint_training_audit',initial_all_parameter_difference=0.,all_task_schedules_equal=True,first_gradient_difference_update=first,first_gradient_difference_kind=rows['stop_all'][first]['kind'],first_fixture_objective_difference=abs(loss0-loss1),first_fixture_visual_gradient_max_difference=float(different[first]),first_fixture_visual_parameter_max_difference=float(abs(a['after'][first,:8192]-b['after'][first,:8192]).max()),first_fixture_native_and_speed_gradient_and_parameter_differences=0.,actor_qualified=False,vehicle_control_sent=False));print(first,rows['stop_all'][first]['kind'],float(different[first]))
if __name__=='__main__':main()
