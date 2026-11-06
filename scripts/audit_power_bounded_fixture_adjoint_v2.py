"""Audit recorded fixture adjoint evidence and the fixed-neutral objective."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 p=read(ROOT/'plan-01.json');reports={};loss_difference=None
 for number in [1,2,3]:
  path=ROOT/f'fixture-visual-adjoint-0{number}';r=read(path/'result.json');assert sha256(path/'values.npz')==r['values_sha256'];assert r['maximum_numeric_prediction_difference']==0
  with np.load(path/'values.npz') as z:
   assert np.array_equal(z['stop_all_prediction'],z['frozen_parameters_prediction']);g0=z['stop_all_gradient'].astype(float);g1=z['frozen_parameters_gradient'].astype(float);difference=g1-g0;direction=difference/np.linalg.norm(difference)
   for mode,g in [('stop_all',g0),('frozen_parameters',g1)]:assert abs(np.linalg.norm(g)-r['reports'][mode]['visual_gradient_norm'])<1e-6 and r['reports'][mode]['speed_and_native_grad_is_none']
   if number==3:
    x=z['stop_all_prediction'];anchor=z['frozen_neutral_anchor'];margin=-(x[0,:,0]-x[1,:,0])/2;loss=p['image_direction_weight']*(np.maximum(p['image_target_margin']-margin,0)**2).mean()+p['image_anchor_weight']*(((x[0]+x[1])*.5-anchor)**2*np.array([4.,1.,1.])).sum(1).mean();loss_difference=float(loss)-r['reports']['stop_all']['registered_image_loss'];assert abs(loss_difference)<1e-8
  rows=[]
  for row in r['directional_derivatives']:
   assert abs(np.sum(g0*direction)-row['stop_all_analytic'])<1e-6 and abs(np.sum(g1*direction)-row['frozen_parameters_analytic'])<1e-6;fd=row['finite_difference'];rows.append(dict(row,stop_all_relative_difference=abs(row['stop_all_analytic']-fd)/abs(fd),frozen_parameters_relative_difference=abs(row['frozen_parameters_analytic']-fd)/abs(fd)))
  reports[str(number)]=dict(gradient_difference_norm=r['gradient_difference_norm'],maximum_numeric_prediction_difference=0,directional_derivatives=rows,scope='01 signed margin;02 full loss with moving neutral target;03 correct per-update objective with frozen neutral target')
 selected=next(x for x in reports['3']['directional_derivatives'] if x['step']==.0001);assert selected['stop_all_relative_difference']>.5 and selected['frozen_parameters_relative_difference']<.01
 write_json(ROOT/'fixture-adjoint-audit-01.json',dict(program_sha256=sha256(__file__),registered_loss_numpy_difference=loss_difference,registered_loss_tolerance=1e-8,status='completed_fixture_visual_adjoint_evidence_audit',reports=reports,registered_loss_comparison_at_step1e_minus4=selected,helper_only_not_applied_to_current_training=True,scope='One TRAIN retimed image fixture and selected gradient direction, not all-parameter gradient accuracy or a sole cause of poor visual driving.',actor_qualified=False,vehicle_control_sent=False));print(selected)
if __name__=='__main__':main()
