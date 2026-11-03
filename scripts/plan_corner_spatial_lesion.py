"""Predeclare removal of one spatial visual path in known synthetic corners."""
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 prior=Path('D:/CodexResearchRuns/flybrain-corner-wind-reference-20261102-01');ck=Path('D:/CodexResearchRuns/flybrain-synthetic-airflow-fit-20261102-03/synthetic-airflow-feedback.pt')
 plan=dict(schema='flybrain.synthetic_corner_spatial_lesion.v1',program_sha256=sha256('scripts/evaluate_corner_spatial_lesion.py'),vehicle_sha256=sha256('src/flybrain_drive/synthetic_crosswind_vehicle.py'),combination_sha256=sha256('src/flybrain_drive/unified_airflow_speed.py'),unified_sha256=sha256('src/flybrain_drive/unified_flight_sensorimotor.py'),airflow_checkpoint_sha256=sha256(ck),prior_root=str(prior),prior_result_sha256=sha256(prior/'result.json'),prior_trajectory_sha256=sha256(prior/'zero-wind-trajectory.npz'),seed=883201,initial_speeds_m_s=[3.,6.],bends=[-1,1],wind_left_m_s=[0.],drift_gain=.02,target_x_m=16.,max_steps=121,modes=['spatial_ablated'],intervention='remove learned spatial contrast boundary only, keep all other visual/body paths and fixed speed readout',decision='if any absolute trajectory change >=0.02m, boundary has measurable influence under this synthetic criterion; otherwise prioritize testing direct/temporal paths; neither outcome proves physiological function or adequate driving',limitations='known previously observed synthetic texture, one procedural road and heuristic bicycle; post-failure causal decomposition only',optimizer_updates=0,vehicle_control_sent=False)
 run=begin_run('D:/CodexResearchRuns/flybrain-corner-spatial-lesion-plan-20261102-01',dict(program_sha256=sha256(__file__),optimizer_updates=0));write_json(run/'plan.json',plan);write_json(run/'result.json',dict(status='sealed_corner_spatial_lesion_plan',plan_sha256=sha256(run/'plan.json'),actor_qualified=False));print(plan)
if __name__=='__main__':main()
