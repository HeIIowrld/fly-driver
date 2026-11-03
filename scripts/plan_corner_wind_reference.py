"""Predeclare exploratory zero-wind reference after the original failed test."""
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 prior=Path('D:/CodexResearchRuns/flybrain-corner-wind-speed-evaluation-20261102-01');ck=Path('D:/CodexResearchRuns/flybrain-synthetic-airflow-fit-20261102-03/synthetic-airflow-feedback.pt')
 plan=dict(schema='flybrain.exploratory_corner_wind_reference.v1',program_sha256=sha256('scripts/evaluate_synthetic_corner_wind_reference.py'),vehicle_sha256=sha256('src/flybrain_drive/synthetic_crosswind_vehicle.py'),combination_sha256=sha256('src/flybrain_drive/unified_airflow_speed.py'),unified_sha256=sha256('src/flybrain_drive/unified_flight_sensorimotor.py'),airflow_checkpoint_sha256=sha256(ck),prior_root=str(prior),prior_result_sha256=sha256(prior/'result.json'),prior_trajectory_sha256=sha256(prior/'trajectory.npz'),seed=883201,initial_speeds_m_s=[3.,6.],bends=[-1,1],wind_left_m_s=[0.],drift_gain=.02,target_x_m=16.,max_steps=121,modes=['missing','trained'],comparison='absolute change from same-mode zero wind for each previously recorded +/-5m/s case',purpose='explore whether a constant corrective response reduces the imposed wind displacement despite worsening absolute center tracking',limitations='post-hoc to already observed failure, one synthetic track and wind model; cannot replace original failed criterion or qualify actor',optimizer_updates=0,vehicle_control_sent=False)
 run=begin_run('D:/CodexResearchRuns/flybrain-corner-wind-reference-plan-20261102-01',dict(program_sha256=sha256(__file__),optimizer_updates=0));write_json(run/'plan.json',plan);write_json(run/'result.json',dict(status='sealed_corner_wind_reference_plan',plan_sha256=sha256(run/'plan.json'),actor_qualified=False));print(plan)
if __name__=='__main__':main()
