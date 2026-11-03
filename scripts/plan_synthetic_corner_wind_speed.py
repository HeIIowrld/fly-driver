"""Predeclare held synthetic corner/wind and variable-speed comparison."""
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 ck=Path('D:/CodexResearchRuns/flybrain-synthetic-airflow-fit-20261102-03/synthetic-airflow-feedback.pt');plan=dict(schema='flybrain.synthetic_corner_wind_speed.v1',program_sha256=sha256('scripts/evaluate_synthetic_corner_wind_speed.py'),vehicle_sha256=sha256('src/flybrain_drive/synthetic_crosswind_vehicle.py'),combination_sha256=sha256('src/flybrain_drive/unified_airflow_speed.py'),unified_sha256=sha256('src/flybrain_drive/unified_flight_sensorimotor.py'),airflow_checkpoint_sha256=sha256(ck),seed=883201,initial_speeds_m_s=[3.,6.],bends=[-1,1],wind_left_m_s=[-5.,5.],drift_gain=.02,target_x_m=16.,max_steps=121,modes=['missing','trained'],success='all eight paired conditions reach x=16m and trained absolute lateral center error is strictly lower; no retries or selection',speed_control='existing fixed flight-state speed boundary with graph throttle/brake mapping',limitations='one synthetic texture and symmetric procedural curves, synthetic lateral drift and bicycle, not AC physics, physiology or autonomous lap',real_fit=False,vehicle_control_sent=False)
 run=begin_run('D:/CodexResearchRuns/flybrain-corner-wind-speed-plan-20261102-01',dict(program_sha256=sha256(__file__),optimizer_updates=0));write_json(run/'plan.json',plan);write_json(run/'result.json',dict(status='sealed_synthetic_corner_wind_plan',plan_sha256=sha256(run/'plan.json'),actor_qualified=False));print(plan)
if __name__=='__main__':main()
