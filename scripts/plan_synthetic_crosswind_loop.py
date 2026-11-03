"""Seal straight-road crosswind closed-loop evaluation before execution."""
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 ck=Path('D:/CodexResearchRuns/flybrain-synthetic-airflow-fit-20261102-03/synthetic-airflow-feedback.pt')
 plan=dict(schema='flybrain.synthetic_crosswind_loop.v1',program_sha256=sha256('scripts/evaluate_synthetic_crosswind_loop.py'),vehicle_sha256=sha256('src/flybrain_drive/synthetic_crosswind_vehicle.py'),unified_sha256=sha256('src/flybrain_drive/unified_flight_sensorimotor.py'),airflow_checkpoint_sha256=sha256(ck),seeds=[882901,882902],wind_left_m_s=[-5.,0.,5.],drift_gain=.02,dt_seconds=.04,steps=81,speed_m_s=3.,road='straight',controls='existing graph steering mapping; speed held constant',modes=['missing','trained'],success='each wind/seed trained final absolute center error strictly lower than missing; no-wind difference <=0.05m; both conditions complete 81 steps',training_data_used=False,actor_deployed=False,vehicle_control_sent=False,interpretation='Hypothetical drift and simplified bicycle; no aerodynamic, physiological or Assetto Corsa claim')
 run=begin_run('D:/CodexResearchRuns/flybrain-crosswind-loop-plan-20261102-01',dict(program_sha256=sha256(__file__),optimizer_updates=0));write_json(run/'plan.json',plan);write_json(run/'result.json',dict(status='sealed_synthetic_crosswind_plan',plan_sha256=sha256(run/'plan.json'),actor_deployed=False));print(plan)
if __name__=='__main__':main()
