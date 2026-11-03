"""Predeclare a synthetic-only wind feedback experiment; no real inputs or actuation."""
import json
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');old=read(base/'20261101-adaptive-contrast-plan/plan.json')
 groups={}
 for key in ('fit_sources','validation_sources'):
  groups[key]=[]
  for item in old[key]:
   root=base/item[0];r=read(root/'result.json');f=root/'responses.npz'
   if r['status']!='completed_causal_four_view_synthetic_road' or sha256(f)!=r['responses_sha256']:raise ValueError('Original synthetic source required')
   groups[key].append(dict(folder=item[0],seed=item[1],curvature=item[2],bend_start_m=item[3],response_sha256=sha256(f)))
 if {x['seed'] for x in groups['fit_sources']}&{x['seed'] for x in groups['validation_sources']}:raise ValueError('Distinct synthetic source seeds required')
 plan=dict(schema='flybrain.synthetic_airflow_response_plan.v1',groups=groups,source_plan_sha256=sha256(base/'20261101-adaptive-contrast-plan/plan.json'),unified_source_sha256=sha256('src/flybrain_drive/unified_flight_sensorimotor.py'),airflow_source_sha256=sha256('src/flybrain_drive/flight_airflow_feedback_adapter.py'),role='synthetic functional test only',fit_real_data=False,actor_qualified=False,wind_body_left_m_s=[-5.,5.],neutral_wind=0.,target_steering_delta_per_m_s=-.001,target_power_delta=0.,target_interpretation='arbitrary engineering response sign; not a biological or vehicle-control target',episodes='left/right road bends per seed, same synthetic body in each pair',training='only airflow projection; 8 Adam steps, lr 0.03, truncate before final five frames',validation='two separately predeclared synthetic source seeds; no validation gradients or selection',success='each validation seed: squared target error lower than zero-weight baseline and opposite winds produce opposite signed mean delta; power mean absolute drift <=0.01',risk='fixed screen-road generator and pre-existing seeds have been viewed in earlier project work; not a novel physical or global holdout')
 run=begin_run('D:/CodexResearchRuns/flybrain-synthetic-airflow-plan-20261102-01',dict(program_sha256=sha256(__file__),optimizer_updates=0));write_json(run/'plan.json',plan);write_json(run/'result.json',dict(status='sealed_synthetic_plan',plan_sha256=sha256(run/'plan.json'),fit_real_data=False,actor_qualified=False,optimizer_updates=0));print(json.dumps(plan))
if __name__=='__main__':main()
