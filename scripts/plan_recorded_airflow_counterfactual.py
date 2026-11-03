"""Predeclare a non-deployment wind-fixture intervention against fixed recorded inputs."""
import json
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');fit=Path('D:/CodexResearchRuns/flybrain-synthetic-airflow-fit-20261102-03');model=fit/'synthetic-airflow-feedback.pt';result=json.loads((fit/'result.json').read_text(encoding='utf8'))
 if result['status']!='completed_synthetic_airflow_training' or result['predeclared_pass'] is not True or sha256(model)!=result['checkpoint_sha256'] or result['actor_qualified'] is not False:raise ValueError('Exact non-deployed synthetic checkpoint required')
 sources={}
 for key,name in [('positive','20261031-wind-motion-v2-recording'),('negative','20261031-wind-negative-motion-recording'),('zero','20261031-wind-zero-motion-recording')]:
  root=base/name;r=json.loads((root/'result.json').read_text(encoding='utf8'))
  if r['status']!='completed' or sha256(root/'diagnostic-frames.json')!=r['metadata_sha256']:raise ValueError('Original diagnostic source required')
  sources[key]={'path':str(root),'result_sha256':sha256(root/'result.json'),'metadata_sha256':r['metadata_sha256']}
 plan=dict(schema='flybrain.airflow_counterfactual_plan.v1',checkpoint_sha256=sha256(model),recorded_visual_run='D:/CodexResearchRuns/flybrain-body-owner-live-20261102-07',wind_sources=sources,mode='offline numerical counterfactual only',mapping='select each source speed>1m/s; repeat each candidate by relative sequence index over independent stationary episode',conditions=['missing','zero','positive','negative'],graph='fixed synthetic-trained airflow boundary and original unified circuit; independent recurrent state per condition',fit_real_data=False,optimizer_updates=0,source_verified=False,physical_synchronization_verified=False,actor_qualified=False,expectation='compare positive/negative candidate signed steering against zero; not a real steering label, calibration, physical clock or success criterion for driving')
 run=begin_run('D:/CodexResearchRuns/flybrain-airflow-recorded-fixture-plan-20261102-01',dict(program_sha256=sha256(__file__),optimizer_updates=0));write_json(run/'plan.json',plan);write_json(run/'result.json',dict(status='sealed_counterfactual_plan',plan_sha256=sha256(run/'plan.json'),actor_qualified=False));print(json.dumps(plan))
if __name__=='__main__':main()
