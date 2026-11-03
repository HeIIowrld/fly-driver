"""Evaluate declared source-model flash gate against preserved model 000."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.analyze_flyvis_probe import load_dataset,numpy_flash_index

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--flash',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),flash_result_sha256=sha256(Path(a.flash)/'result.json'),actor=False));out=dict(status='failed',actor_deployed=False,optimizer_updates=0)
 try:
  from flyvis.analysis.flash_responses import flash_response_index
  refpath=Path('D:/CodexResearchRuns/flybrain-t4d-prespecified-models-comparison-20261102-01/result.json')
  if sha256(refpath)!=plan['reference_comparison_sha256'] or sha256(plan['candidate'])!=plan['candidate_sha256']:raise ValueError('declared sources differ')
  manifest=read(Path(a.flash)/'manifest.json')
  if manifest['config']['checkpoint_sha256']!=plan['candidate_sha256']:raise ValueError('candidate source differs')
  config=read('configs/flyvis-author-grid-probe-20261029.json');d=load_dataset(a.flash,config['flash']);names=d.cell_type.values;ref=read(refpath)['reports']['000']['cells'];cells=[f'T4{x}' for x in 'abcd']+[f'T5{x}' for x in 'abcd'];rows=[];parity=0.
  for radius in (-1,6):
   v=flash_response_index(d.copy(deep=True),radius).values.squeeze();independent=numpy_flash_index(d,radius);np.testing.assert_allclose(v,independent,rtol=2e-6,atol=2e-7);parity=max(parity,float(np.max(np.abs(v-independent))))
   for c in cells:
    value=float(v[int(np.flatnonzero(names==c)[0])]);baseline=float(ref[c]['flash_full' if radius==-1 else 'flash_radius6']);sign=value>0 if c.startswith('T4') else value<0;change=abs(value-baseline);passed=bool(sign and (c=='T4d' or change<=.1));rows.append(dict(radius=radius,cell=c,baseline=baseline,candidate=value,change=change,expected_polarity=bool(sign),passed=passed))
  out.update(status='completed_t4d_candidate_flash_gate',rows=rows,passed=all(x['passed'] for x in rows),passed_conditions=sum(x['passed'] for x in rows),total_conditions=len(rows),author_numpy_parity_max=parity,actor_qualified=False,training_admissible=False,interpretation='Full author flashes; one radius held from gradient; screen only, not full biological or driving qualification.')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',out)
 print(json.dumps(out,allow_nan=False))
 if out['status']!='completed_t4d_candidate_flash_gate':raise SystemExit(1)
if __name__=='__main__':main()
