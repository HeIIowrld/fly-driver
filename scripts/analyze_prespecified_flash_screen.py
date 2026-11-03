"""Prespecified flash polarity screen; does not qualify directional function."""
import argparse,json,os
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.analyze_flyvis_probe import load_dataset,numpy_flash_index

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args();planpath=Path('D:/CodexResearchRuns/flybrain-flash-prescreen-plan-20261102-01/plan.json');plan=json.loads(planpath.read_text(encoding='utf8'));config=json.loads(Path('configs/flyvis-author-grid-probe-20261029.json').read_text(encoding='utf8'));os.environ['MPLBACKEND']='Agg'
 from flyvis.analysis.flash_responses import flash_response_index
 run=begin_run(args.output,dict(plan_sha256=sha256(planpath),program_sha256=sha256(__file__),optimizer_updates=0,actor=False));result={'status':'failed','actor_qualified':False,'optimizer_updates':0}
 try:
  cells=[f'T4{x}' for x in 'abcd']+[f'T5{x}' for x in 'abcd'];rows=[]
  for m in [{'id':'001','control':True}]+plan['models']:
   mid=m['id'];folder=Path(plan['control_run'] if m.get('control') else f'D:/CodexResearchRuns/flybrain-flash-prescreen-{mid}-20261102-01');manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'));checkpoint=manifest['config']['checkpoint_sha256']
   if not m.get('control') and (checkpoint!=m['checkpoint_sha256'] or manifest['config']['config_sha256']!=m['config_sha256']):raise ValueError('checkpoint lineage')
   d=load_dataset(folder,config['flash']);names=d.cell_type.values;values={};parity=0.
   for radius in (-1,6):
    original=flash_response_index(d.copy(deep=True),radius).values.squeeze();independent=numpy_flash_index(d,radius);np.testing.assert_allclose(original,independent,rtol=2e-6,atol=2e-7);parity=max(parity,float(np.max(np.abs(original-independent))));values[str(radius)]={cell:float(original[int(np.flatnonzero(names==cell)[0])]) for cell in cells}
   passes=all((v>0 if cell.startswith('T4') else v<0) for group in values.values() for cell,v in group.items());rows.append(dict(id=mid,control=bool(m.get('control')),indices=values,pass_flash=passes,parity_max=parity,checkpoint_sha256=checkpoint))
  previous=json.loads(Path('D:/CodexResearchRuns/flybrain-t4d-model-001-analysis-20261102-01/result.json').read_text(encoding='utf8'));max_diff=max(abs(rows[0]['indices'][rad][cell]-previous['flash_index'][rad][cell]) for rad in ('-1','6') for cell in cells)
  if max_diff>2e-6:raise ValueError('control differs from original analysis')
  result.update(status='completed_prespecified_flash_screen',rows=rows,control_reproduction_max=max_diff,interpretation='Flash polarity screen only: not direction, physiology, vehicle integration or model qualification.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_prespecified_flash_screen':raise SystemExit(1)
if __name__=='__main__':main()
