"""Fit two frozen-state engineering probes and test spatial ablation; no driving."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
planfile=B/'20261101-muscle-symmetry-diagnostic-plan/plan.json';plan=read(planfile)
if plan['no_fit_on_evaluation'] is not True or plan['feature_scale']!=1e-6 or plan['ridge']!=1.:raise ValueError('exact sealed diagnostic plan required')
sources=[B/'20261101-visual-corner-speed-fit-1',B/'20261101-flight-state-corner-probe-1',B/'20261101-flight-state-frozen-replication-1'];files=[sources[0]/'evaluation.npz',sources[1]/'probe.npz',sources[2]/'replication.npz']
for root,f in zip(sources,files):
 r=read(root/'result.json')
 if sha256(f)!=r['output_sha256']:raise ValueError('input hash mismatch')
run=begin_run(B/'20261101-muscle-symmetry-diagnostic-1',dict(program_sha256=sha256(__file__),plan_sha256=sha256(planfile),input_sha256=[sha256(f) for f in files],optimizer_updates_planned=2,actor=False))
result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,fit_used_evaluation=False)
try:
 with np.load(files[0],allow_pickle=False) as z: fit=z['fit_features'].astype('float64')/1000.;target=z['fit_targets'].astype('float64')
 if fit.shape!=(60,6) or not np.array_equal(np.unique(target),[2.5,3.5]):raise ValueError('fit shape or labels')
 center=fit[target==3.5].mean(axis=0);scale=plan['feature_scale']; fit0=(fit-center)/scale
 def features(x,nonlinear):
  x=(x-center)/scale
  return np.concatenate([x,np.abs(x[:,[1,3,5]]-x[:,[0,2,4]])],axis=1) if nonlinear else x
 def design(x):return np.column_stack((x,np.ones(len(x))))
 predictors={}
 for name,nonlinear in [('linear',False),('absolute_bilateral',True)]:
  d=design(features(fit,nonlinear));reg=np.diag([plan['ridge']]*(d.shape[1]-1)+[0.]);w=np.linalg.solve(d.T@d+reg,d.T@target);predictors[name]=w
 evaluation={}
 with np.load(files[1],allow_pickle=False) as z: evaluation['881403']=(z['muscles6_validation'].astype('float64'),z['muscles6_lesion'].astype('float64'))
 with np.load(files[2],allow_pickle=False) as z:
  for seed in (881411,881412,881413):evaluation[str(seed)]=(z[f'{seed}_muscles6_intact'].astype('float64'),z[f'{seed}_muscles6_lesion'].astype('float64'))
 labels=np.repeat([3.5,2.5,2.5],10); reports={}; arrays={'fit':fit,'fit_target':target,'center':center}
 for name,w in predictors.items():
  nonlinear=name=='absolute_bilateral'; arrays[name+'_weights']=w
  def report(x):
   out=design(features(x,nonlinear))@w;means=out.reshape(3,10).mean(axis=1);return dict(means=means.tolist(),straight_minus_bend=float(means[0]-means[1:].mean()),mse=float(np.mean((out-labels)**2)),straight_bend_margin=float(min(means[0]-means[1],means[0]-means[2])))
  reports[name]=dict(fit_mse=float(np.mean((design(features(fit,nonlinear))@w-target)**2)),fixed_input_gain_per_unit=float(np.max(np.abs(w[:-1]))/scale),conditions={k:{'intact':report(v[0]),'spatial_off':report(v[1])} for k,v in evaluation.items()})
  for k,(intact,lesion) in evaluation.items():arrays[name+'_'+k+'_intact']=design(features(intact,nonlinear))@w;arrays[name+'_'+k+'_lesion']=design(features(lesion,nonlinear))@w
 out=run/'diagnostic.npz';np.savez_compressed(out,**arrays)
 result.update(status='completed_synthetic_muscle_symmetry_diagnostic',models=reports,plan_sha256=sha256(planfile),input_sha256=[sha256(f) for f in files],output_sha256=sha256(out),probe_fits=2,optimizer_updates=0,interpretation='Post-hoc frozen synthetic states, fit and evaluation use adjacent state timing; temporally correlated; unbounded diagnostic predictions are not deployed speed commands or biological proof')
except Exception as e:result.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',result)
print(json.dumps(result,allow_nan=False))
if result['status']!='completed_synthetic_muscle_symmetry_diagnostic':raise SystemExit(1)
