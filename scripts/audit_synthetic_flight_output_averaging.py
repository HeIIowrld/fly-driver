"""Inspect cancellation in fixed synthetic output groups; no actor fitting or vehicle control."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run, write_json
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
prior=B/'20261101-flight-state-frozen-replication-1'; result=json.loads((prior/'result.json').read_text());data=prior/'replication.npz'
if result['status']!='completed_synthetic_frozen_probe_replication' or sha256(data)!=result['output_sha256']:raise ValueError('prior evidence mismatch')
plan=json.loads((B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json').read_text());spec=json.loads(Path(plan['model']['spec']).read_text())
if sha256(spec['graph_file'])!=spec['graph_sha256'] or spec['graph_sha256']!=result['graph_sha256']:raise ValueError('graph mismatch')
with np.load(spec['graph_file'],allow_pickle=False) as z: ids=z['ids'][z['roles']=='output'].tolist()
run=begin_run(B/'20261101-flight-output-averaging-audit-2',dict(program_sha256=sha256(__file__),source_sha256=sha256(data),graph_sha256=spec['graph_sha256'],optimizer_updates_planned=0))
r=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_deployed=False)
try:
 comparisons=[]
 with np.load(data,allow_pickle=False) as z:
  for item in result['conditions']:
   seed=item['condition']['seed'];x=z[f'{seed}_wing_outputs_intact'].reshape(3,10,-1);mus=z[f'{seed}_muscles6_intact'].reshape(3,10,6);details={}
   for j,(group,side) in enumerate((('b1','L'),('b1','R'),('b2','L'),('b2','R'),('power','L'),('power','R'))):
    idx=[ids.index(i) for i in spec['motor_groups'][group][side]]; values=x[:,:,idx];avg=values.mean(axis=2)
    if not np.allclose(avg,mus[:,:,j],atol=2e-7,rtol=1e-5):raise ValueError('stored muscle mean mismatch')
    delta=values[1:]-values[0:1]; avg_delta=delta.mean(axis=2); absolute=float(np.mean(np.abs(delta))); net=float(np.mean(np.abs(avg_delta)))
    details[f'{group}_{side}']=dict(neurons=len(idx),mean_absolute_neuron_delta=absolute,absolute_group_mean_delta=net,retained_fraction=net/absolute if absolute>0 else None,mean_signed_bend_deltas=avg_delta.mean(axis=1).tolist())
   delta=mus[1:]-mus[0:1]; common=(delta[:,:,4]+delta[:,:,5])*.5; component=.5*(np.abs(delta[:,:,4])+np.abs(delta[:,:,5])); denominator=float(component.mean()); common_audit=dict(mean_absolute_common_delta=float(np.abs(common).mean()),mean_absolute_side_delta=denominator,retained_fraction=float(np.abs(common).mean())/denominator if denominator>0 else None,mean_signed_bend_deltas=common.mean(axis=1).tolist()); comparisons.append(dict(condition=item['condition'],groups=details,power_common=common_audit))
 r.update(status='completed_output_averaging_audit',comparisons=comparisons,interpretation='Same-frame synthetic bend minus straight contrasts; cancellation ratio describes linear group averages, not physiology or causal intervention')
except Exception as e:r.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',r)
print(json.dumps(r))
if r['status']!='completed_output_averaging_audit':raise SystemExit(1)
