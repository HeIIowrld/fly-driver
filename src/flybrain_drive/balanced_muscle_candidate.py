"""Strict completed constrained motor registration; no authority or input widening."""
from pathlib import Path
import torch,numpy as np
from .graph import sha256
from .sac_unified_student import read
from .current_flight_runtime import load_current_flight_runtime
from .dual_reference_candidate import load_dual_reference_runtime
from .joint_balance_muscle_readout import JointBalanceMusclePedalReadout

def load_balanced_pedal_runtime(path,expected):
 path=Path(path);m=read(path.parent/'manifest.json')['config'];r=read(path.parent/'result.json')
 if r['status']!='completed_balanced_muscle_pedal_fit' or not r['optimizer_success'] or sha256(path)!=expected or expected!=r['checkpoint_sha256'] or m['development_used_for_fit'] is not False or m['training_mse_ratio']>1 or any(sha256(p)!=h for p,h in m['source_sha256'].items()) or sha256(m['split'])!=m['split_sha256'] or sha256(m['parent_checkpoint'])!=m['parent_checkpoint_sha256']:raise ValueError('Completed TRAIN-constrained exact motor registration required')
 if any(max(v)>1.00001 for v in r['training_high_mse_ratios']):raise ValueError('High TRAIN performance degraded')
 cp=torch.load(path,map_location='cpu',weights_only=True)
 if set(cp)!={'schema','variant','parent_checkpoint_sha256','actor_kind','state','actor_qualified'} or cp['schema']!='flybrain.balanced_muscle_pedals.experimental.v1' or cp['variant']!='joint_balance_rbf' or cp['actor_qualified'] is not False or cp['actor_kind']!=m['actor_kind'] or cp['parent_checkpoint_sha256']!=m['parent_checkpoint_sha256']:raise ValueError('Exact experimental balanced schema required')
 loader=load_current_flight_runtime if m['actor_kind']=='parent' else load_dual_reference_runtime;runtime,receipt=loader(m['parent_checkpoint'],m['parent_checkpoint_sha256'])
 try:
  st=cp['state'];assert set(st)=={'centers','width','offset','pedal_projection'}
  head=JointBalanceMusclePedalReadout(runtime.head,st['centers'],st['width']);head.offset.load_state_dict(st['offset'],strict=True);head.pedal_projection.load_state_dict(st['pedal_projection'],strict=True);head.verify();runtime.head=head.eval().requires_grad_(False)
  return runtime,dict(checkpoint_sha256=expected,parent_receipt=receipt,actor_kind='balanced_'+m['actor_kind'],variant=cp['variant'],actor_qualified=False,vehicle_control_sent=False)
 except BaseException:runtime.close();raise
