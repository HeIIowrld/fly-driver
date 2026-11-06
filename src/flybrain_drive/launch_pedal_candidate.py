"""Strict experimental restoration of genuinely supervised flight pedal rows."""
from pathlib import Path
import torch
from .graph import sha256
from .sac_unified_student import read
from .current_flight_runtime import load_current_flight_runtime
from .dual_reference_candidate import load_dual_reference_runtime
from .learned_flight_pedal_readout import LearnedFlightPedalReadout

def load_launch_pedal_runtime(checkpoint,expected_sha256):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config']
 if r['status']!='completed_fixed_schedule_flight_pedal_learning' or sha256(path)!=expected_sha256 or expected_sha256!=r['checkpoint_sha256'] or m['development_used_for_fit'] is not False or m['steps']!=r['optimizer_updates'] or any(sha256(f)!=h for f,h in m['source_sha256'].items()) or sha256(m['parent_checkpoint'])!=m['parent_checkpoint_sha256'] or sha256(m['fixed_decoder'])!=m['fixed_decoder_sha256'] or sha256(m['split'])!=m['split_sha256']:raise ValueError('Exact completed source-bound supervised pedal continuation required')
 cp=torch.load(path,map_location='cpu',weights_only=True)
 if set(cp)!={'schema','parent_checkpoint_sha256','fixed_decoder_sha256','actor_kind','pedal_projection','actor_qualified'} or cp['schema']!='flybrain.learned_flight_pedals.experimental.v1' or cp['actor_kind']!=m['actor_kind'] or cp['actor_qualified'] is not False or any(cp[k]!=m[k] for k in ['parent_checkpoint_sha256','fixed_decoder_sha256']):raise ValueError('Exact experimental pedal schema required')
 if m['actor_kind'] not in ['parent','dual_guard']:raise ValueError('Explicit original parent lineage required')
 loader=load_current_flight_runtime if m['actor_kind']=='parent' else load_dual_reference_runtime
 runtime,receipt=loader(m['parent_checkpoint'],m['parent_checkpoint_sha256'])
 try:
  head=LearnedFlightPedalReadout(runtime.head);state=cp['pedal_projection'];expected=head.pedal_projection.state_dict()
  if set(state)!=set(expected) or any(not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Finite original-shaped pedal coefficients required')
  head.pedal_projection.load_state_dict(state,strict=True);head.verify();runtime.head=head.eval().requires_grad_(False)
  return runtime,dict(parent_receipt=receipt,checkpoint_sha256=expected_sha256,actor_kind='launch_'+m['actor_kind'],readout='LearnedFlightPedalReadout; two genuine supervised pedal rows, original steering/circuit frozen',actor_qualified=False,vehicle_control_sent=False)
 except Exception:
  runtime.close();raise

def load_control_actor_runtime(actor_kind,checkpoint,expected_sha256):
 if actor_kind in ('compact_parent','compact_dual_guard'):
  from .compact_pedal_candidate import load_compact_pedal_runtime
  return load_compact_pedal_runtime(actor_kind,checkpoint,expected_sha256)
 if actor_kind in ['balanced_parent','balanced_dual_guard']:
  from .balanced_muscle_candidate import load_balanced_pedal_runtime
  m=read(Path(checkpoint).parent/'manifest.json')['config']
  if actor_kind!='balanced_'+m['actor_kind']:raise ValueError('Balanced lineage differs')
  return load_balanced_pedal_runtime(checkpoint,expected_sha256)
 if actor_kind=='parent':return load_current_flight_runtime(checkpoint,expected_sha256)
 if actor_kind=='dual_guard':return load_dual_reference_runtime(checkpoint,expected_sha256)
 if actor_kind not in ['launch_parent','launch_dual_guard']:raise ValueError('Explicit control actor lineage required')
 m=read(Path(checkpoint).parent/'manifest.json')['config']
 if actor_kind!='launch_'+m['actor_kind']:raise ValueError('Requested learned actor differs from checkpoint lineage')
 return load_launch_pedal_runtime(checkpoint,expected_sha256)
