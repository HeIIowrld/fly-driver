"""Opt-in assessed native acceleration/tyre sensations; legacy body21 unchanged.

Only current native physical fields are selected. World velocity/pose, heading,
teacher commands and numerical derivatives never appear in this observation.
Two acceleration axes may be assessed inside a hash-bound observed envelope;
this is not a global gravity, real-vehicle or physiological registration.
"""
import hashlib,json,math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX
from .graph import sha256
from .sensor_contract import readonly_array,readonly_mask

EXTENSION_NAMES=(
 'assessed_acceleration_body.forward_m_s2','assessed_acceleration_body.left_m_s2','assessed_acceleration_body.up_m_s2',
 *(f'wheel_slip.native_{i}_unregistered_scalar' for i in range(4)),
 *(f'wheel_load.native_{i}_n' for i in range(4)),
 *(f'tyre_core_temperature.native_{i}_c' for i in range(4)),
 *(f'tyre_pressure.native_{i}_unregistered_units' for i in range(4)))

@dataclass(frozen=True)
class NativeExtensionSample:
 packet_id:int
 received_monotonic_seconds:float
 prefix_sha256:str
 raw19:np.ndarray
 pitch_rad:float
 roll_native_rad:float
 speed_m_s:float

@dataclass(frozen=True)
class NativeExtensionObservation:
 episode_id:str
 packet_id:int
 original_received_monotonic_seconds:float
 decision_monotonic_seconds:float
 values:np.ndarray
 valid:np.ndarray
 contract_sha256:str
 acceleration_out_of_assessed_scope:bool
 thermal_out_of_training_support:tuple
 feature_names:tuple=EXTENSION_NAMES
 schema:str='flybrain.native_multisensory_extension.observation.v1'
 actor_qualified:bool=False
 physiological_registration_verified:bool=False


def decode_native_extension(prefix,receipt):
 if not isinstance(prefix,bytes) or len(prefix)!=PHYSICS_PREFIX.size or not math.isfinite(receipt) or receipt<0:raise ValueError('Exact native prefix and finite receipt required')
 data=PHYSICS_PREFIX.unpack(prefix);f={};offset=0
 for key,kind,count in PHYSICS_LAYOUT:
  values=data[offset:offset+count];f[key]=values[0] if count==1 else values;offset+=count
 raw=np.concatenate([f[k] for k in ['accG','wheelSlip','wheelLoad','tyreCoreTemperature','wheelsPressure']])
 if f['packetId']<0 or f['speedKmh']<0 or not np.isfinite(raw).all() or not np.isfinite([f['pitch'],f['roll'],f['speedKmh']]).all():raise ValueError('Finite native body fields and unsigned packet/speed required')
 return NativeExtensionSample(f['packetId'],float(receipt),hashlib.sha256(prefix).hexdigest(),readonly_array(raw),f['pitch'],f['roll'],f['speedKmh']/3.6)


def assessed_extension_values(raw,pitch,roll,speed,contract):
 raw=np.asarray(raw,float);n=len(raw)
 if raw.shape!=(n,19) or not np.isfinite(raw).all():raise ValueError('Finite current native19 required')
 pitch,roll,speed=[np.asarray(x,float) for x in [pitch,roll,speed]]
 if any(x.shape!=(n,) or not np.isfinite(x).all() for x in [pitch,roll,speed]):raise ValueError('Current native tilt/speed envelope required')
 bounds=contract['acceleration_assessed_envelope'];inside=np.ones(n,bool)
 for name,x in [('pitch_native_rad',pitch),('roll_native_rad',roll),('speed_m_s',speed)]:
  lo,hi=bounds[name];inside&=(x>=lo-1e-6)&(x<=hi+1e-6)
 values=raw.copy();values[:,:3]=raw[:,[2,0,1]]*9.80665
 valid=np.ones((n,19),bool);valid[:,:3]=inside[:,None]&np.asarray(contract['assessed_acceleration_axes'],bool)
 valid[:,7:11]=raw[:,7:11]>=0;valid[:,11:15]=raw[:,11:15]>-273.15;valid[:,15:19]=raw[:,15:19]>=0
 values[~valid]=0
 thermal=contract['thermal_training_support_c'];ood=(raw[:,11:15]<np.asarray(thermal['minimum']))|(raw[:,11:15]>np.asarray(thermal['maximum']))
 return values,valid,~inside,ood


class NativeMultisensoryExtensionEncoder:
 def __init__(self,contract_file,*,native_identity):
  path=Path(contract_file);self.contract=json.loads(path.read_text(encoding='utf-8'));c=self.contract
  if c['schema'] not in ['flybrain.native_multisensory_extension.contract.v1','flybrain.native_multisensory_extension.contract.v2'] or tuple(c['feature_names'])!=EXTENSION_NAMES or c['native_acceleration_indices']!=[2,0,1]:raise ValueError('Explicit assessed native body extension required')
  if native_identity!=c['native_identity']:raise ValueError('Assessed native car/version identity differs')
  if sha256(c['audit_result'])!=c['audit_result_sha256']:raise ValueError('Raw source assessment changed')
  evidence=json.loads(Path(c['audit_result']).read_text(encoding='utf-8'))
  if evidence['status']!='completed_qualified_native_multisensory_audit':raise ValueError('Original native assessment required')
  if c['schema'].endswith('.v1'):
   if c['assessed_acceleration_axes']!=[True,True,False] or evidence['candidate_axes_passed_all_episodes']!=c['assessed_acceleration_axes']:raise ValueError('Original two-axis assessment required')
  else:
   if sha256(c['bandwidth_assessment_result'])!=c['bandwidth_assessment_sha256']:raise ValueError('Bandwidth assessment changed')
   matched=json.loads(Path(c['bandwidth_assessment_result']).read_text(encoding='utf-8'))
   if matched['status']!='completed_bandwidth_matched_acceleration_diagnostic' or len(matched['episodes'])!=3 or c['assessed_acceleration_axes']!=[True,True,True] or not all(all(r['fast_frames']['candidate_axis_threshold_passed']) and r['fast_frames']['n']>=100 for r in matched['episodes']):raise ValueError('Three-axis bandwidth assessment does not support contract')
   manifest=json.loads((Path(c['bandwidth_assessment_result']).parent/'manifest.json').read_text(encoding='utf-8'))
   if manifest['config']['original_audit_sha256']!=c['audit_result_sha256']:raise ValueError('Bandwidth evidence belongs to another native source')
  if sha256(c['installed_sdk_file'])!=c['installed_sdk_sha256']:raise ValueError('Installed SDK source changed')
  self.maximum_age=c['maximum_host_age_seconds']
  if not 0<self.maximum_age<=.05:raise ValueError('Bounded original50ms age required')
  self.contract_sha256=sha256(path);self.episode=None;self.last=None;self.last_decision=None
 def encode(self,sample,*,episode_id,decision_monotonic_seconds):
  if not isinstance(sample,NativeExtensionSample) or not isinstance(episode_id,str) or not episode_id or not math.isfinite(decision_monotonic_seconds):raise ValueError('Explicit native sample, episode and clock required')
  if episode_id!=self.episode:self.episode=episode_id;self.last=None;self.last_decision=None
  if self.last_decision is not None and decision_monotonic_seconds<self.last_decision:raise ValueError('Decision clock reversed')
  if self.last is not None:
   if sample.packet_id<self.last.packet_id:raise ValueError('Packet reset requires episode boundary')
   if sample.packet_id==self.last.packet_id:
    if sample.prefix_sha256!=self.last.prefix_sha256:raise ValueError('Repeated native packet has different bytes')
    sample=self.last
   elif sample.received_monotonic_seconds<self.last.received_monotonic_seconds:raise ValueError('Native receipt clock reversed')
  age=decision_monotonic_seconds-sample.received_monotonic_seconds
  if age<0:raise ValueError('Native sample is after decision')
  values,valid,outside,thermal=assessed_extension_values(sample.raw19[None],[sample.pitch_rad],[sample.roll_native_rad],[sample.speed_m_s],self.contract)
  if age>self.maximum_age:valid[:]=False;values[:]=0
  self.last=sample;self.last_decision=float(decision_monotonic_seconds)
  return NativeExtensionObservation(episode_id,sample.packet_id,sample.received_monotonic_seconds,float(decision_monotonic_seconds),readonly_array(values[0]),readonly_mask(valid[0]),self.contract_sha256,bool(outside[0]),tuple(map(bool,thermal[0])),schema='flybrain.native_multisensory_extension.observation.v2' if self.contract['schema'].endswith('.v2') else 'flybrain.native_multisensory_extension.observation.v1')
