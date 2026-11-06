"""Prepare opt-in native sensations without changing legacy body21 or masks."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_multisensory_extension import EXTENSION_NAMES,assessed_extension_values
from scripts.diagnose_sac_anticipation import load_verified,read
from scripts.sac_cohort_contract import DEFAULT_SPLIT,three_episode_cohort

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);p.add_argument('--bandwidth');p.add_argument('--split',default=DEFAULT_SPLIT);a=p.parse_args()
 root=Path(a.audit);evidence=read(root/'result.json')
 if evidence['status']!='completed_qualified_native_multisensory_audit' or evidence['candidate_axes_passed_all_episodes'][:2]!=[True,True]:raise ValueError('Assessed forward/left evidence required; vertical still requires matched-bandwidth assessment')
 if not a.bandwidth and evidence['candidate_axes_passed_all_episodes']!=[True,True,False]:raise ValueError('Legacy v1 requires the original two-axis assessment; use matched bandwidth for v2')
 entries=three_episode_cohort(a.split);data=[load_verified(e) for e in entries]
 sdk=read(root/'manifest.json')['config'];sdkfile=Path(sdk['installed_primary_sdk']);assert sha256(sdkfile)==sdk['installed_primary_sdk_sha256']
 run=begin_run(a.output,dict(arguments=vars(a),seed=780602,program_sha256=sha256(__file__),assessment_sha256=sha256(root/'result.json'),entries=entries,component_only=True,actor_deployed=False,vehicle_control_sent=False))
 train_temperature=np.concatenate([d['body'][d['fit_mask'],12:16] for d in data[:2]])
 pitch=np.concatenate([d['body'][:,0] for d in data]);roll=np.concatenate([-d['body'][:,1] for d in data]);speed=np.concatenate([d['body'][:,16] for d in data])
 contract=dict(schema='flybrain.native_multisensory_extension.contract.v1',feature_names=list(EXTENSION_NAMES),native_identity=dict(car_model='bmw_z4_gt3',shared_memory_version='1.7',ac_version='1.16.4'),audit_result=str(root/'result.json'),audit_result_sha256=sha256(root/'result.json'),installed_sdk_file=str(sdkfile),installed_sdk_sha256=sha256(sdkfile),assessed_acceleration_axes=[True,True,False],native_acceleration_indices=[2,0,1],native_acceleration_signs=[1,1,1],acceleration_g_to_m_s2=9.80665,maximum_host_age_seconds=.05,acceleration_assessed_envelope=dict(pitch_native_rad=[float(pitch.min()),float(pitch.max())],roll_native_rad=[float(roll.min()),float(roll.max())],speed_m_s=[float(speed.min()),float(speed.max())]),thermal_training_support_c=dict(minimum=train_temperature.min(0).tolist(),maximum=train_temperature.max(0).tolist()),units=dict(acceleration='Assessed body-forward/left kinematic candidate m/s2; native G times9.80665; vertical remains invalid',wheel_slip='Native scalar, not claimed slip ratio/angle or friction utilization',wheel_load='Native load; installed SDK specifies N; offline original car only',tyre_core_temperature='Native core temperature; installed SDK specifies Celsius',tyre_pressure='Native pressure units, no PSI/kPa conversion claimed'),limits='Observed BMW/Monza range only. Acceleration assessment uses noncausal diagnostic references, not actor inputs. Exact physics/exposure atomicity, large bank/grade, other cars and cold/overheated traction adaptation remain unverified.',source_body21_and_certificates_unchanged=True,actor_qualified=False,physiological_registration_verified=False)
 if a.bandwidth:
  matched_root=Path(a.bandwidth);matched=read(matched_root/'result.json');manifest=read(matched_root/'manifest.json')
  if matched['status']!='completed_bandwidth_matched_acceleration_diagnostic' or manifest['config']['original_audit_sha256']!=sha256(root/'result.json') or not all(all(r['fast_frames']['candidate_axis_threshold_passed']) and r['fast_frames']['n']>=100 for r in matched['episodes']):raise ValueError('Matched derivative/acceleration temporal assessment required')
  contract.update(schema='flybrain.native_multisensory_extension.contract.v2',assessed_acceleration_axes=[True,True,True],bandwidth_assessment_result=str(matched_root/'result.json'),bandwidth_assessment_sha256=sha256(matched_root/'result.json'),assessment_protocol='Discrete LS derivative-equivalent acceleration kernel; diagnostic only; actor uses causal current native G',high_frequency_physics_atomicity_verified=False)
  contract['units']['acceleration']='Assessed body forward/left/up kinematic candidate m/s2, native G times9.80665; assessment limited to observed conditions and matched temporal bandwidth'
 write_json(run/'contract.json',contract);reports=[];hashes={};fit=[];masks=[]
 for i,(entry,d) in enumerate(zip(entries,data)):
  path=root/f'native-auxiliary-{i}.npz'
  if sha256(path)!=evidence['files_sha256'][path.name]:raise ValueError('Recovered raw native sidecar changed')
  with np.load(path,allow_pickle=False) as z:
   np.testing.assert_array_equal(z['body_packet'],d['body_packet']);np.testing.assert_array_equal(z['body_receipt'],d['body_receipt']);np.testing.assert_array_equal(z['decision_time'],d['decision_time']);raw=z['source_values']
  value,valid,outside,thermal=assessed_extension_values(raw,d['body'][:,0],-d['body'][:,1],d['body'][:,16],contract)
  age=d['decision_time']-d['body_receipt'];assert (age>=0).all();valid[age>.05]=False;value[~valid]=0
  assert (not valid[:,2].any() if not a.bandwidth else valid[:,2].all()) and not d['body_valid'][:,5:8].any()
  file=run/f'extension-{i}.npz';np.savez_compressed(file,values=value.astype(np.float32),valid=valid,body_packet=d['body_packet'],body_receipt=d['body_receipt'],decision_time=d['decision_time'],thermal_out_of_training_support=thermal,acceleration_out_of_assessed_scope=outside)
  hashes[file.name]=sha256(file)
  reports.append(dict(episode=i,events=len(value),valid_counts=valid.sum(0).tolist(),out_of_assessed_acceleration_scope=int(outside.sum()),thermal_out_of_training_support_counts=thermal.sum(0).tolist(),same_packet_receipt_join=True,old_acceleration_masks_unchanged=True))
  if i<2:fit.append(value[d['fit_mask']]);masks.append(valid[d['fit_mask']])
 fit=np.concatenate(fit);masks=np.concatenate(masks);reference=np.zeros(19,np.float32);scale=np.ones(19,np.float32)
 floors=np.array([.1,.1,1,*([.01]*4),*([10]*4),*([1]*4),*([.01]*4)])
 for j in range(19):
  x=fit[masks[:,j],j]
  if len(x):reference[j]=x.mean();scale[j]=max(float(x.std()),float(floors[j]))
 np.savez_compressed(run/'training-normalization.npz',reference=reference,scale=scale)
 hashes['training-normalization.npz']=sha256(run/'training-normalization.npz')
 result=dict(status='completed_assessed_native_multisensory_preparation',episodes=reports,contract_sha256=sha256(run/'contract.json'),files_sha256=hashes,normalization_training_labels=len(fit),source_body21_unchanged=True,eligible_for_experimental_offline_component_fit=True,training_admissible=False,actor_qualified=False,vehicle_control_sent=False,termination_reason=f'Same-packet assessed{sum(contract["assessed_acceleration_axes"])}-axis/tyre extension and TRAIN-only normalization ready; no old input/certificate modified or actor promoted')
 write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
