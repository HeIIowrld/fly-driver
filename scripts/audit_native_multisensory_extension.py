"""Independent streaming-codec and normalization audit for native extensions."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder,decode_native_extension
from scripts.diagnose_sac_anticipation import load_verified,read
from scripts.sac_cohort_contract import DEFAULT_SPLIT,three_episode_cohort

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extension',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default=DEFAULT_SPLIT);a=p.parse_args()
 root=Path(a.extension);result=read(root/'result.json');contract=read(root/'contract.json')
 entries=three_episode_cohort(a.split);items=[load_verified(e) for e in entries]
 run=begin_run(a.output,dict(arguments=vars(a),seed=780604,program_sha256=sha256(__file__),extension_result_sha256=sha256(root/'result.json'),entries=entries,scope='Independent streaming native-byte codec, same-packet causal observation and TRAIN normalization audit',actor_deployed=False,vehicle_control_sent=False))
 out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False);reports=[];train=[];train_valid=[]
 try:
  if result['status']!='completed_assessed_native_multisensory_preparation' or sha256(root/'contract.json')!=result['contract_sha256']:raise ValueError('Prepared source changed')
  for i,(entry,d) in enumerate(zip(entries,items)):
   cfg=read(Path(entry['data'])/'manifest.json')['config'];source=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';cert=read(Path(cfg['arguments']['qualification'])/'result.json')
   assert sha256(source)==cert['native_body_sha256'];samples={};last=-1
   with source.open(encoding='utf-8') as f:
    for line in f:
     row=json.loads(line);authority=row['authority'];raw=base64.b64decode(authority['prefix_base64'],validate=True)
     assert hashlib.sha256(raw).hexdigest()==authority['prefix_sha256']
     pid=authority['packet_id']
     if pid==last:continue
     sample=decode_native_extension(raw[:PHYSICS_PREFIX.size],authority['received_monotonic_seconds']);assert sample.packet_id==pid
     samples[pid]=sample;last=pid
   encoder=NativeMultisensoryExtensionEncoder(root/'contract.json',native_identity=contract['native_identity']);values=[];valid=[];maximum_error=0.
   path=root/f'extension-{i}.npz';assert sha256(path)==result['files_sha256'][path.name]
   with np.load(path) as prepared:
    for j,pid in enumerate(d['body_packet']):
     observation=encoder.encode(samples[int(pid)],episode_id=entry['episode_id'],decision_monotonic_seconds=float(d['decision_time'][j]))
     assert observation.original_received_monotonic_seconds==d['body_receipt'][j]
     assert observation.packet_id==pid
     assert observation.valid[2]==contract['assessed_acceleration_axes'][2]
     if not contract['assessed_acceleration_axes'][2]:assert observation.values[2]==0
     np.testing.assert_array_equal(observation.valid,prepared['valid'][j])
     error=float(abs(observation.values-prepared['values'][j]).max());maximum_error=max(maximum_error,error)
     np.testing.assert_allclose(observation.values,prepared['values'][j],atol=2e-6,rtol=1e-6)
     # Independently check the fixed native axes and withheld vertical channel.
     expected=samples[int(pid)].raw19[[2,0]]*9.80665
     np.testing.assert_allclose(observation.values[:2],expected,atol=1e-8,rtol=0)
     if contract['assessed_acceleration_axes'][2]:np.testing.assert_allclose(observation.values[2],samples[int(pid)].raw19[1]*9.80665,atol=1e-8,rtol=0)
     values.append(observation.values);valid.append(observation.valid)
   values=np.asarray(values);valid=np.asarray(valid);assert not d['body_valid'][:,5:8].any()
   reports.append(dict(episode=i,events=len(values),native_codec_max_difference_from_float32_sidecar=maximum_error,all_receipts_original=True,all_original_acceleration_masks_false=True,new_vertical_axis_assessed=contract['assessed_acceleration_axes'][2]))
   if i<2:train.append(values[d['fit_mask']]);train_valid.append(valid[d['fit_mask']])
  x=np.concatenate(train);m=np.concatenate(train_valid);floors=np.array([.1,.1,1,*([.01]*4),*([10]*4),*([1]*4),*([.01]*4)])
  with np.load(root/'training-normalization.npz') as z:
   for k in range(19):
    v=x[m[:,k],k];mean=v.mean() if len(v) else 0.;scale=max(v.std(),floors[k]) if len(v) else 1.
    np.testing.assert_allclose(z['reference'][k],mean,atol=1e-5,rtol=1e-6);np.testing.assert_allclose(z['scale'][k],scale,atol=1e-5,rtol=1e-6)
  out.update(status='completed_native_extension_streaming_audit',episodes=reports,training_only_normalization_verified=True,original_body21_and_masks_unchanged=True,actor_qualified=False,termination_reason='Streaming native observations and training-only normalization verified')
  out['termination_reason']=f'All{sum(r["events"] for r in reports)} original observations reconstructed causally from native prefixes; no future or world fields in extension'
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit streaming audit failure')
 write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
