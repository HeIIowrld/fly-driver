"""Causal stationary-reference output counterfactual on saved synthetic histories."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();reports=[];arrays={};source_hashes=[]
 for seed in (881301,881302):
  d=B/f'20261101-unified-road-yaw-{seed}';source=B/f'20261101-synthetic-road-yaw-{seed}';r=read(d/'result.json');sr=read(source/'result.json');cfg=read(source/'manifest.json')['config'];out=d/'responses.npz';src=source/'responses.npz'
  if r['status']!='completed_road_yaw_unified_lesions' or r['training_admissible'] is not False or sr['status']!='completed_causal_synthetic_road_with_yaw' or sha256(out)!=r['output_sha256'] or sha256(src)!=r['source_sha256'] or sha256(src)!=sr['responses_sha256'] or cfg['stationary_seconds']!=.4 or cfg['texture_seed']!=seed:raise ValueError('exact synthetic-only causal source required')
  with np.load(out,allow_pickle=False) as z:y=z['intents'][0,:,:,:,0].copy();times=z['times'].copy();bends=z['bends'].copy();yaws=z['yaws'].copy()
  with np.load(src,allow_pickle=False) as z:retina=z['retina'].copy()
  if y.shape!=(3,3,41) or not np.isfinite(y).all() or np.any(np.abs(y)>=1) or not np.allclose(times,np.arange(41)*.04,atol=1e-12,rtol=0) or not np.array_equal(bends,[0,-1,1]) or not np.allclose(yaws,[0.,-.12,.12]) or not np.all(retina[:,:,:11]==retina[:,:,:1]):raise ValueError('identical static 11-frame prefixes required')
  latent=np.arctanh(y.astype(np.float64));reference=latent[:,:,8:11].mean(axis=2);corrected=np.tanh(latent-reference[:,:,None]);raw_final=y[:,:,-5:].mean(axis=2);corr_final=corrected[:,:,-5:].mean(axis=2)
  report={'seed':seed,'reference_frames':[8,9,10],'reference_ready_time_s':float(times[10]),'raw_bend_delta_by_yaw':(raw_final[2]-raw_final[1]).tolist(),'corrected_bend_delta_by_yaw':(corr_final[2]-corr_final[1]).tolist(),'raw_final_turns':raw_final.tolist(),'corrected_final_turns':corr_final.tolist(),'raw_zero_yaw_bend_signs':np.sign(raw_final[1:,0]).tolist(),'corrected_zero_yaw_bend_signs':np.sign(corr_final[1:,0]).tolist(),'source_sha256':sha256(out)};reports.append(report);arrays[f'raw_{seed}']=y;arrays[f'corrected_{seed}']=corrected;arrays[f'reference_{seed}']=reference;source_hashes.append(sha256(out))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),sources_sha256=source_hashes,reference='first eleven 40ms stationary synthetic frames; mean unbounded output of last three; frozen after frame 10',optimizer_updates=0,actor_deployed=False,real_data_used_for_fit=False));np.savez_compressed(run/'counterfactual.npz',**arrays)
 result={'status':'completed_stationary_reference_road_counterfactual','reports':reports,'output_sha256':sha256(run/'counterfactual.npz'),'optimizer_updates':0,'actor_qualified':False,'vehicle_control_sent':False,'interpretation':'post-hoc output-only transform on previously examined synthetic scenes; no independent validation, calibrated actual stationary sensor or closed-loop proof'};write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
