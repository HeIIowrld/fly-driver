"""Independent qualified-episode readout/metrics/packet sidecar audit."""
import argparse,base64,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_multisensory_extension import decode_native_extension,assessed_extension_values
from flybrain_drive.sac_unified_student import read
from scripts.diagnose_sac_anticipation import load_verified
from scripts.fit_bounded_sac_steering import pre_mask


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.evaluation);r=read(root/'result.json');m=read(root/'manifest.json')['config'];entry=m['entry'];d=load_verified(entry)
    if r['status']!='completed_qualified_native_candidate_evaluation' or sha256(root/'predictions.npz')!=r['predictions_sha256'] or sha256(root/'executed-entrypoint.py')!=m['program_sha256']:raise ValueError('Completed original evaluation required')
    plan_path=Path(m['arguments']['candidate_plan']);assert sha256(plan_path)==m['candidate_plan_sha256'];plan=read(plan_path);head_path=Path(plan['native']['head'])/'native-steering.pt';assert sha256(head_path)==plan['native']['head_sha256']
    state=torch.load(head_path,map_location='cpu',weights_only=True)['readout'];state={k:v.numpy().astype(float) for k,v in state.items()}
    native_fit=read(Path(plan['native']['fit'])/'manifest.json')['config'];encoder=read(Path(native_fit['arguments']['extension'])/'contract.json')
    cfg=read(Path(entry['data'])/'manifest.json')['config'];native=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';assert sha256(native)==m['native_source_sha256'];needed=set(map(int,d['body_packet']));samples={}
    for line in native.open(encoding='utf-8'):
        auth=json.loads(line)['authority'];pid=auth['packet_id']
        if pid in samples or pid not in needed:continue
        samples[pid]=decode_native_extension(base64.b64decode(auth['prefix_base64'])[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
    with np.load(root/'predictions.npz') as z:
        pred=z['prediction'];mus=z['muscles'];assert pred.shape==(len(d['labels']),3) and mus.shape==(len(pred),6)
        logits=((mus-state['training_reference'])/state['training_scale'])@state['projection.weight'].T+state['projection.bias'];independent=np.column_stack([np.tanh(logits[:,0]),1/(1+np.exp(-logits[:,1])),1/(1+np.exp(-logits[:,2]))]);difference=float(abs(independent-pred).max());assert difference<=2e-6
        for i,pid in enumerate(d['body_packet']):
            sample=samples[int(pid)];assert sample.received_monotonic_seconds==d['body_receipt'][i]
            values,valid,outside,ood=assessed_extension_values(sample.raw19[None],[sample.pitch_rad],[sample.roll_native_rad],[sample.speed_m_s],encoder)
            np.testing.assert_allclose(z['native_values'][i],values[0],atol=2e-6,rtol=1e-6);np.testing.assert_array_equal(z['native_valid'][i],valid[0]);np.testing.assert_array_equal(z['thermal_ood'][i],ood[0]);assert z['acceleration_outside'][i]==outside[0]
        pre=pre_mask(d,m['sign_evidence']['sign']);masks={'full_lap':d['fit_mask'],'precorner':pre,'speed_below100':d['fit_mask']&(d['body'][:,16]*3.6<100),'speed_atleast100':d['fit_mask']&(d['body'][:,16]*3.6>=100)}
        for name,mask in masks.items():
            if not mask.any():continue
            error=pred[mask].astype(float)-d['labels'][mask].astype(float);metrics=r['metrics'][name]
            np.testing.assert_allclose(np.sqrt(np.mean(error**2,axis=0)),metrics['rmse'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.mean(abs(error),axis=0),metrics['mae'],atol=1e-12,rtol=0);assert abs(float(np.mean(error**2*[4,1,1]))-metrics['weighted_mse'])<1e-12
        assert (z['output_body_age']<=.05).all() and np.isfinite(z['inference_seconds']).all()
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),evaluation_result_sha256=sha256(root/'result.json'),seed=None,optimizer_updates=0,vehicle_control_sent=False))
    write_json(out/'result.json',dict(status='completed_independent_qualified_native_candidate_audit',events=len(pred),full_lap_labels=int(d['fit_mask'].sum()),precorner_labels=int(pre.sum()),independent_muscle_readout_max_difference=difference,original_packet_receipts_and_native_sidecar_verified=True,actor_qualified=False,vehicle_control_sent=False,termination_reason='Independent muscle readout, qualified scoring masks/metrics and original native sidecar joins verified; no independent full-neural trajectory, live control or lap qualification.'))

if __name__=='__main__':main()
