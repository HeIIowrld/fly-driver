"""Independent muscle/readout and temporal binding audit for moving interventions."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.replay);r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];shadow=Path(cfg['arguments']['shadow'])
    if r['status']!='completed_moving_native_sensory_interventions' or sha256(root/'arrays.npz')!=r['arrays_sha256'] or sha256(root/'executed-entrypoint.py')!=cfg['program_sha256']:raise ValueError('Completed original replay binding required')
    if sha256(shadow/'result.json')!=cfg['shadow_result_sha256']:raise ValueError('Shadow result changed')
    plan=cfg['candidate_plan'];head=Path(plan['native']['head'])/'native-steering.pt'
    if sha256(head)!=plan['native']['head_sha256']:raise ValueError('Frozen muscle head changed')
    sd=torch.load(head,map_location='cpu',weights_only=True)['readout'];sd={k:v.numpy().astype(float) for k,v in sd.items()}
    original=read(shadow/'result.json');ticker=read(shadow/'owner-ticker-events.json')
    if sha256(shadow/'owner-ticker-events.json')!=original['ticker_events_sha256']:raise ValueError('Original predictions changed')
    verified=[]
    with np.load(root/'arrays.npz') as z:
        times=z['decision_time'];packets=z['body_packet'];ids=z['capture_id'];reference=z['original_prediction']
        if not np.array_equal(times,[x['decision_time'] for x in ticker]) or not np.array_equal(packets,[x['body_packet'] for x in ticker]) or not np.array_equal(ids,[x['capture_id'] for x in ticker]):raise ValueError('Original episode ordering changed')
        difference=float(np.max(np.abs(reference-np.asarray([x['prediction'] for x in ticker]))))
        if difference>2e-6 or abs(difference-r['original_prediction_max_difference'])>1e-12:raise ValueError('Original output replay differs')
        for branch in r['branches']:
            mode=branch['mode'];pred=z[mode+'_prediction'];muscles=z[mode+'_muscles'];ood=z[mode+'_thermal_ood']
            if pred.shape!=(len(times),3) or muscles.shape!=(len(times),6) or ood.shape!=(len(times),4) or not np.isfinite(muscles).all():raise ValueError('Full finite branch arrays required')
            logits=((muscles-sd['training_reference'])/sd['training_scale'])@sd['projection.weight'].T+sd['projection.bias']
            expected=np.column_stack([np.tanh(logits[:,0]),1/(1+np.exp(-logits[:,1])),1/(1+np.exp(-logits[:,2]))])
            error=float(np.max(np.abs(expected-pred)))
            if error>2e-6:raise ValueError('Independent muscle readout differs')
            for column in range(3):
                delta=[float(v)-float(ref) for v,ref in zip(pred[:,column],reference[:,column])]
                for key,value in [('maximum_absolute',max(map(abs,delta))),('mean',math.fsum(delta)/len(delta)),('rms',math.sqrt(math.fsum(d*d for d in delta)/len(delta)))]:
                    if abs(value-branch['difference_from_original'][key][column])>2e-6:raise ValueError('Intervention statistic differs')
            if int(np.any(ood,axis=1).sum())!=branch['temperature_ood_outputs']:raise ValueError('OOD count differs')
            preparation=None
            if 'preroll_first_prepared_event' in branch:
                records=read(root/(mode+'-preroll.json'));captures=set();indices=[]
                if len(records)!=len(times):raise ValueError('Preparation record missing')
                for i,item in enumerate(records):
                    captures.add(int(ids[i]));prepared=times[i]-times[0]>=cfg['arguments']['preroll_seconds'] and len(captures)>=3
                    if item['decision_time']!=times[i] or item['capture_id']!=ids[i] or item['prepared']!=prepared or item['distinct_captures']!=len(captures) or abs(item['integrated_sensor_seconds']-(times[i]-times[0]))>1e-12:raise ValueError('Sensory preparation differs')
                    if prepared:indices.append(i)
                if indices[0]!=branch['preroll_first_prepared_event'] or not np.array_equal(pred[indices[0]],branch['preroll_first_prepared_prediction']):raise ValueError('Continuation candidate differs')
                preparation=dict(first_prepared_event=indices[0],elapsed_seconds=float(times[indices[0]]-times[0]),same_full_replay_continuation=True)
            verified.append(dict(mode=mode,events=len(times),independent_muscle_readout_max_difference=error,preparation=preparation))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),replay_result_sha256=sha256(root/'result.json'),seed=None,vehicle_control_sent=False,optimizer_updates=0))
    write_json(out/'result.json',dict(status='completed_independent_moving_sensory_replay_audit',events=len(times),original_prediction_max_difference=difference,branches=verified,actor_qualified=False,vehicle_control_sent=False,physical_adaptation_verified=False,action_correctness_verified=False,termination_reason='Original temporal/body/camera ordering, predictions, all muscle readouts and intervention statistics verified; preparation retains continuous predictions. No physical counterfactual, action correctness or lap qualification.'))

if __name__=='__main__':main()
