"""Recompute saved readout and same-body experiment metrics independently."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.sustained_turn_diagnostic import steering_yaw_alignment,sustained_onsets
from scripts.sac_steering_experiment import dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fit','exchange','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    fit=Path(a.fit);exchange=Path(a.exchange)
    parent=Path('runs/steering-precorners-20261103/optical-pre16-fit-01')
    r=read(fit/'result.json');er=read(exchange/'result.json');selection=read(fit/'train-selection.json')
    assert r['status']=='completed_bounded_steering_readout'
    assert er['status']=='completed_same_body_precorner_visual_exchange'
    assert sha256(fit/'steering.pt')==r['checkpoint_sha256']
    assert sha256(fit/'predictions.npz')==r['predictions_sha256']
    assert sha256(exchange/'predictions.npz')==er['predictions_sha256']
    cp=torch.load(fit/'steering.pt',map_location='cpu',weights_only=True)
    old=torch.load(parent/'steering.pt',map_location='cpu',weights_only=True)
    for component in ['adapter','optical']:
        for k,v in old[component].items():assert torch.equal(v,cp[component][k])
    for k,v in old['readout'].items():
        assert torch.equal(v[1:] if k.startswith('projection.') else v,
                           cp['readout'][k][1:] if k.startswith('projection.') else cp['readout'][k])
    assert not selection['development_used_for_selection']
    assert selection['chosen']==min(selection['candidates'],key=lambda x:x['score'])
    entries,items,_=dataset();alignment,_=steering_yaw_alignment([x[0] for x in items[:2]])
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),
        fit_checkpoint_sha256=r['checkpoint_sha256'],exchange_predictions_sha256=er['predictions_sha256'],entries=entries))
    reports=[]
    with np.load(fit/'predictions.npz') as z,np.load(parent/'predictions.npz') as before,np.load(exchange/'predictions.npz') as branches:
        for i,key in enumerate(['train_0','train_1','dev']):
            mu=z[key+'_muscles'];assert np.array_equal(mu,before[key+'_muscles'])
            assert np.array_equal(z[key][:,1:],before[key][:,1:])
            # Explicit normalized projection algebra, rather than readout.forward.
            state=cp['readout'];x=((torch.tensor(mu)-state['training_reference'])/state['training_scale']).numpy()
            expected=np.tanh(x@state['projection.weight'][0].numpy()+float(state['projection.bias'][0]))
            np.testing.assert_allclose(expected,z[key][:,0],atol=3e-7,rtol=0)
            d=items[i][0];t=d['decision_time'];pre=np.zeros(len(t),bool)
            for onset in sustained_onsets(d,alignment):
                pre|=(t>=onset['time']-.8)&(t<=onset['time']-.08)&(abs(d['body'][:,4])<.07)
            pre&=d['fit_mask'];e=z[key][:,0]-d['labels'][:,0]
            values=dict(all_rmse=float(np.sqrt(np.mean(e[d['fit_mask']]**2))),pre_rmse=float(np.sqrt(np.mean(e[pre]**2))))
            for k,v in values.items():assert abs(v-r['comparisons'][i]['candidate'][k])<1e-10
            reports.append(values)
        np.testing.assert_allclose(branches['normal'],z['dev'],atol=2e-6,rtol=0)
        d=items[2][0];window_reports=[];errors={k:[] for k in ['normal','source_path','opposite_video']}
        for i,w in enumerate(er['windows']):
            prefix=f'window_{i}_';ids=branches[prefix+'indices'];m=branches[prefix+'pre_mask']
            normal=branches[prefix+'normal'];opposite=branches[prefix+'opposite_video']
            np.testing.assert_allclose(normal,branches['normal'][ids],atol=2e-6,rtol=0)
            signed=float(np.mean(normal[m,0]-opposite[m,0])*w['direction'])
            assert abs(signed-w['mean_signed_normal_minus_opposite'])<1e-8
            for case in errors:
                errors[case].extend((branches[prefix+case][m,0]-d['labels'][ids[m],0]).tolist())
            window_reports.append(dict(event=w['event'],mean_signed_normal_minus_opposite=signed,
                body_and_body_history_equal=w['body_and_body_history_equal'],initial_state_equal=w['initial_state_equal']))
    # Opposite-video error uses original labels only as mismatch diagnostic;
    # the counterfactual has no independently known correct action.
    rmse={k:float(np.sqrt(np.mean(np.square(v)))) for k,v in errors.items()}
    out=dict(status='completed_independent_bounded_steering_audit',metrics=reports,
        windows=window_reports,pre_rmse=rmse,pedal_rows_and_predictions_exact=True,
        original_adapters_and_neural_trajectories_exact=True,actor_qualified=False,
        source_path_intervention='Disable parent optical residual while retaining its DN adapter and new head',
        opposite_video_has_no_action_ground_truth=True,
        termination_reason='Checkpoint/prediction hashes, independent projection, metrics, TRAIN choice, protected states and stored branch joins verified')
    write_json(run/'result.json',out);print(json.dumps(out))


if __name__=='__main__':main()
