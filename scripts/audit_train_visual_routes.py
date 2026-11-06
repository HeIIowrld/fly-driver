"""Verify TRAIN component clamps, same prefix/body arrays and output formulas."""
import argparse,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.functional_motion_overlay import load_functional_motion_overlay
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--routes',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.routes);r=read(root/'result.json');m=read(root/'manifest.json')['config'];assert r['status']=='completed_train_visual_component_clamps' and sha256(root/'responses.npz')==r['responses_sha256'] and sha256(root/'groups.json')==r['groups_sha256'] and sha256(root/'executed-entrypoint.py')==m['program_sha256'];components,motion,receipt=load_readonly_motion_candidate(read(m['plan']['candidate_plan']));receipt=load_functional_motion_overlay(m['arguments']['experimental_checkpoint'],components,motion,receipt);assert receipt==m['candidate'];state={k:v.numpy().astype(float) for k,v in components[1].state_dict().items()};groups=read(root/'groups.json');counts={k:0 for k in r['hypothesized_positive_bend_pairs']};clamps={'hold_motion':'motion_delta','hold_dn':'effective_dn','hold_optical':'effective_optical','hold_pooled':'effective_pooled','hold_temporal':'effective_temporal'};maximum=0.;verified=0
    with np.load(root/'responses.npz') as z:arrays={k:z[k].copy() for k in z.files}
    for g in groups:
        j=g['group'];assert g['temperature_c']==80.;body=arrays[f'group_{j}_body21'];mask=arrays[f'group_{j}_body21_valid'];native=arrays[f'group_{j}_native19'];valid=arrays[f'group_{j}_native19_valid'];assert hashlib.sha256(body.tobytes()+mask.tobytes()+native.tobytes()+valid.tobytes()).hexdigest()==g['fixed_body_sha256']
        for arm in counts:
            prediction=arrays[f'group_{j}_{arm}_prediction'];mu=arrays[f'group_{j}_{arm}_muscles'];logits=((mu-state['training_reference'])/state['training_scale'])@state['projection.weight'].T+state['projection.bias'];expected=np.concatenate([np.tanh(logits[...,:1]),1/(1+np.exp(-logits[...,1:]))],axis=-1);maximum=max(maximum,float(abs(expected-prediction).max()));assert maximum<=2e-6;late=prediction[:,m['plan']['evaluation_first_frame']:];margin=float(m['plan']['expected_canonical_steering_sign_per_left_bend']*np.mean(late[1,:,0]-late[0,:,0])/2);assert margin==g['branches'][arm]['hypothesized_bend_margin'];counts[arm]+=int(margin>m['plan']['bend_ordering_minimum_margin']);np.testing.assert_array_equal(prediction[:,:11],arrays[f'group_{j}_fine_prediction'][:,:11])
        for arm,field in clamps.items():
            value=arrays[f'group_{j}_{arm}_{field}'];np.testing.assert_array_equal(value[:,11:],np.broadcast_to(value[0,10],value[:,11:].shape));verified+=1
    assert counts==r['hypothesized_positive_bend_pairs'];reference=Path(m['arguments']['reference_response']);rr=read(reference/'result.json');assert sha256(reference/'result.json')==r['reference_result_sha256'] and sha256(reference/'responses.npz')==rr['responses_sha256']
    with np.load(reference/'responses.npz') as z:error=max(float(abs(arrays[f'group_{g}_fine_prediction']-z[f'group_{3*g+1}_fine_prediction']).max()) for g in range(len(groups)))
    assert error==r['normal_reference_max_difference']<=2e-6
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),routes_result_sha256=sha256(root/'result.json'),vehicle_control_sent=False,optimizer_updates=0));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_train_visual_route_audit',groups_verified=len(groups),clamped_component_trajectories_verified=verified,hypothesized_positive_bend_pairs=counts,independent_muscle_readout_max_difference=maximum,normal_reference_max_difference=error,body_and_prefix_arrays_verified=True,full_neural_branch_replay_by_audit=False,actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0))


if __name__=='__main__':main()
