"""Independent stored-port topology, adjoint/finite difference and muscle audit."""
import argparse,collections,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_controlled_motion_function import digest


def distances(matrix,targets,nonzero):
    c=matrix.coalesce();indices=c.indices().numpy();values=c.values().numpy();reverse=[[] for _ in range(c.shape[0])]
    for post,pre,value in zip(indices[0],indices[1],values):
        if not nonzero or value!=0:reverse[int(post)].append(int(pre))
    d=np.full(c.shape[0],-1,dtype=np.int32);q=collections.deque()
    for v in targets:d[v]=0;q.append(int(v))
    while q:
        post=q.popleft()
        for pre in reverse[post]:
            if d[pre]<0:d[pre]=d[post]+1;q.append(pre)
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.probe);r=read(root/'result.json');m=read(root/'manifest.json')['config'];source=Path(m['arguments']['stimuli']);sm=read(source/'manifest.json')['config'];components,motion,parent=load_readonly_motion_candidate(read(sm['plan']['candidate_plan']));model,head=components[:2];assert parent==m['candidate'] and r['status']=='completed_train_flight_port_transfer_probe' and sha256(root/'transfer.npz')==r['transfer_sha256'] and sha256(root/'cases.json')==r['cases_sha256'];reports=read(root/'cases.json');b=model.motor.backbone;matrix=b.circuit.matrix;names=['b1_L','b1_R','b2_L','b2_R','power_L','power_R'];source_ports=np.array(m['port_indices']);topology={};maximum_readout_error=0.;fd=[];gain=[];cancellation=[]
    for support in ('stored_index_support','nonzero_signed_weight_support'):
        topology[support]={name:distances(matrix,b.circuit.output_ports[getattr(b.readout,name)].numpy(),support.startswith('nonzero'))[source_ports].tolist() for name in names}
    with np.load(root/'transfer.npz') as z:
        for case in reports:
            k=case['case'];base=z[f'case_{k}_baseline'];jac=z[f'case_{k}_jacobian'];mu=base[:,:6];logits=((mu-head.training_reference.numpy())/head.training_scale.numpy())@head.projection.weight.detach().numpy().T+head.projection.bias.detach().numpy();expected=np.concatenate([np.tanh(logits[:,:1]),1/(1+np.exp(-logits[:,1:]))],axis=1);maximum_readout_error=max(maximum_readout_error,float(abs(expected-base[:,6:]).max()));assert maximum_readout_error<2e-6
            assert np.isfinite(jac).all()
            for section,bounds in m['port_sections'].items():
                start,end=bounds;direction=z[f'case_{k}_{section}_direction'];numeric=z[f'case_{k}_{section}_finite_difference'];analytic=z[f'case_{k}_{section}_adjoint_direction'];np.testing.assert_allclose(jac@direction,analytic,atol=1e-9,rtol=1e-6);error=float(abs(numeric-analytic).max());fd.append(dict(case=k,section=section,maximum_absolute_error=error,relative_l2_error=float(np.linalg.norm(numeric-analytic)/max(np.linalg.norm(numeric),np.linalg.norm(analytic),1e-12))));gain.append(dict(case=k,section=section,muscle_norm=float(np.linalg.norm(jac[:6,start:end])),steer_norm=float(np.linalg.norm(jac[6,start:end])),pedal_norm=float(np.linalg.norm(jac[7:,start:end])),muscle_silent_columns=int((np.linalg.norm(jac[:6,start:end],axis=0)<1e-10).sum()),columns=end-start,drive_near_boundary_fraction=float(((z[f'case_{k}_port_drive'][:,start:end]<.01)|(z[f'case_{k}_port_drive'][:,start:end]>.99)).mean())))
        late=sm['plan']['evaluation_first_frame']-sm['plan']['branch_frame']
        for pair in range(6):
            delta=(z[f'case_{2*pair+1}_baseline'][late:,:6]-z[f'case_{2*pair}_baseline'][late:,:6]).mean(0);expected=delta*(head.projection.weight[0]/head.training_scale).detach().numpy();actual=z[f'pair_{pair}_steering_logit_muscle_contributions'];np.testing.assert_allclose(expected,actual,atol=1e-9,rtol=1e-6);den=float(abs(actual).sum());cancellation.append(dict(pair=pair,cancellation_fraction=1-abs(float(actual.sum()))/max(den,1e-15),summed_absolute_contributions=den,net_logit_difference=float(actual.sum()),contributions=actual.tolist()))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),probe_result_sha256=sha256(root/'result.json'),parent_candidate=parent,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_flight_port_transfer_audit',fixed_parent_model_sha256=digest(model.state_dict()),case_count=len(reports),port_indices=source_ports.tolist(),port_sections=m['port_sections'],muscle_path_distances=topology,gains=gain,finite_difference_checks=fd,finite_difference_max_absolute_error=max(v['maximum_absolute_error'] for v in fd),muscle_readout_max_absolute_error=maximum_readout_error,muscle_cancellation=cancellation,full_neural_replay_by_audit=False,physiological_correctness_verified=False,actual_driving_improvement_verified=False,actor_qualified=False,vehicle_control_sent=False,termination_reason='Independent signed-matrix support distances, stored Jacobian directional products, finite-difference errors and muscle-to-action/logit contribution formulas. Does not repeat full neural paths or infer anatomical disconnection from zero signed weights.'))

if __name__=='__main__':main()
