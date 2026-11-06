"""Summarize sealed rotation responses and fixed visual-intent operating point."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1)
    root=Path('runs/visual-body-rotation-20261104')
    plan=read(root/'stimulus-plan-01.json'); audit=read(root/'audit-01/result.json')
    assert audit['status']=='completed_independent_visual_body_rotation_audit'
    rows=[]; bound={}
    for name in ['parent','supervised','functional_priority']:
        run=root/(name+'-responses-01'); result=read(run/'result.json'); groups=read(run/'groups.json')
        assert result['responses_sha256']==sha(run/'responses.npz') and result['groups_sha256']==sha(run/'groups.json')
        arms={}
        with np.load(run/'responses.npz') as z:
            for arm in ['normal','context_off','context_reverse','context_delayed120']:
                margins=[g['branches'][arm]['hypothesized_bend_margin'] for g in groups]
                arms[arm]=dict(direction_positive=sum(g['branches'][arm]['direction_positive'] for g in groups),target_margin_002=sum(v>=.002 for v in margins),margin_range=[min(margins),max(margins)],maximum_action_change=np.max([g['branches'][arm]['post_prefix_max_abs_action_change_vs_normal'] for g in groups],axis=0).tolist(),by_yaw={str(y):sum(g['branches'][arm]['direction_positive'] for g in groups if g['yaw_rate_target_rad_s']==y) for y in [-.3,0.,.3]})
            pure_image_delta=max(float(np.max(abs(z[f"group_{g['group']}_normal_prediction"][1,:,0]-z[f"group_{g['group']}_normal_prediction"][0,:,0]))) for g in groups)
        rows.append(dict(model=name,groups=len(groups),arms=arms,same_body_different_image_max_steering_difference=pure_image_delta,zero_rate_intervention_difference=result['zero_rate_intervention_prediction_max_difference']))
        for f in ['result.json','groups.json','responses.npz','manifest.json']: bound[str(run/f)]=sha(run/f)
    cp=Path('runs/functional-priority-learning-20261104/supervised-fit-01/temporal-gain.pt')
    state=torch.load(cp,map_location='cpu',weights_only=True)['adapter']
    with np.load(root/'stimuli-01/features.npz') as z: static=torch.tensor(z['static'])
    value=((static-state['reference'])/state['scale']).clamp(-5,5)
    raw=(value@state['projection.weight'].T)[...,0]; contrast=8*raw.tanh()
    derivative=8*(1-raw.tanh().square())
    currents=.2*torch.sigmoid(state['tonic_logit']+contrast[...,None]*state['orientations'])
    bend_delta=abs(currents[1::2]-currents[::2]).amax(dim=(0,1))
    intent=dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),raw_projection_range=[float(raw.min()),float(raw.max())],contrast_range=[float(contrast.min()),float(contrast.max())],contrast_derivative_range=[float(derivative.min()),float(derivative.max())],derivative_fraction_of_center_range=[float(derivative.min()/8),float(derivative.max()/8)],current_range_per_port=[[float(currents[...,j].min()),float(currents[...,j].max())] for j in range(4)],max_paired_bend_current_delta_per_port=bend_delta.tolist(),all_contrasts_negative=bool((contrast<0).all()),interpretation='One-sided near-tanh-limit operation on this synthetic stimulus only; reduced contrast slope versus zero. Not proof of sole cause, training distribution saturation, biological tuning or improved control.')
    receipt=read('runs/motion-live-integration-20261104/source-integration-01.json')
    preserved={**receipt['sources'],**receipt['preserved_checkpoints']}
    prior=read('runs/flight-transfer-sensitivity-20261104/source-receipt-final-01.json')['preserved_prior_research_checkpoint_sha256']
    preserved.update(prior)
    preserved.update({
      'runs/temporal-port-recalibration-20261104/fit-01/temporal-gain.pt':'29a5cbd08ed1b442ab12f5d06e0ad33b0f9c6ee5fee4086eccae83909f195073',
      str(cp):'5fce4823399efa2c05e9d9c707f2bc8f0c2d6f28241431a231d50a7228e2f5c0',
      'runs/functional-priority-learning-20261104/functional_priority-fit-01/temporal-gain.pt':'8a2319eab4513edaa756e8309ce52d1b5235240305e4e697d0a57aa5fe0e30d5'})
    assert all(sha(p)==h for p,h in preserved.items())
    for p in ['stimulus-plan-01.json','stimuli-01/result.json','stimuli-01/features.npz','audit-01/result.json','tests-02.txt']:bound[str(root/p)]=sha(root/p)
    result=dict(status='completed_visual_body_rotation_summary',models=rows,visual_intent_operating_point=intent,audit=audit,source_bound_sha256=bound,preserved_source_and_checkpoint_sha256=preserved,optimizer_updates=0,actual_driving_improvement_verified=False,actor_qualified=False,vehicle_control_sent=False,goal_status='active',program_sha256=sha(__file__),original_observed_terminal_handles={'stimuli':79513,'parent':43533,'supervised':58058,'functional_priority':28831,'independent_audit':61606},original_observed_exit_codes=[0,0,0,0,0])
    (root/'summary-01.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(models=rows,intent=intent,preserved_files=len(preserved)),ensure_ascii=False))
if __name__=='__main__':main()
