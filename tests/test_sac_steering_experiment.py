import numpy as np
import torch
from types import SimpleNamespace
from scripts.sac_steering_experiment import onsets,paired_onsets,snapshot,restore,DYNAMIC


def episode():
    t=np.arange(180)*.04
    yaw=np.zeros(180)
    yaw[50:65]=.25
    yaw[125:140]=-.25
    label=np.zeros((180,3));label[50:65,0]=-.2;label[125:140,0]=.2
    body=np.zeros((180,21));body[:,4]=yaw;body[:,16]=25
    return dict(fit_mask=np.ones(180,dtype=bool),decision_time=t,body=body,labels=label)


def test_transition_windows_are_before_body_turn_and_donor_is_opposite():
    d=episode();events=paired_onsets(d)
    assert len(events)==2
    for w in events:
        assert w['start']<w['event']
        assert d['decision_time'][w['start']]>=w['time']-1.2-1e-9
        assert w['direction']*np.sign(d['labels'][w['donor_event'],0])==-1
        assert abs(d['body'][w['event']-1,4])==0


def test_unqualified_or_missing_opposite_windows_are_not_invented():
    d=episode();d['fit_mask'][120:]=False
    assert len(onsets(d))==1
    assert paired_onsets(d)==[]
    d['fit_mask'][:]=False
    assert onsets(d)==[]


def test_restore_is_repeatable_and_has_no_alias_to_snapshot():
    model=SimpleNamespace(**{key:None for key in DYNAMIC})
    model.state=torch.tensor([[1.,2.]])
    model.traces=(torch.tensor([[3.]]),torch.tensor([[4.]]),torch.tensor([[5.]]))
    model.previous_time=1.2
    saved=snapshot(model);model.state.add_(7);model.traces[0].add_(9)
    restore(model,saved)
    assert torch.equal(model.state,torch.tensor([[1.,2.]]))
    assert model.traces[0].item()==3
    model.state.zero_();restore(model,saved)
    assert model.state.sum()==3
    assert model.previous_time==1.2
