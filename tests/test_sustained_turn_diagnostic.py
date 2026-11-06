import numpy as np
import pytest
from flybrain_drive.sustained_turn_diagnostic import steering_yaw_alignment,paired_sustained_onsets,sustained_onsets


def test_unpaired_turns_remain_eligible_for_precorner_supervision():
    n=120;t=np.arange(n)*.04;b=np.zeros((n,21));b[:,16]=25
    b[50:70,4]=.3;y=np.zeros((n,3));y[50:70,0]=-.2
    d=dict(fit_mask=np.ones(n,dtype=bool),decision_time=t,body=b,labels=y)
    assert [w['event'] for w in sustained_onsets(d,-1)]==[50]
    assert paired_sustained_onsets(d,-1)==[]


def test_sustained_direction_uses_body_turn_not_transient_opposite_steering():
    n=300;t=np.arange(n)*.04;b=np.zeros((n,21));b[:,16]=25
    b[50:70,4]=.3;b[120:126,4]=.3;b[225:245,4]=-.3
    y=np.zeros((n,3));y[50:56,0]=.1;y[56:70,0]=-.2;y[225:245,0]=.2
    d=dict(fit_mask=np.ones(n,dtype=bool),decision_time=t,body=b,labels=y)
    windows=paired_sustained_onsets(d,-1)
    assert [w['event'] for w in windows]==[50,225]
    assert [w['direction'] for w in windows]==[-1,1]
    assert all(w['event']!=120 for w in windows)


def test_empirical_sign_requires_enough_training_evidence_and_agreement():
    b=np.zeros((200,21));b[:,4]=.3;y=np.zeros((200,3));y[:,0]=-.2
    d=dict(body=b,labels=y,fit_mask=np.ones(200,dtype=bool))
    sign,evidence=steering_yaw_alignment([d])
    assert sign==-1 and evidence['agreement']==1 and evidence['count']==200
    y[::2,0]=.2
    with pytest.raises(ValueError,match='Ambiguous'):steering_yaw_alignment([d])
    d['fit_mask'][:]=False
    with pytest.raises(ValueError,match='Insufficient'):steering_yaw_alignment([d])
