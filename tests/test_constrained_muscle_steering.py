import numpy as np
from flybrain_drive.constrained_muscle_steering import fit_group_protected_steering


def test_common_signal_improves_each_train_group():
    x=np.random.default_rng(1704).normal(size=(80,6));y=np.tanh(.2*x[:,0]-.1*x[:,1]+.05)
    groups=[np.arange(80)<40,np.arange(80)>=40,np.arange(80)%4==0]
    fitted=fit_group_protected_steering(x,y,groups,np.zeros(6),0.)
    prediction=np.tanh(x@fitted['beta']+fitted['bias'])
    assert fitted['minimum_training_mse_improvement']>.9
    for m in groups:assert np.mean((prediction[m]-y[m])**2)<.1*np.mean(y[m]**2)


def test_incompatible_groups_are_not_sacrificed_for_average_loss():
    x=np.zeros((40,6));y=np.r_[np.full(20,.2),np.full(20,-.2)]
    groups=[np.arange(40)<20,np.arange(40)>=20]
    fitted=fit_group_protected_steering(x,y,groups,np.zeros(6),0.)
    prediction=np.tanh(x@fitted['beta']+fitted['bias'])
    assert fitted['minimum_training_mse_improvement']<1e-6
    for m in groups:assert np.mean((prediction[m]-y[m])**2)<=.04000001
