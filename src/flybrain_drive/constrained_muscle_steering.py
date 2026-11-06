"""TRAIN group protection for an engineering six-muscle steering readout."""
import numpy as np
from scipy.optimize import minimize


def fit_group_protected_steering(x,y,groups,initial_beta,initial_bias):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);beta=np.asarray(initial_beta,dtype=float)
    masks=[np.asarray(m,dtype=bool) for m in groups]
    if (x.ndim!=2 or x.shape[1]!=6 or y.shape!=(len(x),) or beta.shape!=(6,)
            or len(x)<7 or not masks or any(m.shape!=y.shape or not m.any() for m in masks)
            or not all(np.isfinite(v).all() for v in (x,y,beta))
            or not np.isfinite(initial_bias) or np.any(abs(y)>1)):
        raise ValueError('Finite six-muscle TRAIN rows and nonempty protected groups required')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    design=np.column_stack([(x-mean)/scale,np.ones(len(x))])
    initial=np.r_[beta*scale,initial_bias+mean@beta]
    target=np.array([np.mean((np.tanh(design[m]@initial)-y[m])**2) for m in masks])
    if np.any(target<1e-12):raise ValueError('Already-perfect group cannot define relative improvement')
    def ratios(p):
        error=np.tanh(design@p[:7])-y
        return np.array([np.mean(error[m]**2) for m in masks])/target
    def constraints(p):return 1.-p[7]-ratios(p)
    def constraint_jac(p):
        prediction=np.tanh(design@p[:7]);factor=2*(prediction-y)*(1-prediction**2)
        gradients=np.array([np.mean(factor[m,None]*design[m],0) for m in masks])/target[:,None]
        return np.column_stack([-gradients,-np.ones(len(masks))])
    def objective(p):return -p[7]+.0001*np.sum((p[:6]-initial[:6])**2)
    def objective_jac(p):return np.r_[.0002*(p[:6]-initial[:6]),0.,-1.]
    solved=minimize(objective,np.r_[initial,0.],jac=objective_jac,method='SLSQP',
        bounds=[(None,None)]*7+[(0.,1.)],
        constraints=[dict(type='ineq',fun=constraints,jac=constraint_jac)],
        options=dict(maxiter=300,ftol=1e-11))
    if not solved.success or np.min(constraints(solved.x)) < -1e-7:
        raise ValueError('Protected fit failed or violated TRAIN group bound: '+str(solved.message))
    fitted=solved.x
    return dict(beta=fitted[:6]/scale,bias=float(fitted[6]-mean@(fitted[:6]/scale)),
        minimum_training_mse_improvement=float(fitted[7]),group_mse_ratios=ratios(fitted).tolist(),
        optimizer_iterations=int(solved.nit),objective_evaluations=int(solved.nfev),
        optimizer_message=str(solved.message),mean=mean,scale=scale)
