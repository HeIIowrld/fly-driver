"""Common-descent direction for a small number of diagnostic task gradients."""
import numpy as np
from scipy.optimize import minimize


def minimum_norm_common_gradient(gradients, *, tolerance=1e-7):
    """Return simplex minimum-norm gradient and diagnostics, never a vehicle actor.

    A nonzero optimum should have nonnegative dot product with each task gradient;
    verify the numerical result explicitly. A near-zero norm signals no useful
    shared first-order direction, not a license to ignore a task.
    """
    g=np.asarray(gradients,dtype=np.float64)
    if g.ndim!=2 or g.shape[0]<2 or g.shape[1]<1 or not np.isfinite(g).all() or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Finite task-by-parameter gradients required')
    gram=g@g.T;scale=float(np.max(np.diag(gram)))
    if scale==0:return np.zeros(g.shape[1]),np.full(g.shape[0],1/g.shape[0]),{'status':'zero_gradient','task_dots':np.zeros(g.shape[0])}
    normalized=gram/scale;n=g.shape[0];solution=minimize(lambda w:.5*w@normalized@w,np.full(n,1/n),jac=lambda w:normalized@w,bounds=[(0.,1.)]*n,constraints={'type':'eq','fun':lambda w:np.sum(w)-1,'jac':lambda w:np.ones(n)},method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    if not solution.success or abs(solution.x.sum()-1)>1e-8 or np.any(solution.x < -1e-10):raise ValueError('Simplex optimization did not converge')
    direction=solution.x@g;dots=g@direction;norm2=float(direction@direction)
    if np.min(dots-norm2)<-tolerance*scale:raise ValueError('Common-descent numerical condition not met')
    return direction,solution.x,{'status':'stationary' if norm2<=tolerance**2*scale else 'common_descent','task_dots':dots}
