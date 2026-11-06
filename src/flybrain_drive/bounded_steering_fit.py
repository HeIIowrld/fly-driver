"""Offline engineering readout fit on six flight-muscle summaries only."""
import numpy as np
import torch


def fit_bounded_steering(x, y, weights, initial_beta, initial_bias, regularization=.001):
    """Minimize weighted command-space MSE; normalize using fit rows only.

    The intercept is unpenalized. No image/body bypass or recurrent reset is
    introduced. Return coefficients in the caller's original feature units.
    """
    x, y, weights = map(lambda v: np.asarray(v, dtype=float), (x, y, weights))
    beta = np.asarray(initial_beta, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x),)
            or weights.shape != y.shape or beta.shape != (6,) or len(x) < 7
            or not all(np.isfinite(v).all() for v in (x, y, weights, beta))
            or not np.isfinite(initial_bias) or not np.isfinite(regularization)
            or regularization <= 0 or np.any(weights <= 0) or np.any(abs(y) > 1)):
        raise ValueError('Finite six-muscle fit rows, bounded targets and positive weights required')
    mean = np.average(x, axis=0, weights=weights)
    scale = x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    z = torch.tensor((x - mean) / scale, dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    w = torch.tensor(weights / weights.sum(), dtype=torch.float64)
    parameters = torch.nn.Parameter(torch.tensor(
        np.r_[beta * scale, initial_bias + mean @ beta], dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameters], lr=1., max_iter=200,
                                tolerance_grad=1e-10, tolerance_change=1e-13,
                                line_search_fn='strong_wolfe')
    evaluations = 0

    def objective():
        prediction = torch.tanh(z @ parameters[:6] + parameters[6])
        return (w * (prediction - target).square()).sum() + regularization * parameters[:6].square().sum()

    initial_loss = float(objective().detach())

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        loss = objective()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite readout objective')
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    final_loss = float(objective().detach())
    if final_loss > initial_loss + 1e-10:
        raise ValueError('Bounded steering optimizer increased the fit objective')
    fitted = parameters.detach().numpy()
    output_beta = fitted[:6] / scale
    return dict(beta=output_beta, bias=float(fitted[6] - mean @ output_beta),
                initial_loss=initial_loss, final_loss=final_loss,
                objective_evaluations=evaluations, mean=mean, scale=scale)
