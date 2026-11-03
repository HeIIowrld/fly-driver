"""Fundamental sinusoid fit for explicitly synthetic model-time response experiments."""

import numpy as np


def fundamental(times, values, frequency):
    times, values = np.asarray(times), np.asarray(values)
    if (times.ndim != 1 or values.shape != times.shape or len(times) < 10
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or not np.all(np.diff(times) > 0) or not np.isfinite(frequency) or frequency <= 0):
        raise ValueError('Ordered finite model-time samples and positive frequency required')
    design = np.column_stack([np.sin(2*np.pi*frequency*times),
                              np.cos(2*np.pi*frequency*times), np.ones(len(times))])
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    error = values-design@coefficients
    variance = np.sum((values-values.mean())**2)
    phasor = complex(coefficients[0], coefficients[1])
    return dict(amplitude=float(abs(phasor)), phase_degrees=float(np.angle(phasor, deg=True)),
                offset=float(coefficients[2]), residual_rms=float(np.sqrt(np.mean(error**2))),
                r_squared=float(1-np.sum(error**2)/variance) if variance > 1e-24 else None)
