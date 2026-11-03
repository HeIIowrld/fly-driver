"""Bounded incremental steering requests for stationary calibration only."""
import math


def steering_ramp(previous, target, *, maximum_step=.05):
    if (not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
                for x in (previous,target,maximum_step))
            or not -1 <= previous <= 1 or not -1 <= target <= 1
            or not 0 < maximum_step <= .05):
        raise ValueError('finite bounded steering ramp required')
    steps=math.ceil(abs(target-previous)/maximum_step)
    return [previous+(target-previous)*i/steps for i in range(1,steps+1)] if steps else []
