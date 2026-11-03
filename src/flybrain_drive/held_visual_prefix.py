"""Speculative held-input FlyVis states; commit only the actual elapsed prefix."""
import math


class HeldVisualPrefix:
    """One immutable visual input and starting state; never create camera frames."""
    def __init__(self, network, state, held, *, steps=16):
        if type(steps) is not int or not 1 <= steps <= 30:
            raise ValueError('Explicit bounded five-millisecond prefix required')
        self.network, self.start, self.held, self.steps = network, state, held, steps
        self.states = network.simulate(held[:,None,None,:].expand(-1,steps,-1,-1),
                                       dt=.005,initial_state=state,as_states=True)
        if len(self.states) != steps:
            raise ValueError('Each held-input state must be retained')
        self.used = False

    def commit(self, elapsed):
        if self.used:
            raise RuntimeError('Visual prefix already consumed')
        if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or not 0 < elapsed <= .15:
            raise ValueError('Original bounded elapsed time required')
        full = int(math.floor(elapsed/.005))
        count = min(full, self.steps)
        state = self.start if count == 0 else self.states[count-1]
        remaining = full-count
        if remaining:
            state = self.network.simulate(self.held[:,None,None,:].expand(-1,remaining,-1,-1),
                                          dt=.005,initial_state=state,as_states=True)[-1]
        remainder = elapsed-full*.005
        if remainder > 1e-12:
            state = self.network.simulate(self.held[:,None,None,:],dt=remainder,
                                          initial_state=state,as_states=True)[-1]
        self.used = True
        return state
