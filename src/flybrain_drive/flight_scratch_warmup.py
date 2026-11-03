"""Disposable worst-interval operator warmup; actual numerical episode remains untouched."""

import time

import torch


@torch.no_grad()
def prime_flight_operators(stream):
    if (stream.steps != 0 or stream.order.clock is not None or stream.failed
            or stream.flight_held is not None or stream.retina_held is not None):
        raise ValueError('Scratch warmup is only allowed before the first actual observation')
    started = time.perf_counter()
    visual_before = stream.visual_state.nodes.activity.clone()
    flight_before = stream.flight_state.clone()
    durations = []
    with torch.device(stream.device):
        from flyvis.utils.tensor_utils import AutoDeref

        retinal = torch.full((4, 721), .5)
        # Independent clone of the canonical grey initial state; never replace the actual state.
        initial = stream.visual_state
        visual = stream.network._state_api(AutoDeref(
            nodes=AutoDeref(**{key: value.clone() for key, value in initial.nodes.items()}),
            edges=AutoDeref(**{key: value.clone() for key, value in initial.edges.items()})))
        body = torch.zeros((1, 16))
        valid = torch.zeros((1, 16), dtype=torch.bool)
        valid[:, [0, 1, 4, *range(8, 16)]] = True
        flight = stream.flight.initial_state(1)
        cases = [(.005, 8), (.00317, 1), (.005, 16), (.00123, 1), (.005, 8)]
        for dt, count in cases:
            step_started = time.perf_counter()
            held = retinal[:, None, None].expand(-1, count, -1, -1)
            visual = stream.network.simulate(held, dt=dt,
                initial_state=visual, as_states=True)[-1]
            scene = torch.cat([retinal[:, None],
                               visual.nodes.activity[:, stream.motion_indices]], dim=1)[None]
            features = stream.flight.scene_pool(scene)
            drive = stream.flight._drive_tensor(stream.visual_zero, stream.visual_valid,
                                                body, valid, stream.goals, features)
            for flight_dt in (.01, .01, .00317):
                flight = stream.flight._advance_elapsed_tensor(drive, flight, flight_dt)
            stream.flight._read_state_tensor(flight)[0].cpu()
            durations.append(dict(visual_dt=dt, visual_steps=count,
                                  elapsed_seconds=time.perf_counter()-step_started))
    assert torch.equal(stream.visual_state.nodes.activity, visual_before)
    assert torch.equal(stream.flight_state, flight_before)
    assert stream.steps == stream.visual_ticks == stream.flight_ticks == 0
    assert stream.order.clock is None and stream.flight_held is stream.retina_held is None
    return dict(seconds=time.perf_counter()-started, scratch_cases=durations,
                actual_neural_state_unchanged=True, observations_created=0,
                actual_prefix_consumed=0, actual_counters=0, default_device_restored=True)
