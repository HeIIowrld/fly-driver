"""Actual checkpoint action reachability on sealed sensory replay; no AC actuation."""

import argparse
import copy

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import (
    condition_independent_pedals,
    configure_trainable_groups,
    episode_chunk,
    load_episodes,
    replay_chunk,
)
from flybrain_drive.vehicle_circuit import positive_gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--decisions', type=int, default=64)
    parser.add_argument('--pedal-mode', choices=['independent', 'brake_attenuated'])
    parser.add_argument('--condition-pedals', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.decisions <= 256:
        raise ValueError('Finite recurrent replay budget required')
    torch.set_num_threads(1)
    run = begin_run(args.output, dict(arguments=vars(args),
        checkpoint_sha256=sha256(args.checkpoint), scope=__doc__))
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        graph = {k: np.asarray(v) for k, v in checkpoint['graph'].items()}
        constructor = copy.deepcopy(checkpoint['constructor'])
        constructor['sparse_backend'] = 'cached_csr_cpu'
        if args.pedal_mode is not None:
            constructor['pedal_mode'] = args.pedal_mode
        model = MultiViewCircuitController(graph, **constructor)
        model.load_state_dict(checkpoint['model'])
        configure_trainable_groups(model, ['visual', 'organs', 'readout'])
        episodes, manifest = load_episodes(args.teacher, model.sensory_calibration)
        length = min(args.decisions, len(episodes[0]['action']))
        obs, _, _, active = episode_chunk([episodes[0]], 0, length)
        activity = []
        hook = model.readout.register_forward_pre_hook(
            lambda module, inputs: activity.append(inputs[0].detach().clone()))
        try:
            prediction, _ = replay_chunk(model, obs, active, model.initial_state(1),
                                         manifest['brain_ticks_per_decision'])
        finally:
            hook.remove()
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        gradients = {}
        for index, action in enumerate(('steering', 'throttle', 'brake')):
            values = torch.autograd.grad(prediction[..., index].mean(),
                                        [p for _, p in named], retain_graph=index < 2,
                                        allow_unused=True)
            groups = dict(visual=0., organs=0., readout=0.)
            for (name, _), gradient in zip(named, values, strict=True):
                group = ('visual' if name.startswith('adapter.visual_adapter') else
                         'organs' if name.startswith('adapter.organ_adapters') else 'readout')
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise ValueError('Nonfinite action sensitivity')
                    groups[group] = max(groups[group], float(gradient.abs().max()))
            gradients[action] = groups
        gains = positive_gain(model.readout.pedal_log_gain,
                              model.readout.gain_parameterization).detach()
        conditioning = after_min = after_max = after_rms = None
        if args.condition_pedals:
            final_activity = torch.cat(activity[manifest['brain_ticks_per_decision'] - 1::
                                               manifest['brain_ticks_per_decision']])
            conditioning = condition_independent_pedals(model.readout, final_activity,
                target_rms=[.15, .15], max_gain=1e8)
            with torch.no_grad():
                after, _ = replay_chunk(model, obs, active, model.initial_state(1),
                                         manifest['brain_ticks_per_decision'])
            after_min = after.amin((0, 1)).tolist()
            after_max = after.amax((0, 1)).tolist()
            after_rms = after.square().mean((0, 1)).sqrt().tolist()
            if not torch.equal(after[..., 0], prediction.detach()[..., 0]):
                raise ValueError('Pedal-only conditioning unexpectedly changed steering')
        write_json(run / 'result.json', dict(
            status='completed', checkpoint_sha256=sha256(args.checkpoint),
            replay_episode=episodes[0]['source']['episode_id'], replay_decisions=length,
            scope='MuJoCo sensory replay; derivatives only, not pedal labels or native training',
            action_profile=checkpoint.get('action_profile'),
            output_min=prediction.detach().amin((0, 1)).tolist(),
            output_max=prediction.detach().amax((0, 1)).tolist(),
            mean_action_parameter_gradient_max_abs=gradients,
            pedal_positive_gains=gains.tolist(),
            pedal_mode=model.readout.pedal_mode,
            pedal_conditioning=conditioning,
            after_conditioning_min=after_min, after_conditioning_max=after_max,
            after_conditioning_rms=after_rms,
            source_checkpoint_unchanged=True,
            independent_pedal_upper_bounds=(-torch.expm1(-gains)).tolist(),
            bound_assumption='every corresponding output neuron at maximum activity 1; throttle '
                             'bound additionally assumes zero brake in legacy mode',
            optimizer_updates=0, structural_weights_changed=False,
            native_AC_training_verified=False, native_pedals_trained=False,
            actuator_messages_sent=0, goal_achieved=False))
    except Exception as error:
        write_json(run / 'result.json', dict(status='failed', error=type(error).__name__,
                                            message=str(error), goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
