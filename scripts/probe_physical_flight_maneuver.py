"""Named-circuit injections and whole original vision/body replay; no actuator access."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_visual_body_preview import direct_body_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    for path, digest in plan['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Declared input changed: '+path)
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), optimizer_updates=0, actuator_messages_sent=0,
        ui_actions_sent=0, no_actor_qualification=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'executed-physical-maneuver.py').write_bytes(
        Path('src/flybrain_drive/flight_maneuver.py').read_bytes())
    result = dict(status='failed', optimizer_updates=0, actuator_messages_sent=0,
                  actor_deployed=False, goal_achieved=False)
    try:
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required')
        payload = torch.load(plan['source_checkpoint'], map_location='cuda', weights_only=False)
        model = PhysicalManeuverCircuit(plan['source_spec'], payload).to('cuda').eval()
        original_matrix = model.backbone.circuit.matrix.coalesce().clone()
        original_body = {k: v.clone() for k, v in model.backbone.body.state_dict().items()}
        kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        body = torch.zeros(1, 21, device='cuda')
        body[:, 16] = 60/3.6
        valid = torch.zeros_like(body, dtype=torch.bool)
        valid[:, [0, 1, 4, 16]] = True
        neutral = torch.tensor([[0., 60/3.6]], device='cuda')
        goal = torch.tensor(racing_goal(100, 1), device='cuda')
        with torch.no_grad():
            steady = model.initial_state(1)
            for _ in range(100):
                _, muscles, steady = model.advance(neutral, body, valid, goal, steady, .01,
                                                   **kwargs)
            old_reference = {k: v.detach().cpu().tolist() for k, v in
                             model.backbone.readout.state_dict().items()}
            # Only a declared model-unit neutral offset is rebaselined. Original
            # gains/mix and protected body encoder remain unchanged; no plant fit.
            readout = model.backbone.readout
            old_turn_gain = readout.turn_log_gain.clone()
            old_forward_gain = readout.forward_log_gain.clone()
            readout.calibrate_reference(muscles[0], turn_scale=1., forward_scale=1.)
            readout.turn_log_gain.copy_(old_turn_gain)
            readout.forward_log_gain.copy_(old_forward_gain)
            new_reference = {k: v.detach().cpu().tolist() for k, v in readout.state_dict().items()}
            cases = ['neutral', 'left', 'right', 'accelerate', 'brake', 'left_accelerate',
                     'left_with_positive_yaw', 'left_with_negative_yaw', 'yaw_only',
                     'same_command_higher_ceiling']
            n = len(cases)
            commands = neutral.expand(n, -1).clone()
            commands[[1, 5, 6, 7], 0] = .3
            commands[2, 0] = -.3
            commands[[3, 5], 1], commands[4, 1] = 100/3.6, 20/3.6
            bodies, masks, goals = body.expand(n, -1).clone(), valid.expand(n, -1).clone(), (
                goal.expand(n, -1).clone())
            bodies[[6, 8], 4], bodies[7, 4] = .3, -.3
            goals[9, 1] = 2.
            injection = {}
            for lesion in (None, 'desired_yaw', 'speed_error', 'yaw_feedback'):
                state = steady.expand(n, -1).clone()
                history, muscle_history = [], []
                for _ in range(20):
                    intent, muscle, state = model.advance(commands, bodies, masks, goals,
                                                          state, .0407, lesion=lesion, **kwargs)
                    history.append(intent.cpu().numpy())
                    muscle_history.append(muscle.cpu().numpy())
                key = lesion or 'intact'
                injection[key] = np.asarray(history)
                np.savez_compressed(run/('injection-'+key+'.npz'), intent=injection[key],
                                    muscles=np.asarray(muscle_history))
            actual = injection['intact'][-1]
            signs = dict(left=bool(actual[1, 0] > actual[0, 0]+1e-5),
                right=bool(actual[2, 0] < actual[0, 0]-1e-5),
                acceleration=bool(actual[3, 1] > actual[0, 1]+1e-5),
                braking=bool(actual[4, 1] < actual[0, 1]-1e-5),
                positive_yaw_feedback=bool(actual[6, 0] < actual[1, 0]-1e-5),
                negative_yaw_feedback=bool(actual[7, 0] > actual[1, 0]+1e-5),
                ceiling_not_cruise=bool(np.array_equal(injection['intact'][:, 0],
                                                       injection['intact'][:, 9])))
            target_lesion = float(np.abs(injection['desired_yaw'][:, 1]-
                                        injection['desired_yaw'][:, 0]).max())
            power_lesion = float(np.abs(injection['speed_error'][:, 3]-
                                       injection['speed_error'][:, 0]).max())
            feedback_lesion = float(np.abs(injection['yaw_feedback'][:, 6]-
                                          injection['yaw_feedback'][:, 1]).max())
            # Cut all three externally driven neural populations in a separate
            # declared experiment. Every effect must disappear at the motor end.
            ports = torch.cat([model.backbone.body_ports, model.backbone.steering_ports,
                               model.backbone.goal_ports])
            model.backbone.circuit.state_mask[ports] = 0
            state = model.initial_state(n)
            for _ in range(100):
                _, cut_muscles, state = model.advance(commands, bodies, masks, goals,
                                                       state, .01, **kwargs)
            cut_activity = float(cut_muscles.abs().max())
            model.backbone.circuit.state_mask.fill_(1)
            result.update(completed_injection_cases=cases, partial_sign_tests=signs,
                partial_target_lesion_error=target_lesion, partial_power_lesion_error=power_lesion,
                partial_feedback_lesion_error=feedback_lesion, partial_cut_activity=cut_activity,
                old_motor_reference=old_reference, rebased_motor_reference=new_reference)
            dataset = read(Path(plan['registered_data'])/'dataset.json')
            entry = next(e for e in dataset['episodes'] if e['role'] == 'development')
            if sha256(entry['targets_file']) != entry['targets_sha256']:
                raise ValueError('Original replay clock reference changed')
            # No route/target values are read: this file supplies the original clock only.
            with load_array(entry['targets_file'], allow_pickle=False) as a:
                clock = a['observation_time'].copy()
            with load_array(plan['prediction_file'], allow_pickle=False) as a:
                predicted = a['adapted'].copy()
            bodies_np, masks_np, _ = direct_body_reference(entry, clock,
                sha256('configs/ac-virtual-navigation-body-v2.json'))
            commands_np = np.column_stack([-bodies_np[:, 16]*predicted[:, 5],
                                           np.full(len(clock), 60/3.6)]).astype(np.float32)
            if predicted.shape != (len(clock), 6) or np.any(np.diff(clock) <= 0):
                raise ValueError('Complete original visual predictions and ordered clock required')
            c = torch.tensor(commands_np, device='cuda', dtype=torch.float32)
            b = torch.tensor(bodies_np, device='cuda', dtype=torch.float32)
            m = torch.tensor(masks_np, device='cuda')
            g = torch.tensor(racing_goal(100, len(clock)), device='cuda')
            # Validate every command, including the held final command, without clipping.
            model.drive(c, b, m, g, **kwargs)
            replays = []
            for lesion in (None, 'desired_yaw', 'yaw_feedback', 'speed_error'):
                state = model.initial_state(1)
                intent, muscle = model.backbone._read_state_tensor(state)
                intents, muscles = [intent[0].cpu().numpy()], [muscle[0].cpu().numpy()]
                for i in range(1, len(clock)):
                    intent, muscle, state = model.advance(c[i-1:i], b[i-1:i], m[i-1:i],
                        g[i-1:i], state, float(clock[i]-clock[i-1]), lesion=lesion, **kwargs)
                    intents.append(intent[0].cpu().numpy())
                    muscles.append(muscle[0].cpu().numpy())
                intents = np.asarray(intents)
                controls = np.asarray([canonical_controls(row) for row in intents])
                key = lesion or 'intact'
                np.savez_compressed(run/('whole-replay-'+key+'.npz'), intent=intents,
                    muscles=np.asarray(muscles), canonical_requests=controls,
                    observation_time=clock, physical_commands=commands_np,
                    body_valid=masks_np)
                replays.append(dict(lesion=key, observations=len(clock), intervals=len(clock)-1,
                    original_elapsed_seconds=float(clock[-1]-clock[0]),
                    numeric_resets_inside_episode=0,
                    max_absolute_request=abs(controls).max(0).tolist(),
                    intent_std=intents.std(0).tolist(), commanded_vehicle=False))
        matrix = model.backbone.circuit.matrix.coalesce()
        if (not torch.equal(matrix.indices(), original_matrix.indices())
                or not torch.equal(matrix.values(), original_matrix.values())
                or any(not torch.equal(v, original_body[k])
                       for k, v in model.backbone.body.state_dict().items())):
            raise ValueError('Measured matrix or protected body encoder changed')
        result.update(status='completed_physical_maneuver_component_probe',
            sign_tests=signs, cases=cases, final_injected_intents=actual.tolist(),
            desired_yaw_lesion_pair_error=target_lesion, speed_error_lesion_pair_error=power_lesion,
            yaw_feedback_lesion_pair_error=feedback_lesion, cut_input_motor_activity=cut_activity,
            old_motor_reference=old_reference, rebased_motor_reference=new_reference,
            reference_scope='Synthetic neutral; original gains/mix retained; no physical plant fit',
            replays=replays, graph_and_body_unchanged=True,
            physical_vehicle_plant_tested=False, physiological_function_verified=False,
            vision_prediction_qualified=False, racing_pace_policy_implemented=False,
            all_component_checks_passed=bool(all(signs.values()) and max(
                target_lesion, power_lesion, feedback_lesion, cut_activity) < 1e-7))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
