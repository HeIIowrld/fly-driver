"""Finite low-speed input/vehicle response measurement; no policy or desktop automation."""

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, NativePhysicsReader
from flybrain_drive.ac_native_position import NativeDiagnosticPositionReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.actuator_codec import CODEC_CONTRACT, project_pedals
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.calibrate_ac_sdk_steering import validate_stationary_calibration_state
from scripts.record_ac_wgc import physics_record

PLANT_PROGRAM = [
    ['baseline', 0., 0., 1., .2],
    ['launch', 0., .15, 0., 1.2],
    ['steering_positive', .05, .04, 0., .4],
    ['center', 0., 0., 0., .2],
    ['steering_negative', -.05, .04, 0., .4],
    ['coast', 0., 0., 0., .2],
    ['braking', 0., 0., 1., .6],
]
NEURAL_PROGRAM = [
    ['neural_baseline', 0., 0., .2],
    ['neural_launch', 0., 1., 1.6],
    ['neural_left', .15, 1., .5],
    ['neural_right', -.15, 1., .5],
    ['neural_decelerate', 0., 0., .8],
]
PEDAL_PROGRAM = [
    ['brake_'+str(v), 0., 0., v, .2]
    for v in (1., .5, .1, .05, .03, .025, .02, .015, .01, .005, 0., .03, .1, 1.)
] + [['gas_'+str(v), 0., v, 1., .2] for v in (.005, .01, .02, .03, .05, .1, .15)]
ACTUATOR_PROGRAM = [
    ['raw_brake_'+str(v), 0., 0., v, 1.5]
    for v in (.015, .016, .017, .018, .019, .0195, .02, .0205, .021, .025)
] + [
    ['raw_gas_'+str(v), 0., v, 1., 1.5]
    for v in (.015, .016, .017, .018, .019, .0195, .02, .0205, .021, .025)
] + [
    ['encoded_brake_'+str(v), 0., 0., v, 1.5]
    for v in (0., .005, .01, .011, .0165, .019, .02, .025, .05)
] + [
    ['encoded_gas_'+str(v), 0., v, 1., 1.5]
    for v in (.005, .01, .011, .0165, .02, .025)
]


def native_values(sample):
    values, offset, result = PHYSICS_PREFIX.unpack(sample.prefix_bytes), 0, {}
    for name, _, count in PHYSICS_LAYOUT:
        result[name] = values[offset] if count == 1 else values[offset:offset+count]
        offset += count
    return result


def validate_response_envelope(speed_kmh, displacement_m, damage_delta, outside, rpm):
    if (not np.isfinite([speed_kmh, displacement_m, damage_delta, rpm]).all()
            or abs(speed_kmh) >= 12 or not 0 <= displacement_m < 8
            or damage_delta > .01 or outside != 0 or rpm > 6500):
        raise ValueError('Finite low-speed plant measurement envelope exceeded')


def single_gear_transition(current, required):
    """Native gear1 is neutral and2 first; one reviewed button, never repeat."""
    if current not in (1, 2) or required not in (1, 2):
        raise ValueError('Only neutral/first single transitions are reviewed')
    if current == required:
        return None
    return ('down', 1) if required == 1 else ('up', 0)


EMBODIMENT_V2_MODE = 'native_neural_embodiment_v2_component'
EMBODIMENT_V2_INPUT_SCHEMA = 'flybrain.flight.embodiment_v2_component_inputs.v1'
EMBODIMENT_V2_CANONICAL_SCHEMA = 'flybrain.flight.reachable_power_brake_canonical.v2'
EMBODIMENT_V2_REQUIRED_SOURCES = (
    'configs/flight-mn-embodiment-component-proof-contract-20261030.json',
    'src/flybrain_drive/flight_embodiment_trial.py',
    'src/flybrain_drive/flight_embodiment.py',
    'src/flybrain_drive/flight_maneuver_trial.py',
    'src/flybrain_drive/flight_maneuver.py',
    'src/flybrain_drive/flight_circuit.py',
    'src/flybrain_drive/flight_commands.py',
    'src/flybrain_drive/vehicle_circuit.py',
    'src/flybrain_drive/flight_actuator.py',
    'src/flybrain_drive/actuator_codec.py',
)


def validate_embodiment_v2_plan(plan):
    """New mode only: explicit artifact/source pins, never implicit legacy loading."""
    if plan.get('mode') != EMBODIMENT_V2_MODE:
        return None
    inputs = plan.get('embodiment_v2')
    fields = ('cpu_reference_file', 'gpu_qualification_file', 'readout_state_file')
    required = {'schema'} | set(fields) | {name.removesuffix('_file')+'_sha256'
                                         for name in fields}
    if (not isinstance(inputs, dict) or set(inputs) != required
            or inputs.get('schema') != EMBODIMENT_V2_INPUT_SCHEMA):
        raise ValueError('Explicit separate v2 reference/GPU/readout artifact contract required')
    pinned = plan.get('input_sha256', {})
    for name in fields:
        path, digest = inputs[name], inputs[name.removesuffix('_file')+'_sha256']
        if (not isinstance(path, str) or not path or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or pinned.get(path) != digest):
            raise ValueError('Every v2 artifact must be explicitly hash-pinned in the plan')
    for path in (*EMBODIMENT_V2_REQUIRED_SOURCES,
                 plan.get('neural_spec'), plan.get('neural_checkpoint')):
        if (not isinstance(path, str) or path not in pinned
                or not isinstance(pinned[path], str) or len(pinned[path]) != 64):
            raise ValueError('Every v2 component source/spec/checkpoint requires a plan hash')
    return {key: value for key, value in inputs.items() if key != 'schema'}


def make_neural_component(plan, *, episode_id):
    """Keep old trial construction exact; new mode has a separate proof-gated class."""
    inputs = validate_embodiment_v2_plan(plan)
    if inputs is not None:
        from flybrain_drive.flight_embodiment_trial import NeuralEmbodimentV2Probe

        return NeuralEmbodimentV2Probe(plan['neural_spec'], plan['neural_checkpoint'],
                                      episode_id=episode_id, **inputs)
    from flybrain_drive.flight_maneuver_trial import NeuralPhysicalProbe

    return NeuralPhysicalProbe(plan['neural_spec'], plan['neural_checkpoint'],
                               episode_id=episode_id)


def validate_embodiment_v2_return(control, metadata):
    """The generic canonical field must identify exactly the command sent to codec."""
    values = np.asarray(control)
    declared = np.asarray(metadata.get('canonical_request'))
    if (metadata.get('canonical_request_schema') != EMBODIMENT_V2_CANONICAL_SCHEMA
            or values.shape != (3,) or values.dtype.kind != 'f'
            or not np.isfinite(values).all() or not np.array_equal(values, declared)):
        raise ValueError('Returned v2 canonical command and versioned metadata must match exactly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    embodiment_v2_mode = plan.get('mode') == EMBODIMENT_V2_MODE
    native_mode = embodiment_v2_mode or plan.get('mode') == 'native_neural_actuator_component'
    if embodiment_v2_mode:
        validate_embodiment_v2_plan(plan)
    neural_mode = native_mode or plan.get('mode') in (
        'neural_physical_component', 'neural_actuator_component')
    pedal_mode = plan.get('mode') == 'stationary_pedal_transfer'
    codec_mode = native_mode or plan.get('mode') in (
        'stationary_actuator_codec', 'neural_actuator_component')
    stationary_mode = pedal_mode or plan.get('mode') == 'stationary_actuator_codec'
    expected_program = NEURAL_PROGRAM if neural_mode else (
        ACTUATOR_PROGRAM if codec_mode else PEDAL_PROGRAM if pedal_mode else PLANT_PROGRAM)
    if plan['program'] != expected_program:
        raise ValueError('Exact bounded first-gear measurement program required')
    if codec_mode and plan.get('actuator_codec') != dict(
            contract=CODEC_CONTRACT, minimum_positive=.02):
        raise ValueError('Explicit measured minimum-positive codec contract required')
    stationary_codec = plan.get('mode') == 'stationary_actuator_codec'
    if stationary_codec and plan.get('stationary_sampling') != dict(
            maximum_samples_per_stage=8, minimum_samples_per_stage=5,
            file_timeout_seconds=.5, preserves_original_timestamps=True):
        raise ValueError('Explicit finite stationary sampling contract required')
    for path, digest in plan['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Declared measurement input changed')
    if native_mode:
        qualification = json.loads(Path(plan['native_position_qualification']).read_text())
        if (qualification['status'] != 'completed_stationary_native_position_qualification'
                or qualification['samples'] != 24
                or qualification['maximum_coordinate_error_m'] > .02
                or plan['diagnostic_world_input_to_actor'] is not False):
            raise ValueError('Explicit stationary-qualified native diagnostic coordinates required')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), scope=__doc__, actor_input=False,
        model_action_requests=0, ui_actions_sent=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', actor_input=False, model_action_requests=0,
        ui_actions_sent=0, actor_deployed=False, goal_achieved=False)
    reader = ai = session = position_reader = None
    rows, requests, neural_rows = [], [], []
    neural_requests_sent = 0
    origin, damage_start, generation, engine_session = None, None, None, None
    stream = (run/'events.jsonl').open('x', encoding='utf8')

    def emit(kind, value):
        stream.write(json.dumps(dict(kind=kind, **value), allow_nan=False)+'\n')
        stream.flush()

    try:
        axis = json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
        cal = json.loads(Path(plan['calibration']).read_text(encoding='utf8'))
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one reviewed AC engine required')
        window = windows[0]
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        if sdk.GetVJDStatus(axis['device_id']) != 1:
            raise ValueError('Reviewed vJoy device must be FREE')
        reader = NativePhysicsReader(timeout=.15, clock=time.perf_counter)
        ai = NativeTeacherStatusReader()
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry',
                                     timeout=.5 if stationary_codec else .15, poll_seconds=.001)
        initial_packet = None
        if native_mode:
            position_reader = NativeDiagnosticPositionReader()
            if (qualification['engine_pid'] != window.pid
                    or qualification['producer_session_id'] != cal['producer_session_id']):
                raise ValueError('Native position qualification belongs to a different engine')
        status_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        calibration_rows = [json.loads(line) for line in
            (Path(plan['calibration']).parent/'samples.jsonl').read_text().splitlines()]
        calibration_rows = [row for row in calibration_rows if 'stage' in row]
        if (cal['status'] != 'completed' or cal['samples'] != 32 or len(calibration_rows) != 32
                or cal['request_amplitude'] != .1 or cal['profile_sha256'] != axis['profile_sha256']
                or cal['bindings'] != axis['bindings'] or cal['dll_sha256'] != axis['dll_sha256']
                or cal['device_id'] != axis['device_id']):
            raise ValueError('Fresh explicit small-domain32-sample axis calibration required')
        for row in calibration_rows:
            if hashlib.sha256(base64.b64decode(row['native']['prefix_base64'])).hexdigest() != (
                    row['native']['prefix_sha256']):
                raise ValueError('Native steering calibration bytes changed')
            if (row['native_ai_before']['builtin_ai_controlled']
                    or row['native_ai_after']['builtin_ai_controlled']):
                raise ValueError('AI-off calibration required')
        medians = [float(np.median([row['native']['fields']['steerAngle']
            for row in calibration_rows
            if row['stage'] == stage and row['index'] >= 2])) for stage in range(4)]
        native_gain = (medians[1]-medians[2])/.2
        if native_gain <= 0 or max(abs(medians[0]), abs(medians[3])) > .001:
            raise ValueError('Independent native steering direction/neutral not measured')

        def receive(phase):
            nonlocal origin, damage_start, generation, engine_session, initial_packet
            before = time.perf_counter()
            flag_before = ai.snapshot()
            if native_mode and initial_packet is None:
                initial_packet = client.receive()
            packet = initial_packet if native_mode else client.receive()
            sample = reader.receive()
            graphics = position_reader.snapshot() if native_mode else None
            status, modified = read_published_json(status_path)
            flag, values = ai.snapshot(), native_values(sample)
            if generation is None:
                generation, engine_session = status['generation'], packet.raw['producer_session_id']
            if (status['generation'] != generation or status.get('paused') is not False
                    or status.get('in_menu') is not False or status.get('is_online') is not False
                    or status.get('ai_controlled') is not False or status.get('camera_active')
                    or status.get('expert_probe_active') or status.get('error')
                    or status.get('pending_receipt_id') or flag['builtin_ai_controlled']
                    or flag_before['builtin_ai_controlled']
                    or not 0 <= time.time()-modified <= 2
                    or status.get('car_id') != 'bmw_z4_gt3' or status.get('track_id') != 'monza'
                    or api.probe(window.hwnd, require_foreground=False) != window
                    or api.user.GetForegroundWindow() == window.hwnd
                    or sha256(axis['profile_path']) != axis['profile_sha256']
                    or packet.raw['producer_session_id'] != engine_session
                    or engine_session != cal['producer_session_id']
                    or not engine_session.startswith('ac-file-'+str(window.pid)+'-')):
                raise ValueError('Fresh same background AI-off engine/profile/source required')
            if native_mode:
                if graphics.status != 'live' or status['generation'] != qualification['generation']:
                    raise ValueError('Live same-generation native diagnostic position required')
                xy = np.array(graphics.coordinates_native_xyz_m)[[0, 2]]
            else:
                namespace = packet.raw['privileged_teacher']
                if namespace['actor_input'] is not False:
                    raise ValueError('World displacement is diagnostic-only')
                truth = namespace['raw']
                xy = np.array([truth['world_position_x'], truth['world_position_y']], float)
            if not np.isfinite(xy).all():
                raise ValueError('Diagnostic displacement bound unavailable')
            if origin is None:
                origin, damage_start = xy.copy(), np.asarray(values['carDamage'])
            displacement = float(np.linalg.norm(xy-origin))
            damage_delta = float((np.asarray(values['carDamage'])-damage_start).max())
            row = dict(phase=phase, read_started_perf=before,
                read_finished_perf=time.perf_counter(),
                native=physics_record(sample),
                sdk=record_packet(packet) if not native_mode or phase == 'preflight' else None,
                native_graphics=graphics.metadata() if native_mode else None,
                diagnostic_sdk_initial_sequence=packet.sequence if native_mode else None,
                native_ai=flag,
                native_ai_before=flag_before, generation=generation,
                gear=values['gear'], rpm=values['rpms'],
                yaw_left_rad_s=values['localAngularVel'][1],
                displacement_m=displacement, wheels_outside=values['numberOfTyresOut'],
                damage_delta=damage_delta, csp_clock=status['script_clock'])
            rows.append(row)
            emit('sample', row)
            validate_response_envelope(sample.fields['speedKmh'], displacement, damage_delta,
                                       values['numberOfTyresOut'], values['rpms'])
            if len(rows) > 400:
                raise ValueError('Finite sensor sample budget exceeded')
            return row, status, sample, flag

        neural = None
        if neural_mode:
            neural = make_neural_component(plan, episode_id=run.name)
            write_json(run/'neural-motor-reference.json', neural.reference)
            if embodiment_v2_mode:
                write_json(run/'neural-embodiment-v2-initialization.json',
                           neural.embodiment_initialization)
        first, status, sample, flag = receive('preflight')
        validate_stationary_calibration_state(status, 0., flag, sample.fields, generation)
        if first['gear'] not in (1, 2):
            raise ValueError('Native neutral or first gear required')
        if native_mode:
            time.sleep(.04)
            second, status, sample, flag = receive('native_preflight_advancing')
            validate_stationary_calibration_state(status, 0., flag, sample.fields, generation)
            if (second['native']['packet_id'] <= first['native']['packet_id']
                    or second['native_graphics']['packet_id'] <= (
                        first['native_graphics']['packet_id'])
                    or second['gear'] != first['gear']):
                raise ValueError('Both native streams must advance before device acquisition')
            first = second
        session = VJoySession(sdk, axis['device_id'], [AxisBinding(**b) for b in axis['bindings']])

        def submit(phase, steering, throttle, brake, buttons=()):
            if abs(steering) > .05 or not 0 <= throttle <= .15 or not 0 <= brake <= 1:
                raise ValueError('Predeclared low-speed request bound exceeded')
            request = session.submit(steering, throttle, brake, pressed_buttons=buttons)
            requests.append(request)
            emit('request', dict(phase=phase, receipt=request))
            return request

        requests.append(session.open())
        emit('request', dict(phase='open', receipt=requests[-1]))
        required_gear = 1 if stationary_mode else 2
        transition = single_gear_transition(first['gear'], required_gear)
        if transition is not None:
            button_name, button = transition
            if axis['gear_buttons_zero_based'][button_name] != button:
                raise ValueError('Reviewed single gear-transition button required')
            receive('before_single_gear_'+button_name)
            submit('single_gear_'+button_name, 0., 0., 1., (button,))
            time.sleep(.12)
            submit('release_gear_button', 0., 0., 1.)
            time.sleep(.12)
        engaged, _, _, _ = receive('required_gear_check')
        if engaged['gear'] != required_gear:
            raise ValueError('Single gear transition did not produce required gear; never repeat')
        for stage_index, stage in enumerate(plan['program']):
            if neural_mode:
                phase, desired_yaw, desired_speed, duration = stage
            else:
                phase, steering, throttle, brake, duration = stage
            deadline = time.perf_counter()+duration
            stage_samples = 0
            while (time.perf_counter() < deadline
                   and (not stationary_codec or stage_samples < 8)):
                row, _, current_sample, _ = receive(phase)
                if row['gear'] != required_gear:
                    raise ValueError('Unexpected gear during plant measurement')
                if stationary_mode and abs(row['native']['fields']['speedKmh']) >= .1:
                    raise ValueError('Pedal calibration must remain independently stationary')
                if neural is not None:
                    command, metadata = neural.request(current_sample, [desired_yaw, desired_speed],
                                                        time.perf_counter())
                    if embodiment_v2_mode:
                        validate_embodiment_v2_return(command, metadata)
                    neural_rows.append(dict(phase=phase, **metadata))
                    emit('neural_component', neural_rows[-1])
                    steering, throttle, brake = (float(v) for v in command)
                if codec_mode and (neural_mode or phase.startswith('encoded_')):
                    canonical = [steering, throttle, brake] if neural_mode else stage[1:4]
                    encoding = project_pedals(canonical)
                    emit('actuator_encoding', dict(phase=phase, stage_index=stage_index,
                        native_packet=row['native']['packet_id'], **encoding))
                    steering, throttle, brake = encoding['encoded_request']
                submit(phase, steering, throttle, brake)
                stage_samples += 1
                if neural is not None:
                    neural_requests_sent += 1
                time.sleep(.04)
            if stationary_codec and stage_samples < 5:
                raise ValueError('Stationary stage lacks predeclared settling/steady samples')
        result.update(status='completed_low_speed_maneuver_measurement',
            generation=generation, producer_session_id=engine_session,
            native_steering_gain=native_gain, first_gear_observed=required_gear == 2,
            required_gear_observed=required_gear,
            max_speed_kmh=max(row['native']['fields']['speedKmh'] for row in rows),
            max_displacement_m=max(row['displacement_m'] for row in rows))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        stopped = False
        if session is not None and session.owned:
            try:
                deadline, stable = time.perf_counter()+3, 0
                while time.perf_counter() < deadline:
                    request = session.submit(0., 0., 1.)
                    requests.append(request)
                    emit('request', dict(phase='final_brake_hold', receipt=request))
                    sample = reader.receive()
                    emit('final_native', dict(native=physics_record(sample)))
                    f = sample.fields
                    stable = stable+1 if (abs(f['speedKmh']) < .1 and f['gas'] < .02
                                          and f['brake'] > .98) else 0
                    if stable >= 3:
                        stopped = True
                        break
                    time.sleep(.04)
            except Exception as error:
                result['cleanup_error'] = str(error)
            finally:
                try:
                    requests.append(session.close())
                    emit('request', dict(phase='close', receipt=requests[-1]))
                    result['vjoy_free_after_close'] = sdk.GetVJDStatus(axis['device_id']) == 1
                except Exception as error:
                    result['close_error'] = str(error)
        result['stopping_native_acknowledged'] = stopped
        if not stopped or result.get('vjoy_free_after_close') is not True:
            result['status'] = 'failed'
        if reader is not None:
            reader.close()
        if ai is not None:
            ai.close()
        if position_reader is not None:
            position_reader.close()
        stream.close()
        result.update(samples=len(rows),
                      actuator_messages_sent=session.submit_attempts if session else 0,
                      neural_component_decisions=len(neural_rows),
                      model_action_requests=neural_requests_sent, vision_used=False,
                      component_model_controls_sent=neural_requests_sent > 0,
                      native_body_and_diagnostic_position_path=native_mode,
                      events_sha256=sha256(run/'events.jsonl'))
        write_json(run/'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result.get('reason', 'Final native stopping state unverified'))


if __name__ == '__main__':
    main()
