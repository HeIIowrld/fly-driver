"""Pure native bytes/orchestration fixtures; no SDK/window/maps/child/model/GPU calls."""

import ast
import copy
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import pytest

from flybrain_drive.native_transit_observer_v1 import (
    decode_observer,
    stable_native_copy,
    validate_pair,
)
from scripts.prepare_native_transit_observer_v1 import SOURCE, TARGET, build
from scripts.transit_native_to_train_v1 import (
    MANDATORY,
    PRIOR_FILES,
    PRIOR_REVIEW,
    PRIOR_REVIEW_SHA256,
    PRIOR_ROOT,
    ROOT_TERMINAL_BASIS,
    child_arguments,
    decision,
    execute_stages,
    validate_plan,
    wait_native_child,
)


def plan():
    value = json.loads(Path('configs/flight-native-transit-to-train-20261030.json').read_bytes())
    value['input_sha256'] = {p: 'a'*64 for p in MANDATORY}
    value['input_sha256'].update({str(PRIOR_ROOT/p).replace('\\', '/'): 'a'*64
                                  for p in PRIOR_FILES})
    value['input_sha256'][PRIOR_REVIEW.as_posix()] = PRIOR_REVIEW_SHA256
    value.update(prior_independent_review_file=PRIOR_REVIEW.as_posix(),
        prior_independent_review_sha256=PRIOR_REVIEW_SHA256,
        root_original_terminal_basis=ROOT_TERMINAL_BASIS)
    return value


def test_observer_exact_validated_derivative_and_no_camera_or_model_import():
    assert TARGET.read_bytes() == build(SOURCE.read_bytes())
    tree = ast.parse(TARGET.read_text(encoding='utf8'))
    assert not any(isinstance(n, ast.FunctionDef) and n.name in ('coordinate_link', 'validate_plan')
                   for n in tree.body)
    text = TARGET.read_text(encoding='utf8')
    assert 'ac_csp_vision' not in text and 'load_vjoy_sdk' in text
    assert 'AcquireVJD(' not in text and 'torch' not in text


def test_existing_command_no_camera_argument_and_global_budget_counts1837():
    value = plan()
    validate_plan(value)
    args = child_arguments(Path('fixture-output'))
    assert '--camera-request-id' not in args and args.count('--execute') == 1
    assert 'expert-motion-speed60' in args and '.9' in args
    for key, bad in [('prior_registered_activations', 0), ('maximum_new_activations', 30),
                     ('maximum_total_activations', 31), ('parent_process_wait_seconds', 14)]:
        changed = copy.deepcopy(value)
        changed[key] = bad
        with pytest.raises(ValueError):
            validate_plan(changed)


def test_mandatory_original1837_raw_observer_and1838terminal_links_cannot_be_omitted():
    value = plan()
    del value['input_sha256'][str(PRIOR_ROOT/PRIOR_FILES[-1]).replace('\\', '/')]
    with pytest.raises(ValueError, match='original1837'):
        validate_plan(value)
    value = plan()
    value['root_original_terminal_basis'] = dict(ROOT_TERMINAL_BASIS, terminal_exit_code=1)
    with pytest.raises(ValueError, match='terminal2103'):
        validate_plan(value)


def test_primary_native264_decoder_and_unchanged_frozen_physics_pair():
    raw = bytearray(264)
    struct.pack_into('<iii', raw, 0, 100, 3, 0)
    struct.pack_into('<f3f', raw, 248, .10346428, 1., 2., 3.)
    first = decode_observer(bytes(raw), 10.)
    assert first['lap_count'] == 0 and first['actor_input'] is False
    assert first['spline_position'] == pytest.approx(.10346428)
    second = dict(first, graphics_packet=101, received_perf_counter_seconds=10.1)
    a = dict(observer=first, physics420_sha256='a'*64, native_packet=7,
             status=dict(script_clock=1.))
    b = dict(a, observer=second, status=dict(script_clock=1.1))
    validate_pair(a, b)
    with pytest.raises(ValueError):
        validate_pair(a, dict(b, physics420_sha256='b'*64))
    with pytest.raises(ValueError):
        validate_pair(a, dict(b, observer=first))


def test_repeated_copy_remains_bounded_without_creating_mapping():
    class Mapping:
        count = 0
        def read(self):
            self.count += 1
            return b'raw'
    value = Mapping()
    assert stable_native_copy(value) == (b'raw', 1) and value.count == 2


def callbacks(tmp_path, points, *, failed_child=None):
    raw = bytes(420)
    file = tmp_path/'SYNTHETIC-physics.bin'
    file.write_bytes(raw)
    calls = []
    def state(point):
        return dict(observer=dict(spline_position=point, lap_count=0),
                    samples=[dict(physics420_file=str(file), native_packet=7)])
    def observe(after, index):
        return state(points[index])
    def child(index):
        calls.append(index)
        if failed_child == index:
            raise RuntimeError('original child failed')
        return dict(after_source=index+1.,
            before_route=dict(spline_position=points[index], lap_count=0, actor_input=False),
            after_route=dict(spline_position=points[index+1], lap_count=0, actor_input=False),
            native_after_packet=7, native_after_prefix_sha256=hashlib.sha256(raw[:308]).hexdigest())
    return observe, child, calls


def run_stages(tmp_path, points, *, failed=None):
    value = plan()
    value['expected_initial_spline'] = points[0]
    observe, child, calls = callbacks(tmp_path, points, failed_child=failed)
    result = dict(attempts=[])
    outcome = execute_stages(value, result, observe=observe, child=child,
        verify=lambda: None, checkpoint=lambda: None, prior_gate={'original1837': True})
    return outcome, result, calls


def test_stop_at_first_eligible_point_and_preserve_all_whole_attempts(tmp_path):
    outcome, result, calls = run_stages(tmp_path, [.103, .13, .2, .28, .3])
    assert outcome['staging_start_reached'] and calls == [0, 1, 2]
    assert len(result['attempts']) == 3 and result['overall_activations_consumed'] == 4
    assert all(r['role'] == 'transit' and r['fit_allowed'] is False for r in result['attempts'])


def test_cap29_new_children_including_one_prior_never_becomes31(tmp_path):
    outcome, result, calls = run_stages(tmp_path, [.103+i*.001 for i in range(30)])
    assert len(calls) == 29 and result['overall_activations_consumed'] == 30
    assert not outcome['staging_start_reached']


@pytest.mark.parametrize('points', [[.103, .09], [.103, .45]])
def test_wrap_reverse_or_overshoot_stops_before_second_child(tmp_path, points):
    with pytest.raises(ValueError):
        run_stages(tmp_path, points)


def test_original_child_failure_stops_before_any_retry(tmp_path):
    with pytest.raises(RuntimeError, match='original child failed'):
        run_stages(tmp_path, [.103, .13, .2], failed=0)


def test_timeout_retains_original_pid_never_kills_terminates_or_retries():
    class Child:
        pid = 12345
        calls = []
        def wait(self, timeout):
            self.calls.append(('wait', timeout))
            raise subprocess.TimeoutExpired('original', timeout)
        def kill(self):
            raise AssertionError('must not kill')
        def terminate(self):
            raise AssertionError('must not terminate')
    child, result = Child(), {}
    with pytest.raises(RuntimeError, match='unknown'):
        wait_native_child(child, result, seconds=30)
    assert child.calls == [('wait', 30)] and result['owned_child_still_live'] == 12345
    assert result['timeout_original_child_pid'] == 12345 and result['unknown_child_outcome']


def test_decision_boundaries_and_nonnumeric_source():
    assert decision(.275) == decision(.44) == 'stop_reached'
    assert decision(.274999) == 'continue'
    for value in (.440001, float('nan'), -1):
        with pytest.raises(ValueError):
            decision(value)
