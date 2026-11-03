"""Prospective TRAIN identity and geographic observer contract; no live interfaces."""

import ast
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = 'flybrain.pedal_train_candidate.declaration.host_v4.v1'
PROTECTED = ((.08, .125), (.2, .25), (.67, .84), (.94, 1.))
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SEEDS = (
    'scripts/qualify_session_heartbeat_evidence_host_v4.py',
    'scripts/record_ac_engine_ai_probe_host_v4.py',
    'src/flybrain_drive/frame_persistence_profile_v4.py',
    'src/flybrain_drive/ac_csp_vision_host_v4.py', 'src/flybrain_drive/ac_csp_retina.py',
    'src/flybrain_drive/ac_native_physics.py', 'src/flybrain_drive/ac_teacher_physics.py',
    'src/flybrain_drive/ac_teacher_status.py', 'src/flybrain_drive/ac_image_buffer.py',
    'scripts/record_ac_wgc.py', 'src/flybrain_drive/session_heartbeat_runtime.py',
    'src/flybrain_drive/session_heartbeat_provenance.py',
    'src/flybrain_drive/session_heartbeat_evidence_v1.py',
    'src/flybrain_drive/session_heartbeat_terminal_v2.py',
    'src/flybrain_drive/session_heartbeat_admission.py',
    'src/flybrain_drive/ac_published_state.py', 'src/flybrain_drive/ac_csp_clock.py',
    'src/flybrain_drive/teacher_pace.py', 'src/flybrain_drive/sensor_contract.py',
    'scripts/control_ac_session.py',
    'src/flybrain_drive/virtual_navigation_body.py', 'src/flybrain_drive/virtual_body.py',
    'src/flybrain_drive/teacher_loss_eligibility.py',
    'src/flybrain_drive/pedal_train_candidate_host_v4.py',
    'src/flybrain_drive/pedal_train_loader_host_v4.py',
    'scripts/record_pedal_train_candidate_host_v4.py',
    'configs/ac-frame-host-v4-20261030.json',
    'configs/ac-virtual-navigation-body-v2.json', 'configs/ac-virtual-body-v1.json',
    'configs/ac-csp-exterior-front-v3-retina.json', 'configs/ac-axis-probe.json',
    'src/flybrain_drive/immutable_frame_writer.py', 'src/flybrain_drive/ac_capture.py',
    'src/flybrain_drive/ac_vjoy.py', 'src/flybrain_drive/evidence.py',
    'scripts/prepare_pedal_train_host_v4.py', 'scripts/prepare_frame_host_v4.py',
    'scripts/prepare_pedal_train_collector_host_v4.py',
    'scripts/prepare_pedal_train_declaration_host_v4.py',
    'scripts/load_pedal_train_candidate_host_v4.py',
    'integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua',
    'integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua',
)


def local_import_closure(entries, *, project_root=_PROJECT_ROOT):
    """AST-only conservative src/scripts closure, including imports inside functions.

    No discovered module is imported/executed, no packages are searched, and no
    arbitrary directory traversal is performed. Inactive optional local imports
    are pinned too; this deliberately over-approximates the capture code path.
    """
    project_root = Path(project_root)
    sources = set(entries)
    pending = [p for p in entries if p.endswith('.py')]

    def resolve(module):
        parts = module.split('.')
        if parts[0] not in ('flybrain_drive', 'scripts'):
            return None
        prefix = Path('src') if parts[0] == 'flybrain_drive' else Path()
        target = prefix.joinpath(*parts).with_suffix('.py')
        if not (project_root/target).is_file():
            target = prefix.joinpath(*parts, '__init__.py')
        return target.as_posix() if (project_root/target).is_file() else None

    while pending:
        name = pending.pop()
        path = Path(name)
        parts = list(path.with_suffix('').parts)
        if parts[0] == 'src':
            parts = parts[1:]
        if parts[-1] == '__init__':
            parts = parts[:-1]
            package = parts
        else:
            package = parts[:-1]
        for node in ast.walk(ast.parse((project_root/path).read_text(encoding='utf8'))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                module = '.'.join(base+node.module.split('.')) if node.module else '.'.join(base)
                modules = [module, *(module+'.'+alias.name for alias in node.names)]
            for module in modules:
                for candidate in (module, module.split('.')[0]):
                    target = resolve(candidate)
                    if target and target not in sources:
                        sources.add(target)
                        pending.append(target)
    return tuple(sorted(sources))


def consumed_configuration_bindings(*, project_root=_PROJECT_ROOT):
    """Explicit runtime-consumed file refs; unrelated historical pixel refs stay unopened."""
    root = Path(project_root)
    navigation = json.loads((root/'configs/ac-virtual-navigation-body-v2.json').read_bytes())
    legacy_name = navigation['legacy_contract']
    legacy = json.loads((root/legacy_name).read_bytes())
    axis = json.loads((root/'configs/ac-axis-probe.json').read_bytes())
    return {legacy_name: navigation['legacy_contract_sha256'],
            legacy['geometry_review']: legacy['geometry_review_sha256'],
            axis['dll_path']: axis['dll_sha256']}


CONFIGURATION_BINDINGS = consumed_configuration_bindings()
SOURCE_FILES = tuple(sorted(set(local_import_closure(SOURCE_SEEDS))
                            | set(CONFIGURATION_BINDINGS)))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_declaration(value, *, sealed=True):
    if (value.get('schema') != SCHEMA or value.get('original_role') != 'train'
            or value.get('fit_allowed') is not False or value.get('optimizer_updates') != 0
            or value.get('duration_seconds') != 30
            or value.get('yaw_policy') != 'prospectively_declared_zero_component'
            or value.get('trainable_parameters') != [
                'command.power_delta', 'command.speed_feedforward_delta']
            or value.get('start_region') != [.275, .44]
            or value.get('whole_region') != [.275, .54]
            or value.get('protected_regions') != [list(r) for r in PROTECTED]
            or value.get('pedal_projection_policy') != 'reject_unreachable'
            or value.get('minimum_positive') != .02):
        raise ValueError('Exact prospective whole-history TRAIN component contract required')
    if (value.get('profile') != 'configs/ac-frame-host-v4-20261030.json'
            or value.get('host_copy_profile_schema') != 'flybrain.frame_host_copy_profile.v4'):
        raise ValueError('Explicit separately qualified hostV4 profile required')
    if value.get('body_contract') != 'configs/ac-virtual-navigation-body-v2.json':
        raise ValueError('Explicit unchanged body21 contract required')
    if not sealed:
        return
    if value.get('actual_pipeline_qualification_completed') is not True:
        raise ValueError('Actual reviewed hostV4 pipeline qualification required '
                         'before role seal/capture')
    bindings = value.get('bindings', {})
    if (value.get('status') != 'sealed_before_capture'
            or not re.fullmatch('[0-9a-f]{32}', str(value.get('nonce')))
            or type(bindings.get('engine_pid')) is not int or bindings['engine_pid'] <= 0
            or type(bindings.get('epoch')) is not int or bindings['epoch'] < 0
            or not isinstance(bindings.get('generation'), str) or not bindings['generation']
            or not is_sha(bindings.get('installed_lua_sha256'))
            or not isinstance(bindings.get('output'), str) or not bindings['output']
            or not is_sha(value.get('native_pedal_calibration_sha256'))
            or not is_sha(value.get('native_pedal_continuity_review_sha256'))
            or not is_sha(value.get('route_preflight_sha256'))
            or type(value.get('expected_start_spline')) not in (int, float)
            or not .275 <= value['expected_start_spline'] <= .44
            or not is_sha(value.get('pipeline_qualification_review_sha256'))
            or set(SOURCE_FILES)-set(value.get('source_sha256', {}))):
        raise ValueError('Fresh identity/calibration/source qualification bindings required')
    if any(not is_sha(v) for v in value['source_sha256'].values()):
        raise ValueError('Explicit sealed source hashes required')
    if any(value['source_sha256'][p] != expected
           for p, expected in CONFIGURATION_BINDINGS.items()):
        raise ValueError('Consumed body geometry/DLL reference binding differs')
    if any(value['source_sha256'][p] != bindings['installed_lua_sha256'] for p in (
            'integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua',
            'integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua')):
        raise ValueError('Actual child main/candidate Lua source binding differs')


def seal_declaration(value, destination):
    """Write a new immutable role file only; never discover a PID or launch capture."""
    validate_declaration(value, sealed=False)
    sealed = dict(value, status='sealed_before_capture')
    validate_declaration(sealed)
    for name, expected in sealed['source_sha256'].items():
        if sha_file(name) != expected:
            raise ValueError('No automatic source repinning: '+name)
    raw = (json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with Path(destination).open('xb') as stream:
        stream.write(raw)
    return dict(sha256=sha_bytes(raw), nonce=sealed['nonce'], original_role='train',
                collection_started=False, fit_allowed=False)


def collector_declaration(args):
    raw = Path(args.train_declaration).read_bytes()
    value = json.loads(raw)
    validate_declaration(value)
    b = value['bindings']
    if (args.mode != 'pace30' or args.expected_engine_pid != b['engine_pid']
            or args.expected_generation != b['generation'] or args.expected_epoch != b['epoch']
            or args.expected_installed_sha256 != b['installed_lua_sha256']
            or Path(args.output).resolve() != Path(b['output']).resolve()
            or sha_file(args.frame_persistence_profile) != value['source_sha256'][
                'configs/ac-frame-host-v4-20261030.json']):
        raise ValueError('Exact prospectively sealed capture identity/output/profile required')
    for name, expected in value['source_sha256'].items():
        if sha_file(name) != expected:
            raise ValueError('Sealed source changed; no repin/retry: '+name)
    return dict(raw=raw, value=value, sha256=sha_bytes(raw), nonce=value['nonce'])


def pedal_gate(review_bytes, declaration):
    """Bind an EXTERNAL review to original plant calibration and fresh nonce/path.

    Parsing an attestation is not physical recalibration or independent verification.
    The originating reviewer/source evidence must remain separately reviewable.
    """
    if review_bytes is None:
        return dict(attached=False, qualified=False, reason='external_review_missing')
    value = json.loads(review_bytes)
    if (sha_bytes(review_bytes) != declaration['native_pedal_continuity_review_sha256']
            or value.get('schema') != 'flybrain.pedal_calibration_continuity_review.v1'
            or value.get('original_calibration_sha256') != (
                declaration['native_pedal_calibration_sha256'])
            or value.get('nonce') != declaration['nonce']
            or value.get('current_engine_pid') != declaration['bindings']['engine_pid']
            or value.get('current_generation') != declaration['bindings']['generation']
            or value.get('physical_pedal_input_path_qualified') is not True
            or value.get('steering_qualification_claimed') is not False
            or value.get('fit_allowed') is not False
            or value.get('source_sha256') != declaration['source_sha256']):
        raise ValueError('Original pedal calibration/fresh continuity review required')
    return dict(attached=True, qualified=True, review_sha256=sha_bytes(review_bytes),
        qualification_is_external_not_loader_measured=True)


def route_admission(receipt, declaration):
    """Check every retained observer sample, never manufacture continuous coverage."""
    reasons = []
    samples = [receipt.get('before'), *receipt.get('telemetry', []), receipt.get('after')]
    positions, laps = [], []
    for sample in samples:
        route = sample.get('diagnostic_route') if isinstance(sample, dict) else None
        if not isinstance(route, dict) or route.get('actor_input') is not False:
            reasons.append('missing_original_observer_route')
            continue
        p, lap = route.get('spline_position'), route.get('lap_count')
        if (type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p < 1
                or type(lap) not in (int, float) or not math.isfinite(lap)
                or lap < 0 or int(lap) != lap):
            reasons.append('invalid_original_observer_route')
            continue
        positions.append(p)
        laps.append(lap)
        if not .275 <= p <= .54:
            reasons.append('whole_route_outside_train_region')
        if any(lo <= p <= hi for lo, hi in PROTECTED):
            reasons.append('protected_DEV_EVAL_overlap')
        if sample.get('wheels_outside') != 0:
            reasons.append('offtrack_or_unknown_wheels')
    telemetry = receipt.get('telemetry', [])
    if not telemetry or not {'driving', 'stopping'} <= {s.get('phase') for s in telemetry}:
        reasons.append('before_driving_stopping_after_coverage_incomplete')
    if positions:
        if not .275 <= positions[0] <= .44:
            reasons.append('start_outside_declared_train_start_region')
        if abs(positions[0]-declaration['expected_start_spline']) > 1e-4:
            reasons.append('original_start_differs_from_preassigned_observer_start')
        if any(a-b > .001 for a, b in zip(positions, positions[1:], strict=False)):
            reasons.append('route_reverse_wrap_or_discontinuity')
        if len(set(laps)) != 1:
            reasons.append('lap_changed')
    return dict(passed=not reasons, reasons=sorted(set(reasons)),
        observer_samples=len(samples), observed_positions=len(positions),
        minimum_spline=min(positions) if positions else None,
        maximum_spline=max(positions) if positions else None,
        continuous_geometry_verified=False, actor_input=False, entire_history_retained=True,
        external_observer_coverage_qualification_required=True, fit_allowed=False)
