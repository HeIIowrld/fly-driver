"""Explicit MMF lifecycle-only camera-status contract; no live work on import."""

import json
from pathlib import Path

POLICY = 'mmf_lifecycle_only_v1'
ECHO_KEY = 'camera_status_policy_echo'
ECHO_SCHEMA = 'flybrain.csp.camera.status_policy_echo.v1'


def profile_record():
    return dict(schema='flybrain.csp.camera.status_publication_profile.v1', policy=POLICY,
        transport='shared_immutable_MMF_only', periodic_active_status_write=False,
        initial_terminal_idle_error_writes_unchanged=True,
        clocks_frames_ACK_cleanup_and_limits_unchanged=True)


def load_profile(path):
    return from_record(json.loads(Path(path).read_bytes()))


def from_record(value):
    expected = profile_record()
    if (type(value) is not dict or set(value) != set(expected)
            or any(type(value[k]) is not type(v) or value[k] != v
                   for k, v in expected.items())):
        raise ValueError('Exact explicit MMF lifecycle-only status profile required')
    return value


def validate_requested_policy(value, *, mmf_metadata, shared_transport):
    if value is None:
        return None
    if (type(value) is not str or value != POLICY
            or mmf_metadata is not True or shared_transport is not True):
        raise ValueError('Exact MMF-only lifecycle camera-status policy required')
    return POLICY


def validate_producer_echo(value, *, policy, request_id, generation):
    if policy is None:
        if value is not None:
            raise ValueError('Unrequested camera-status policy echo')
        return
    expected = dict(schema=ECHO_SCHEMA, policy=POLICY, request_id=request_id,
                    generation=generation)
    if policy != POLICY or value != expected:
        raise ValueError('Exact accepted producer policy echo required;807 ignore is unsupported')
