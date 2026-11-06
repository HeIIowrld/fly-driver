"""Unadmitted read-only transition monitoring; final thermal contract is unchanged."""
import math


def validate_transition_context(state,before,allowed_pending):
    expected=dict(is_online=False,in_menu=False,ai_controlled=False,camera_active=False,
                  expert_probe_active=False,brake_settle_active=False,physics_api_allowed=True)
    if (any(state.get(k) is not v for k,v in expected.items())
            or state.get('generation')!=before['generation'] or state.get('epoch')!=before['epoch']
            or state.get('car_id')!='bmw_z4_gt3' or state.get('track_id')!='monza'
            or state.get('tyres_initialization_epoch')!=before['epoch']
            or state.get('error') or state.get('pending_receipt_id') not in allowed_pending):
        raise ValueError('Owned initialized offline thermal transition context changed')
    values=[state.get(k) for k in ('speed_kmh','gas','brake')]
    if (not all(type(v) in (int,float) and math.isfinite(v) for v in values)
            or abs(values[0])>5 or not 0<=values[1]<.02 or not .98<values[2]<=1):
        raise ValueError('Brake-held thermal transition observation exceeded5km/h envelope')


def validate_transition_native(coupled):
    f=coupled.physics.fields
    if (coupled.builtin_ai_controlled is not False or abs(f['speedKmh'])>5
            or not 0<=f['gas']<.02 or not .98<f['brake']<=1):
        raise ValueError('Native thermal transition authority or brake-held envelope changed')
