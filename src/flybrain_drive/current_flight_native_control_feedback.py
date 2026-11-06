"""Empirical native steering feedback codec from same-profile original calibration bytes.

SDK raw and native raw are different quantities. No physical angle unit or
unmeasured whole-domain linearity is asserted; this is not an actor input.
"""
import base64,hashlib,json,math,statistics
from pathlib import Path
from .ac_native_physics import decode_physics_prefix
from .graph import sha256


def calibration_native_steering(directory):
    root=Path(directory);result=json.loads((root/'result.json').read_text(encoding='utf8'))
    rows=[json.loads(v) for v in (root/'samples.jsonl').read_text(encoding='utf8').splitlines()]
    rows=[v for v in rows if 'stage' in v]
    if result['status']!='completed' or len(rows)!=32 or result['samples']!=32:raise ValueError('Original completed32 calibration samples required')
    previous=-1
    for row in rows:
        native=row['native'];raw=base64.b64decode(native['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=native['prefix_sha256']:raise ValueError('Original native calibration bytes differ')
        sample=decode_physics_prefix(raw,native['received_monotonic_seconds'])
        expected={k:tuple(v) if isinstance(v,list) else v for k,v in native['fields'].items()}
        if (sample.packet_id!=native['packet_id'] or sample.packet_id<=previous
                or dict(sample.fields)!=expected or abs(sample.fields['speedKmh'])>2
                or sample.fields['gas']>.02 or sample.fields['brake']<.98
                or row['native_ai_before']['builtin_ai_controlled'] or row['native_ai_after']['builtin_ai_controlled']
                or row['was_foreground_at_record']):raise ValueError('Calibration bytes/authority/sequence differ')
        previous=sample.packet_id
    amplitude=result['request_amplitude'];stages=[];sdk=[]
    for stage,request in enumerate((0.,amplitude,-amplitude,0.)):
        selected=[v for v in rows if v['stage']==stage and v['index']>=2]
        if len(selected)!=6 or any(abs(v['request']['requested_control']['steering']-request)>1e-9 for v in selected):raise ValueError('Original calibrated request stages differ')
        stages.append(statistics.median(v['native']['fields']['steerAngle'] for v in selected))
        sdk.append(statistics.median(v['sdk_steer_raw'] for v in selected))
    if sdk!=result['sdk_steering_stage_medians'] or abs(stages[0])>.001 or abs(stages[3])>.001 or not stages[2]<0<stages[1]:raise ValueError('Calibrated native center/sign or SDK medians differ')
    native_gain=(stages[1]-stages[2])/(2*amplitude);sdk_gain=result['sdk_raw_per_normalized_request']
    if not math.isfinite(native_gain) or native_gain<=0 or not math.isfinite(sdk_gain) or sdk_gain<=0:raise ValueError('Positive distinct native and SDK gains required')
    return dict(schema='flybrain.current_flight.native_steering_feedback.v1',actor_input=False,
        native_raw_per_sdk_normalized_request=native_gain,sdk_raw_per_sdk_normalized_request=sdk_gain,
        native_stage_medians=stages,sdk_stage_medians=sdk,neutral_tolerance_native=.5*native_gain/sdk_gain,
        neutral_tolerance_sdk_raw=.5,profile_sha256=result['profile_sha256'],producer_session_id=result['producer_session_id'],
        samples_sha256=sha256(root/'samples.jsonl'),result_sha256=sha256(root/'result.json'),
        independent_physical_units_verified=False,whole_domain_linearity_verified=False)
