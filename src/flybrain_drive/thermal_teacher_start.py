"""Teacher collection condition admission; never an actor observation."""
import math


def admit_thermal_teacher_start(audited,session,temperatures,*,source_sha256):
    """Require audited initialization AND current same-session temperatures±5C.

    The independent initializer verifies±2C on three packets. This separately
    declared start tolerance allows finite setup cooling, without claiming that
    the target remains a thermostat or substituting target for actual sensing.
    """
    target=audited.get('target_c')
    if (type(target) not in (int,float) or target not in (40,80,110)
            or audited.get('condition_preparation_verified') is not True
            or audited.get('consecutive_all_four_within_two_c',0)<3
            or audited.get('current_lap_excluded') is not True
            or audited.get('generation')!=session.get('generation')
            or audited.get('epoch')!=session.get('epoch')
            or session.get('tyres_initialization_epoch')!=session.get('epoch')
            or audited.get('deployed_lua_sha256')!=source_sha256):
        raise ValueError('Audited thermal run from this exact initialized producer/epoch/source required')
    required=dict(paused=True,is_online=False,in_menu=False,ai_controlled=False,
                  camera_active=False,expert_probe_active=False,brake_settle_active=False)
    if (any(session.get(k) is not v for k,v in required.items())
            or session.get('error') or session.get('pending_receipt_id')
            or session.get('car_id')!='bmw_z4_gt3' or session.get('track_id')!='monza'):
        raise ValueError('Paused offline idle BMW/Monza thermal collection start required')
    actual=list(temperatures)
    if (len(actual)!=4 or not all(type(t) in (int,float) and math.isfinite(t) for t in actual)
            or not all(abs(t-target)<=5 for t in actual)):
        raise ValueError('Current four-wheel temperature outside declared target±5C start band')
    return dict(schema='flybrain.thermal_teacher_start.v1',actor_input=False,
        initialization_directory=audited['directory'],initialization_evidence_sha256=audited['evidence_sha256'],
        generation=session['generation'],epoch=session['epoch'],target_initial_temperature_c=target,
        actual_start_temperature_c=actual,start_tolerance_c=5,
        initial_three_packet_tolerance_c=2,phase='condition_collection_preparation_lap',
        current_lap_excluded=True,official_valid_lap_verified=False,training_admissible=False)


def admit_teacher_resume(session,fields,*,gear,auto_shifter,thermal_admission=None):
    """Same physical readiness for warm defaults and explicit thermal trials."""
    if (gear!=2 or auto_shifter!=1 or abs(fields['speedKmh'])>=.1
            or fields['gas']>=.02 or fields['brake']<=.98):
        raise ValueError('Actual first gear/autoshifter/stationary full brake changed on resume')
    temperatures=list(fields['tyreCoreTemperature'])
    if len(temperatures)!=4 or not all(math.isfinite(t) for t in temperatures):
        raise ValueError('Four finite resumed native temperatures required')
    if thermal_admission is None:
        if not all(75<=t<=85 for t in temperatures):
            raise ValueError('Actual warm75..85C changed on resume without explicit thermal condition')
        return dict(condition='legacy_warm',actual_resume_temperature_c=temperatures)
    a=thermal_admission
    if (a.get('schema')!='flybrain.thermal_teacher_start.v1' or a.get('actor_input') is not False
            or a.get('generation')!=session.get('generation') or a.get('epoch')!=session.get('epoch')
            or session.get('tyres_initialization_epoch')!=session.get('epoch')
            or a.get('target_initial_temperature_c') not in (40,80,110)
            or a.get('current_lap_excluded') is not True or a.get('start_tolerance_c')!=5
            or not a.get('initialization_evidence_sha256')):
        raise ValueError('Exact previously admitted thermal condition must survive camera resume')
    if not all(abs(t-a['target_initial_temperature_c'])<=5 for t in temperatures):
        raise ValueError('Resumed actual temperature left admitted thermal start band')
    return dict(condition='audited_thermal',actual_resume_temperature_c=temperatures,
                initialization_directory=a['initialization_directory'],actor_input=False,current_lap_excluded=True)
