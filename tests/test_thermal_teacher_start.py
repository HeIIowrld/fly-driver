import pytest
from flybrain_drive.thermal_teacher_start import admit_thermal_teacher_start,admit_teacher_resume


def inputs(target=40):
    evidence=dict(target_c=target,condition_preparation_verified=True,
        consecutive_all_four_within_two_c=3,current_lap_excluded=True,
        generation='g',epoch=4,deployed_lua_sha256='sha',directory='recorded',evidence_sha256={'raw':'hash'})
    session=dict(generation='g',epoch=4,tyres_initialization_epoch=4,paused=True,
        is_online=False,in_menu=False,ai_controlled=False,camera_active=False,
        expert_probe_active=False,brake_settle_active=False,car_id='bmw_z4_gt3',track_id='monza')
    return evidence,session


@pytest.mark.parametrize('target',[40,80,110])
def test_actual_condition_is_bound_to_producer_and_retains_lap_exclusion(target):
    e,s=inputs(target)
    r=admit_thermal_teacher_start(e,s,[target-1]*4,source_sha256='sha')
    assert r['actual_start_temperature_c']==[target-1]*4
    assert r['actor_input'] is False and r['current_lap_excluded'] is True
    assert r['training_admissible'] is False


@pytest.mark.parametrize('key,value',[('generation','other'),('epoch',5),('tyres_initialization_epoch',3),
    ('paused',False),('camera_active',True),('ai_controlled',True),('in_menu',True),('error','bad')])
def test_wrong_or_active_session_does_not_admit_thermal_start(key,value):
    e,s=inputs();s[key]=value
    with pytest.raises(ValueError):admit_thermal_teacher_start(e,s,[40]*4,source_sha256='sha')


@pytest.mark.parametrize('temperatures',[[34]*4,[40,40,40,46],[40,40,40,float('nan')],[40]*3])
def test_target_cannot_replace_current_sensing(temperatures):
    e,s=inputs()
    with pytest.raises(ValueError):admit_thermal_teacher_start(e,s,temperatures,source_sha256='sha')


def test_old_source_and_unverified_initialization_are_rejected():
    e,s=inputs()
    with pytest.raises(ValueError):admit_thermal_teacher_start(e,s,[40]*4,source_sha256='other')
    e['consecutive_all_four_within_two_c']=2
    with pytest.raises(ValueError):admit_thermal_teacher_start(e,s,[40]*4,source_sha256='sha')


def test_resume_preserves_default_warm_guard_and_explicit_thermal_binding():
    e,s=inputs();a=admit_thermal_teacher_start(e,s,[40]*4,source_sha256='sha')
    fields=dict(speedKmh=0,gas=0,brake=1,tyreCoreTemperature=[40]*4)
    with pytest.raises(ValueError,match='warm75'):admit_teacher_resume(s,fields,gear=2,auto_shifter=1)
    r=admit_teacher_resume(s,fields,gear=2,auto_shifter=1,thermal_admission=a)
    assert r['condition']=='audited_thermal'
    with pytest.raises(ValueError):admit_teacher_resume(s,fields,gear=1,auto_shifter=1,thermal_admission=a)
    s['epoch']=5
    with pytest.raises(ValueError):admit_teacher_resume(s,fields,gear=2,auto_shifter=1,thermal_admission=a)
    fields['tyreCoreTemperature']=[80]*4
    assert admit_teacher_resume(s,fields,gear=2,auto_shifter=1)['condition']=='legacy_warm'
