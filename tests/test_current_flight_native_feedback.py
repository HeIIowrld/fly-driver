import base64,hashlib,json,struct
import pytest
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,decode_physics_prefix
from flybrain_drive.current_flight_native_control_feedback import calibration_native_steering


def fixture(tmp_path):
    rows=[]
    for stage,request in enumerate((0.,.6,-.6,0.)):
        for index in range(8):
            packet=len(rows)+1;raw=bytearray(PHYSICS_PREFIX.size)
            struct.pack_into('<i',raw,0,packet);struct.pack_into('<f',raw,8,1.)
            struct.pack_into('<f',raw,24,request*1.40625);raw=bytes(raw)
            sample=decode_physics_prefix(raw,1.+packet*.05)
            rows.append(dict(stage=stage,index=index,native=dict(packet_id=packet,fields=dict(sample.fields),prefix_base64=base64.b64encode(raw).decode(),prefix_sha256=hashlib.sha256(raw).hexdigest(),received_monotonic_seconds=sample.received_monotonic_seconds),
                request=dict(requested_control=dict(steering=request,throttle=0.,brake=1.)),sdk_steer_raw=request*450,
                native_ai_before=dict(builtin_ai_controlled=False),native_ai_after=dict(builtin_ai_controlled=False),was_foreground_at_record=False))
    result=dict(status='completed',samples=32,request_amplitude=.6,sdk_steering_stage_medians=[0.,270.,-270.,0.],sdk_raw_per_normalized_request=450.,profile_sha256='a'*64,producer_session_id='fixture')
    (tmp_path/'result.json').write_text(json.dumps(result))
    return rows


def save(tmp_path,rows):
    (tmp_path/'samples.jsonl').write_text(chr(10).join(json.dumps(v) for v in rows))


def test_sdk_and_native_scales_are_distinct_and_neutral_tolerance_uses_native_units(tmp_path):
    rows=fixture(tmp_path);save(tmp_path,rows);value=calibration_native_steering(tmp_path)
    assert value['native_raw_per_sdk_normalized_request']==1.40625
    assert value['sdk_raw_per_sdk_normalized_request']==450.
    assert value['neutral_tolerance_native']==pytest.approx(.0015625)
    assert not value['independent_physical_units_verified'] and not value['whole_domain_linearity_verified']

@pytest.mark.parametrize('defect',['hash','fields','request','authority','count'])
def test_native_feedback_contract_requires_original_calibration_evidence(tmp_path,defect):
    rows=fixture(tmp_path)
    if defect=='hash':rows[5]['native']['prefix_sha256']='b'*64
    if defect=='fields':rows[5]['native']['fields']['steerAngle']=1.
    if defect=='request':rows[12]['request']['requested_control']['steering']=.4
    if defect=='authority':rows[12]['native_ai_after']['builtin_ai_controlled']=True
    if defect=='count':rows.pop()
    save(tmp_path,rows)
    with pytest.raises(ValueError):calibration_native_steering(tmp_path)
