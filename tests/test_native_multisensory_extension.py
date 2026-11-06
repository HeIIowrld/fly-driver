import hashlib,json,struct
import numpy as np
import pytest
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder,decode_native_extension,EXTENSION_NAMES


def raw(packet=1,heading=0.,pitch=0.,temp=80.,acc=(.2,.3,.4),gas=0.):
 data=bytearray(PHYSICS_PREFIX.size);struct.pack_into('<i',data,0,packet);struct.pack_into('<f',data,4,gas);struct.pack_into('<f',data,28,72.)
 struct.pack_into('<3f',data,44,*acc);struct.pack_into('<4f',data,56,.1,.2,.3,.4);struct.pack_into('<4f',data,72,1000,2000,3000,4000);struct.pack_into('<4f',data,88,25,25,25,25);struct.pack_into('<4f',data,152,*([temp]*4));struct.pack_into('<3f',data,208,heading,pitch,0.)
 return bytes(data)

@pytest.fixture
def encoder(tmp_path):
 audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(status='completed_qualified_native_multisensory_audit',candidate_axes_passed_all_episodes=[True,True,False])))
 sdk=tmp_path/'sdk.lua';sdk.write_text('source fixture, not installed runtime')
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 contract=dict(schema='flybrain.native_multisensory_extension.contract.v1',feature_names=list(EXTENSION_NAMES),assessed_acceleration_axes=[True,True,False],native_acceleration_indices=[2,0,1],native_identity={'car_model':'test'},audit_result=str(audit),audit_result_sha256=digest(audit),installed_sdk_file=str(sdk),installed_sdk_sha256=digest(sdk),maximum_host_age_seconds=.05,acceleration_assessed_envelope=dict(pitch_native_rad=[-.1,.1],roll_native_rad=[-.1,.1],speed_m_s=[0,80]),thermal_training_support_c=dict(minimum=[60]*4,maximum=[100]*4))
 path=tmp_path/'contract.json';path.write_text(json.dumps(contract));return NativeMultisensoryExtensionEncoder(path,native_identity={'car_model':'test'})


def test_assessed_axes_and_tyre_units_are_distinct_from_unregistered_vertical(encoder):
 out=encoder.encode(decode_native_extension(raw(),10),episode_id='a',decision_monotonic_seconds=10.01)
 np.testing.assert_allclose(out.values[:3],[.4*9.80665,.2*9.80665,0],rtol=1e-6)
 assert out.valid[:2].all() and not out.valid[2] and out.valid[3:].all()
 np.testing.assert_array_equal(out.values[11:15],[80]*4)
 assert not out.actor_qualified and not out.physiological_registration_verified
 assert not out.values.flags.writeable and not out.valid.flags.writeable


def test_repeated_packet_cannot_refresh_receipt_or_stale_sensing(encoder):
 encoder.encode(decode_native_extension(raw(),10),episode_id='a',decision_monotonic_seconds=10.01)
 out=encoder.encode(decode_native_extension(raw(),10.06),episode_id='a',decision_monotonic_seconds=10.07)
 assert out.original_received_monotonic_seconds==10
 assert not out.valid.any() and not out.values.any()
 with pytest.raises(ValueError,match='different bytes'):encoder.encode(decode_native_extension(raw(temp=81),10.08),episode_id='a',decision_monotonic_seconds=10.08)


def test_outside_acceleration_envelope_is_not_silently_qualified(encoder):
 out=encoder.encode(decode_native_extension(raw(pitch=.3,temp=120),10),episode_id='a',decision_monotonic_seconds=10.01)
 assert out.acceleration_out_of_assessed_scope and not out.valid[:3].any()
 assert out.valid[11:15].all() and all(out.thermal_out_of_training_support)
 np.testing.assert_array_equal(out.values[11:15],[120]*4)


def test_heading_and_teacher_actuation_do_not_enter_the_values(encoder):
 a=encoder.encode(decode_native_extension(raw(heading=-2,gas=0),10),episode_id='a',decision_monotonic_seconds=10.01)
 b=encoder.encode(decode_native_extension(raw(heading=2,gas=1),11),episode_id='b',decision_monotonic_seconds=11.01)
 np.testing.assert_array_equal(a.values,b.values);np.testing.assert_array_equal(a.valid,b.valid)
 assert not any(any(x in name for x in ['heading','position','command']) for name in a.feature_names)


def test_unregistered_up_signal_cannot_leak_and_future_samples_fail(encoder):
 a=encoder.encode(decode_native_extension(raw(acc=(.2,200.,.4)),10),episode_id='a',decision_monotonic_seconds=10.01)
 assert a.values[2]==0 and not a.valid[2]
 with pytest.raises(ValueError,match='after decision'):encoder.encode(decode_native_extension(raw(packet=2),11),episode_id='a',decision_monotonic_seconds=10.02)

def test_v2_requires_matched_bandwidth_evidence_before_exposing_vertical(encoder,tmp_path):
    import hashlib,json
    matched=tmp_path/'matched';matched.mkdir();audit=matched/'result.json';manifest=matched/'manifest.json'
    audit.write_text(json.dumps(dict(status='completed_bandwidth_matched_acceleration_diagnostic',episodes=[dict(fast_frames=dict(n=200,candidate_axis_threshold_passed=[True,True,True])) for i in range(3)])))
    manifest.write_text(json.dumps(dict(config=dict(original_audit_sha256=encoder.contract['audit_result_sha256']))))
    contract=dict(encoder.contract,schema='flybrain.native_multisensory_extension.contract.v2',assessed_acceleration_axes=[True,True,True],bandwidth_assessment_result=str(audit),bandwidth_assessment_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
    file=tmp_path/'v2.json';file.write_text(json.dumps(contract))
    v2=NativeMultisensoryExtensionEncoder(file,native_identity={'car_model':'test'})
    out=v2.encode(decode_native_extension(raw(),10),episode_id='a',decision_monotonic_seconds=10.01)
    assert out.valid[:3].all() and out.schema.endswith('.v2')
    np.testing.assert_allclose(out.values[2],.3*9.80665,rtol=1e-6)
    changed=json.loads(audit.read_text());changed['episodes'][0]['fast_frames']['candidate_axis_threshold_passed'][2]=False;audit.write_text(json.dumps(changed));contract['bandwidth_assessment_sha256']=hashlib.sha256(audit.read_bytes()).hexdigest();file.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='does not support'):NativeMultisensoryExtensionEncoder(file,native_identity={'car_model':'test'})
