import base64
import hashlib
import struct

from flybrain_drive.online_body_shadow import OnlineBodyShadow

IDENTITY = dict(car_model='bmw_z4_gt3', shared_memory_version='1.7', ac_version='1.16.4')


def native(packet, receipt):
    raw = bytearray(420)
    struct.pack_into('<i', raw, 0, packet)
    struct.pack_into('<f', raw, 28, 36)
    struct.pack_into('<4f', raw, 72, 1000, 2000, 3000, 4000)
    raw = bytes(raw)
    return {'authority': {'prefix_base64': base64.b64encode(raw).decode(),
        'prefix_sha256': hashlib.sha256(raw).hexdigest(), 'packet_id': packet,
        'received_monotonic_seconds': receipt}, 'copy': {'second_copy_end': receipt}}


def test_online_snapshot_distinguishes_camera_receipt_and_decision(tmp_path):
    samples = [native(1, 2.), native(2, 2.01), native(3, 2.03)]
    shadow = OnlineBodyShadow(directory=tmp_path, request_id='x', generation='g',
        offset_lower=0., native_rows=samples, native_identity=IDENTITY, clock=lambda: 2.05)
    frame = {'index': 0, 'request_id': 'x', 'airflow_diagnostic': {'generation': 'g'},
        'views': [{'view_id': name, 'render_started_clock': clock} for name, clock in
                  zip(('front', 'left', 'right', 'rear'), (2.01, 2.02, 2.03, 2.04), strict=True)]}
    event = shadow.consume(frame, camera_receipt=2.045)
    assert event['camera_receipt'] == 2.045 and event['decision'] == 2.05
    assert event['native_rows_consumed'] == 3
    assert event['decision_body']['packet'] == 3
    assert [None if v is None else v['packet'] for v in event['view_body']] == [None, 2, 2, 3]
    assert event['view_body'][1]['receipt'] == 2.01


def test_neural_callback_receives_original_context_and_never_sends_controls(tmp_path):
    class Neural:
        def process(self, frame, event, context, directory):
            assert context.decision_body.packet_id == 3
            assert event['context_ready'] == 2.05
            return {'status': 'diagnostic', 'vehicle_control': False}

    shadow = OnlineBodyShadow(directory=tmp_path, request_id='x', generation='g',
        offset_lower=0., native_rows=[native(1, 2.), native(2, 2.01), native(3, 2.03)],
        native_identity=IDENTITY, clock=lambda: 2.05, neural_shadow=Neural())
    frame = {'index': 0, 'request_id': 'x', 'airflow_diagnostic': {'generation': 'g'},
        'views': [{'view_id': name, 'render_started_clock': clock} for name, clock in
                  zip(('front', 'left', 'right', 'rear'), (2.01, 2.02, 2.03, 2.04), strict=True)]}
    event = shadow.consume(frame, camera_receipt=2.045)
    assert event['neural'] == {'status': 'diagnostic', 'vehicle_control': False}


def test_visual_first_selects_new_current_body_without_changing_view_history(tmp_path):
    samples = [native(1, 2.), native(2, 2.01), native(3, 2.03)]

    class Neural:
        def process_late(self, frame, event, select, directory):
            samples.append(native(4, 2.047))
            context = select()
            assert context.decision_body.packet_id == 4
            assert [None if v is None else v.packet_id for v in context.visual_contexts] == [None, 2, 2, 3]
            return {'status': 'diagnostic', 'vehicle_control': False}

    shadow = OnlineBodyShadow(directory=tmp_path, request_id='x', generation='g',
        offset_lower=0., native_rows=samples, native_identity=IDENTITY,
        clock=lambda: 2.05, neural_shadow=Neural())
    frame = {'index': 0, 'request_id': 'x', 'airflow_diagnostic': {'generation': 'g'},
        'views': [{'view_id': name, 'render_started_clock': clock} for name, clock in
                  zip(('front', 'left', 'right', 'rear'), (2.01, 2.02, 2.03, 2.04), strict=True)]}
    event = shadow.consume(frame, camera_receipt=2.045)
    assert event['decision_body']['packet'] == 4
    assert event['native_rows_consumed'] == 4
    assert event['neural']['vehicle_control'] is False
