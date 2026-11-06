"""Audit finite live shadow outputs, source pixels/native clocks and retained rejections."""
import argparse,base64,hashlib,io,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.camera_body_alignment import source_render_witness
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-only',action='store_true');p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.shadow);r=read(root/'result.json');m=read(root/'manifest.json')['config']
    assert sha256(root/'events.json')==r['events_sha256'] and sha256(root/'native-copy.json')==r['native_sha256']
    assert r['actuator_messages_sent']==r['ui_actions_sent']==r['optimizer_updates']==0
    native=read(root/'native-copy.json');samples={}
    for row in native:
        authority=row['authority'];raw=base64.b64decode(authority['prefix_base64'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==authority['prefix_sha256']
        assert row['copy']['second_copy_end']==authority['received_monotonic_seconds']
        sample=decode_native_extension(raw[:PHYSICS_PREFIX.size],authority['received_monotonic_seconds'])
        assert sample.packet_id==authority['packet_id']
        samples.setdefault(sample.packet_id,(sample,row))
    events=read(root/'events.json');outputs=[];contexts=[];stale_contexts=[];previous=None
    with CameraArchive(root/'vision') as archive:
        assert archive.bundle_sha256==r['camera_archive']['bundle_sha256']
        stream=[json.loads(x) for x in archive.read('stream.jsonl').decode('utf8').splitlines()];clock=archive.json('clock-initial.json')
        for event in events:
            if 'decision_body' not in event:continue
            i=event['capture_id'];receipt=stream[i];frame=archive.json(Path(receipt['file']).name)
            assert event['camera_received']==receipt['received']
            byname={v['view_id']:v for v in frame['views']};gaps=[]
            for j,view in enumerate(receipt['views']):
                name=view['view_id'];source=byname[name];source_render_witness(source)
                rgb=np.asarray(Image.open(io.BytesIO(archive.read(source['file']))).convert('RGB'))
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==view['rgb_sha256']
                assert event['render_start_lowers'][j]==view['render_started_lower']
                history=event['view_body'][j];sample,row=samples[history['packet']]
                assert history['receipt']==sample.received_monotonic_seconds
                upper=source['render_started_clock']+clock['offset_upper_seconds']+clock.get('host_clock_quantization_seconds',0.)
                gap=upper-row['copy']['first_copy_start'];assert 0<=gap<=.025;gaps.append(gap)
            body=event['decision_body'];sample,row=samples[body['packet']]
            assert body['receipt']==sample.received_monotonic_seconds
            decision=event['decision'];input_age=decision-body['receipt'];assert input_age>=0
            if input_age>.05:
                assert a.source_only and r['status']!='completed_stationary_native_flight_shadow' and not any(body['valid'])
                stale_contexts.append(dict(capture_id=i,body_age_seconds=input_age,all_original_valid_masks_false=True))
            context=dict(capture_id=i,current_body_age_seconds=input_age,
                held_previous_body_age_seconds=None if previous is None else decision-previous,
                original_source_render_gap_seconds=gaps)
            contexts.append(context)
            if event['status']=='shadow_output':
                assert event['body_packet']==body['packet'] and event['body_first_receipt']==body['receipt']
                output=event['output_host_time'];body_age=output-body['receipt'];view_age=max(output-t for t in event['render_start_lowers'])
                np.testing.assert_allclose([body_age,view_age],[event['output_body_age_seconds'],event['oldest_view_age_seconds']],atol=1e-12,rtol=0)
                assert body_age<=.05 and view_age<=.15 and 0<=event['inference_seconds']<=output-decision
                np.testing.assert_allclose(gaps,event['source_render_gap_seconds'],atol=1e-12,rtol=0)
                prediction=np.asarray(event['prediction']);assert prediction.shape==(3,) and np.isfinite(prediction).all()
                assert abs(prediction[0])<=1 and ((prediction[1:]>=0)&(prediction[1:]<=1)).all()
                outputs.append(dict(capture_id=i,body_age_seconds=body_age,oldest_view_age_seconds=view_age,inference_seconds=event['inference_seconds']))
                previous=body['receipt']
    close=r['camera_close'];assert close['status']=='stopped' and close['worker_alive'] is False
    native_stopped=r.get('native_sampler_stopped') is True
    if not a.source_only:assert native_stopped
    assert close['producer_terminal']['status'] in ['completed','stopped']
    assert read(root/'final-pause/result.json')['status']=='completed' and r['pause_returncode']==0
    held_failure=r.get('reason')=='Held body receipt stale at decision'
    if held_failure:
        assert contexts[-1]['held_previous_body_age_seconds']>.05 and contexts[-1]['current_body_age_seconds']<.05
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),shadow_result_sha256=sha256(root/'result.json'),manifest_sha256=sha256(root/'manifest.json')))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_live_shadow_evidence_audit',whole_shadow_passed=r['status']=='completed_stationary_native_flight_shadow' and not a.source_only,source_only=a.source_only,
        shadow_outputs_verified=len(outputs),outputs=outputs,contexts=contexts,stale_camera_contexts=stale_contexts,held_body_staleness_verified=held_failure,
        original_failure_reason=r.get('reason'),camera_and_sampler_stopped=native_stopped,final_pause_verified=True,
        actor_qualified=False,vehicle_control_sent=False,physical_exposure_verified=False,
        termination_reason='Original RGB/source witness, first native receipt, output ages and clean termination independently checked; failed shadow remains failed'))


if __name__=='__main__':main()
