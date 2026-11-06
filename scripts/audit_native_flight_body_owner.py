"""Audit isolated current-body receipts separately from camera historical receipts."""
import argparse,base64,hashlib,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher-motion',action='store_true');p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.shadow);r=read(root/'result.json');m=read(root/'manifest.json')['config'];owner=read(root/'owner-terminal.json')
    assert owner==r['owner_summary']
    numerics=owner.get('numerics')
    if m['arguments'].get('flush_body_denormal'):
        assert numerics['requested_flush_denormal'] is True and numerics['supported'] is True
        assert numerics['reset_to_nonflushing_before_thread_exit'] is True
    elif numerics is not None:
        assert numerics['requested_flush_denormal'] is False
    response=r['owner_terminal']['response']
    if response is None:
        assert r['status']=='failed' and r['owner_terminal']['poisoned'] is True
    else:assert owner==response['summary']
    assert sha256(root/'owner-native-copy.json')==owner['native_sha256']
    assert sha256(root/'owner-ticker-events.json')==owner['ticker_events_sha256']==r['ticker_events_sha256']
    assert sha256(root/'ticker-events.json')==r['ticker_events_sha256']
    owner_clean=r['owner_terminal']['exitcode']==0 and r['owner_terminal']['forced_termination'] is False and owner['sampler_stopped'] is True
    assert r['ticker_stopped'] is True
    ready=read(root/'worker-ready.json');assert ready['pid']==r['owner_terminal']['pid'] and '+cpu' in ready['torch_version']
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),shadow_result_sha256=sha256(root/'result.json'),
        source_domains=dict(current='owner-native-copy.json',historical='native-copy.json'),scope='Independent native first receipts per source; original camera-source audit plus actual isolated body-loop output deadlines'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    process=subprocess.run([sys.executable,'-m','scripts.audit_native_flight_live_shadow','--source-only','--shadow',str(root),'--output',str(run/'camera-source-audit')],capture_output=True,text=True)
    write_json(run/'camera-source-process.json',dict(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
    if process.returncode:raise ValueError('Historical camera-source audit failed')
    if r.get('single_authoritative_native_source'):assert sha256(root/'owner-native-copy.json')==sha256(root/'native-copy.json')
    quarantine_sequence_gap=False
    if 'retired_native_sha256' in owner:
        assert sha256(root/'owner-retired-native-copy.json')==owner['retired_native_sha256']
        retired=read(root/'owner-retired-native-copy.json');assert len(retired)==owner['retired_native_records']
        rejected=[]
        if 'rejected_native_sha256' in owner:
            assert sha256(root/'owner-rejected-native-copy.json')==owner['rejected_native_sha256']
            rejected=read(root/'owner-rejected-native-copy.json');assert len(rejected)==owner['rejected_native_records']
        rows=read(root/'owner-native-copy.json');quarantined=sorted(retired+rejected,key=lambda q:q['sequence'])
        expected=list(range(len(rows),len(rows)+len(quarantined)))
        quarantine_sequence_gap=[q['sequence'] for q in quarantined]!=expected
        if quarantine_sequence_gap:assert r['status']=='failed' and r['cleanup_verified'] is False
        for q in quarantined:
            assert q['actor_admitted'] is False
            authority=q['row']['authority'];payload=base64.b64decode(authority['prefix_base64'],validate=True)
            assert hashlib.sha256(payload).hexdigest()==authority['prefix_sha256']
    samples={}
    for row in read(root/'owner-native-copy.json'):
        authority=row['authority'];raw=base64.b64decode(authority['prefix_base64'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==authority['prefix_sha256']
        assert authority['received_monotonic_seconds']==row['copy']['second_copy_end']
        sample=decode_native_extension(raw[:PHYSICS_PREFIX.size],authority['received_monotonic_seconds'])
        assert sample.packet_id==authority['packet_id'];samples.setdefault(sample.packet_id,sample)
    scenes={e['capture_id']:e for e in read(root/'events.json') if e.get('status')=='scene_published'}
    outputs=[];previous=None;last=None
    for e in read(root/'owner-ticker-events.json'):
        if e['status']!='shadow_output':continue
        sample=samples[e['body_packet']];receipt=sample.received_monotonic_seconds;when=e['decision_time'];output=e['output_host_time'];scene=scenes[e['capture_id']]
        assert receipt==e['body_first_receipt'] and e['held_body_receipt']==previous and 0<=when-receipt<=.05
        if previous is not None:assert 0<=when-previous<=.05
        if last is not None:assert when>last
        assert scene['publication']==e['publication']<=when and scene['camera_received']==e['camera_received']
        assert scene['render_start_lowers']==e['render_start_lowers']
        body_age=output-receipt;view_age=max(output-t for t in e['render_start_lowers'])
        np.testing.assert_allclose([body_age,view_age],[e['output_body_age_seconds'],e['oldest_view_age_seconds']],atol=1e-12,rtol=0)
        assert 0<=body_age<=.05 and 0<=view_age<=.15 and 0<=e['inference_seconds']<=output-when
        value=np.asarray(e['prediction']);assert value.shape==(3,) and np.isfinite(value).all() and abs(value[0])<=1 and ((value[1:]>=0)&(value[1:]<=1)).all()
        outputs.append(e);previous=receipt;last=when
    assert len(outputs)==r['shadow_outputs']==owner['outputs']
    if m.get('inference_end_contract')=='stop_owner_before_teacher_wait_and_archives_v1':
        end=r['inference_end']
        assert end['started']<=end['finished'] and end['numerical_reset'] is False and end['student_vehicle_control_sent'] is False
        assert end['reason'] in ('teacher_process_terminal','fixed_six_second_shadow_budget','fixed_eight_second_shadow_budget','failed_or_interrupted_shadow')
        assert all(e['output_host_time']<=end['finished'] for e in outputs)
    motion=None
    if a.teacher_motion:
        from flybrain_drive.teacher_motion_evidence import audit_teacher_motion
        motion=audit_teacher_motion(root,r,m)
    mode_passed=motion['teacher_motion_contract_passed'] if motion is not None else r['status']=='completed_stationary_native_flight_shadow'
    passed=mode_passed and owner['ticker_error'] is None and not owner.get('owner_error') and r['cleanup_verified'] is True and owner_clean and not quarantine_sequence_gap and len(outputs)>0
    write_json(run/'result.json',dict(status='completed_independent_body_owner_shadow_audit',whole_shadow_passed=passed,outputs_verified=len(outputs),
        current_native_source='owner-native-copy.json',historical_native_source='native-copy.json',first_receipts_preserved=True,single_authoritative_native_source=r.get('single_authoritative_native_source',False),cleanup_verified=r['cleanup_verified'],quarantine_sequence_gap=quarantine_sequence_gap,
        same_camera_additional_body_outputs=sum(a['capture_id']==b['capture_id'] for a,b in zip(outputs,outputs[1:])),
        continuous_output_span_seconds=0 if not outputs else outputs[-1]['output_host_time']-outputs[0]['output_host_time'],
        body_age_max_seconds=None if not outputs else max(e['output_body_age_seconds'] for e in outputs),
        view_age_max_seconds=None if not outputs else max(e['oldest_view_age_seconds'] for e in outputs),
        inference_seconds_percentiles=[] if not outputs else np.percentile([e['inference_seconds'] for e in outputs],[50,95,99,100]).tolist(),
        ticker_error=owner['ticker_error'],numerical_execution_mode=numerics,teacher_motion=motion,owner_terminal_receipt_verified=True,owner_cleanly_stopped=owner_clean,owner_exitcode=r['owner_terminal']['exitcode'],owner_forced_termination=r['owner_terminal']['forced_termination'],actor_qualified=False,vehicle_control_sent=False,
        termination_reason='Original current-body and historical camera receipts, publication/continuous recurrence, deadlines and owner/native/camera termination receipts checked; any cleanup failure remains a failed run. '+('Finite external-teacher motion; student has no actuation, limit-driving or lap qualification.' if motion is not None else 'Finite stationary evidence, not moving or lap qualification.')))


if __name__=='__main__':main()
