"""Independent native body-rate output, original-source and worker termination audit."""
import argparse,subprocess,sys,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.shadow);r=read(root/'result.json');m=read(root/'manifest.json')['config']
    assert sha256(root/'ticker-events.json')==r['ticker_events_sha256']
    assert sha256(root/'events.json')==r['events_sha256'] and sha256(root/'native-copy.json')==r['native_sha256']
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),shadow_result_sha256=sha256(root/'result.json'),ticker_source_sha256=m['ticker_sha256']))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    completed=subprocess.run([sys.executable,'-m','scripts.audit_native_flight_live_shadow','--source-only','--shadow',str(root),'--output',str(run/'source-audit')],capture_output=True,text=True)
    write_json(run/'source-audit-process.json',dict(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr))
    if completed.returncode:raise ValueError('Independent original-camera/native source audit failed')
    native=read(root/'native-copy.json');first={}
    for row in native:first.setdefault(row['authority']['packet_id'],row)
    scenes={e['capture_id']:e for e in read(root/'events.json') if e.get('status')=='scene_published'}
    events=read(root/'ticker-events.json');outputs=[];last=None;previous=None
    for event in events:
        if event['status']!='shadow_output':continue
        scene=scenes[event['capture_id']];decision=event['decision_time'];receipt=event['body_first_receipt'];output=event['output_host_time']
        row=first[event['body_packet']]
        assert receipt==row['authority']['received_monotonic_seconds'] and 0<=decision-receipt<=.05
        assert scene['publication']==event['publication']<=decision
        assert scene['render_start_lowers']==event['render_start_lowers']
        assert scene['camera_received']==event['camera_received']
        assert event['held_body_receipt']==previous
        if previous is not None:assert 0<=decision-previous<=.05
        if last is not None:assert decision>last
        body_age=output-receipt;view_age=max(output-t for t in event['render_start_lowers'])
        np.testing.assert_allclose([body_age,view_age],[event['output_body_age_seconds'],event['oldest_view_age_seconds']],atol=1e-12,rtol=0)
        assert 0<=body_age<=.05 and 0<=view_age<=.15 and 0<=event['inference_seconds']<=output-decision
        value=np.asarray(event['prediction']);assert value.shape==(3,) and np.isfinite(value).all() and abs(value[0])<=1 and ((value[1:]>=0)&(value[1:]<=1)).all()
        outputs.append(event);previous=receipt;last=decision
    assert len(outputs)==r['shadow_outputs']
    worker=r.get('worker_terminal')
    if worker is not None:
        assert worker['exitcode']==0 and worker['forced_termination'] is False
        if worker['status']=='stopped':assert len(outputs)<=worker['sequence']<=len(outputs)+sum(e['status']=='rejected' for e in events)
    assert r['ticker_stopped'] is True
    passed=r['status']=='completed_stationary_native_flight_shadow' and r.get('ticker_error') is None and r['cleanup_verified'] is True and len(outputs)>0
    result=dict(status='completed_independent_native_multirate_shadow_audit',whole_shadow_passed=passed,outputs_verified=len(outputs),
        same_camera_additional_body_outputs=sum(a['capture_id']==b['capture_id'] for a,b in zip(outputs,outputs[1:])),
        continuous_output_span_seconds=0 if not outputs else outputs[-1]['output_host_time']-outputs[0]['output_host_time'],
        output_body_age_max_seconds=None if not outputs else max(e['output_body_age_seconds'] for e in outputs),
        oldest_view_age_max_seconds=None if not outputs else max(e['oldest_view_age_seconds'] for e in outputs),
        worker_inference_seconds_percentiles=[] if not outputs else np.percentile([e['inference_seconds'] for e in outputs],[50,95,99,100]).tolist(),
        ticker_error=r.get('ticker_error'),cleanup_verified=r['cleanup_verified'],native_sampler_error=r.get('native_sampler_close_error'),worker_terminal=worker,actor_qualified=False,vehicle_control_sent=False,
        termination_reason='Original source audit plus independent first native/held receipts, body/view output deadlines, same-camera continued recurrence and worker/camera termination checked; no moving/lap/physiology claim')
    write_json(run/'result.json',result)


if __name__=='__main__':main()
