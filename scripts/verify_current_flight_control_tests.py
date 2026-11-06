"""Persist the relevant offline suite and exact tested source receipt."""
import argparse,json,subprocess,sys
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    files=['test_current_flight_control_contract','test_current_flight_control_ticker','test_current_flight_control_owner_protocol','test_current_flight_actuation','test_current_flight_watchdog_process','test_current_flight_dispatcher','test_current_flight_owner_protocol','test_current_flight_gc_owner','test_native_flight_body_ticker','test_ac_student_pilot','test_ac_vjoy','test_flight_control_worker_entry','test_current_flight_actuator_rpc','test_current_flight_native_feedback','test_completed_camera_diagnostic','test_paused_mmf_startup','test_learned_flight_pedals','test_flight_decision_mailbox','test_owned_pause_process','test_joint_balance_muscle_readout']
    tests=['tests/'+v+'.py' for v in files]
    out=begin_run(a.output,dict(tests={v:sha256(v) for v in tests},seed=None,vehicle_control_sent=False))
    with (out/'pytest.log').open('x',encoding='utf8') as log:
        child=subprocess.Popen([sys.executable,'-m','pytest','-q',*tests,'--junitxml='+str(out/'junit.xml')],cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT)
        code=child.wait()
    import xml.etree.ElementTree as ET
    suite=ET.parse(out/'junit.xml').getroot()[0]
    result=dict(status='passed' if code==0 else 'failed',exitcode=code,original_child_handle=True,tests=int(suite.attrib['tests']),failures=int(suite.attrib['failures']),errors=int(suite.attrib['errors']),skipped=int(suite.attrib.get('skipped',0)),pytest_log_sha256=sha256(out/'pytest.log'),junit_sha256=sha256(out/'junit.xml'))
    write_json(out/'result.json',result);print(json.dumps(result));raise SystemExit(code)

if __name__=='__main__':main()
