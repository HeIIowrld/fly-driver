"""Read-only native owner startup/status/termination; no camera or control APIs."""
import os,time,traceback
from pathlib import Path
from flybrain_drive.current_flight_owner import CurrentFlightOwnerProcess
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat


def main():
    checkpoint=Path('runs/fixture-adjoint-learning-20261104/input_adjoint-fit-01/ports.pt')
    heartbeat_path=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    before,modified,retries=read_session_heartbeat(heartbeat_path)
    with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
    out=begin_run('runs/current-flight-owner-20261104/native-probe-01',dict(checkpoint_sha256=sha256(checkpoint),program_sha256=sha256(__file__),heartbeat=before,heartbeat_file_modified=modified,heartbeat_read_retries=retries,native_identity=identity,vehicle_control_sent=False,simulator_mutation_api_calls=0))
    owner=None;result=dict(status='failed',vehicle_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0)
    try:
        owner=CurrentFlightOwnerProcess(checkpoint=checkpoint,checkpoint_sha256=sha256(checkpoint),directory=out,python_executable='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe',library_directory='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages')
        owner.begin_live('current-native-readonly-20261104-01',before['generation'],identity)
        statuses=[]
        for _ in range(5):
            time.sleep(.02);value=owner.status();statuses.append(value)
            if value['error'] is not None:raise RuntimeError(value['error'])
        after,_,_=read_session_heartbeat(heartbeat_path)
        if before['generation']!=after['generation']:raise ValueError('Native session changed during probe')
        result.update(status='completed_readonly_native_startup',statuses=statuses,heartbeat_after=after,actor_qualified=False,live_camera_join_verified=False,actual_driving_improvement_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if owner is not None:result['owner_terminal']=owner.stop()
        write_json(out/'result.json',result)

if __name__=='__main__':main()
