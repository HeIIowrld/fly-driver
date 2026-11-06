"""Finite explicit file-transport SDK/native temperature observation, not actor data."""
import argparse,json,os,shutil,time,uuid
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--expected-lua-sha256',required=True);a=p.parse_args()
    local=Path(os.environ['LOCALAPPDATA'])/'flybrain';camera=local/'csp-camera';s,m,_=read_session_heartbeat(local/'csp-session/status.json');validate_heartbeat(s,time.time()-m,mutating=False)
    api=WindowsACWindow();windows=api.discover();assert len(windows)==1;w=windows[0]
    assert sha256(Path(w.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')==a.expected_lua_sha256
    if (s['camera_active'] or s['ai_controlled'] or s['is_online'] or abs(s['speed_kmh'])>=.1 or s['gas']>=.02 or s['brake']<=.98 or api.user.GetForegroundWindow()==w.hwnd):raise ValueError('Idle stationary offline background observation required')
    previous= json.loads((camera/'request.json').read_text(encoding='utf8'))
    terminal= json.loads((camera/previous['id']/'result.json').read_text(encoding='utf8'))
    assert terminal['status'] in ('failed','stopped','completed')
    rid='thermal-read-'+uuid.uuid4().hex;request=dict(id=rid,frames=3,period_seconds=.1,deadline_seconds=5,exposure=.35,image_format='png',expected_generation=s['generation'])
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),arguments=vars(a),initial_session=s,request=request,seed=None,actuator_messages_sent=0,ui_actions_sent=0,actor_admitted=False,transport='Explicit PNG file diagnostic, not live actor fallback'))
    result=dict(status='failed',actor_admitted=False,actuator_messages_sent=0);reader=NativeTeacherPhysicsReader(timeout=.2,clock=time.perf_counter)
    try:
        write_json(run/'native-before.json',repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter).authority_record())
        write_json(run/'request.json',request);write_json(camera/'request.json',request);folder=camera/rid;deadline=time.perf_counter()+8
        while not (folder/'result.json').exists():
            if time.perf_counter()>deadline:raise TimeoutError('Original camera request terminal unknown; never restart')
            time.sleep(.02)
        result['producer_terminal']=json.loads((folder/'result.json').read_text(encoding='utf8'));assert result['producer_terminal']['status']=='completed'
        observations=[]
        for i in range(3):
            f=folder/f'frame-{i:04d}.json';shutil.copy2(f,run/f.name);frame=json.loads(f.read_text(encoding='utf8'))
            for view in frame['views']:
                q=view['source_body_before_render']
                observations.append(dict(frame=i,view=view['view_id'],core_temperature_c=[v['core_temperature_c'] for v in q['wheels']],physics_timestamp_ms=q['physics_timestamp_ms']))
        write_json(run/'sdk-observations.json',observations)
        result.update(status='completed_sdk_native_thermal_snapshot',sdk_observations_sha256=sha256(run/'sdk-observations.json'),source_directory=str(folder),source_retained=True)
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:
        write_json(run/'native-after.json',repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter).authority_record());reader.close();write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
