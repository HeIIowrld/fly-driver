"""One cold launch only after engine absence, holding reviewed neutral/full brake.

Never an episode-reset fallback: all later session transitions use CSP API.
No window activation, keys, clicks, force termination, or launch retries.
"""
import argparse
import configparser
import json
import os
import subprocess
import time
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
from flybrain_drive.ac_vjoy import AxisBinding,VJoySession,load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.initial_thermal_contract import validate_thermal_native
from flybrain_drive.win_process_receipt import ProcessReceipt, engine_inventory
from scripts.control_ac_session import (
    IMOLA_OWNED_MOTION_SOURCE_SHA256, read_session_heartbeat, validate_heartbeat,
)
from scripts.restart_heldout_track import native_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis-config',required=True);p.add_argument('--race-ini',required=True)
    p.add_argument('--expected-lua-sha256',required=True)
    p.add_argument('--track',choices=('monza','magione','imola'),default='monza')
    p.add_argument('--execute',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();axis=json.loads(Path(a.axis_config).read_text(encoding='utf-8'))
    exe=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/acs.exe')
    lua=exe.parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    cfg=configparser.ConfigParser(interpolation=None);cfg.read(a.race_ini,encoding='utf-8-sig')
    required={('RACE','MODEL'):'bmw_z4_gt3',('RACE','TRACK'):a.track,('RACE','CARS'):'1',
        ('REMOTE','ACTIVE'):'0',('REPLAY','ACTIVE'):'0',('BENCHMARK','ACTIVE'):'0'}
    if any(cfg.get(s,k,fallback=None)!=v for (s,k),v in required.items()):
        raise ValueError('Pinned original offline one-car BMW configuration for the selected track required')
    if axis.get('track') != a.track or axis.get('car_model') != 'bmw_z4_gt3':
        raise ValueError('Axis configuration does not match selected track and car')
    capability='initial_tyre_temperature_v1'
    if a.track == 'imola':
        if (Path(a.race_ini).resolve()!=Path(axis['profile_path']).resolve().with_name('race.ini')
                or cfg.get('SESSION_0','TYPE',fallback=None)!='4'
                or cfg.get('RACE','CONFIG_TRACK',fallback=None)!=''
                or a.expected_lua_sha256!=IMOLA_OWNED_MOTION_SOURCE_SHA256):
            raise ValueError('Actual offline Imola hotlap profile and pinned motion producer required')
        capability='imola_owned_motion_v1'
    if sha256(lua)!=a.expected_lua_sha256 or sha256(axis['profile_path'])!=axis['profile_sha256']:
        raise ValueError('Reviewed deployment and actual control profile required')
    race_hash=sha256(a.race_ini)
    run=begin_run(a.output,dict(arguments=vars(a),seed=780512,program_sha256=sha256(__file__),
        executable_sha256=sha256(exe),race_ini_sha256=race_hash,axis_config_sha256=sha256(a.axis_config),
        profile_sha256=axis['profile_sha256'],cold_launch_only=True,
        requested_actuator=[0,0,1],ui_actions_sent=0,source_config_changed=False))
    result=dict(status='failed',ui_actions_sent=0,launch_attempts=0,actuator_messages_sent=0,
        process_termination_requested=False,episode_restart_attempts=0)
    controller=None;process=None;receipt=None
    try:
        inventory=engine_inventory();write_json(run/'engine-before.json',inventory)
        if inventory:raise ValueError('Existing engine; do not cold-launch or reset it')
        sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
        if sdk.GetVJDStatus(axis['device_id'])!=1:raise ValueError('Reviewed controller must be free')
        if not a.execute:
            result['status']='read_only'
        else:
            controller=VJoySession(sdk,axis['device_id'],[AxisBinding(**x) for x in axis['bindings']])
            write_json(run/'neutral-before-launch.json',controller.open())
            if engine_inventory() or sha256(a.race_ini)!=race_hash or sha256(lua)!=a.expected_lua_sha256:
                raise ValueError('Inputs/engine absence changed before launch')
            startup=subprocess.STARTUPINFO()
            startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
            result['launch_attempts']=1
            process=subprocess.Popen([str(exe)],cwd=exe.parent,startupinfo=startup,
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            receipt=ProcessReceipt(process.pid)
            write_json(run/'launched-process.json',receipt.snapshot())
            deadline=time.monotonic()+60;observations=[];api=WindowsACWindow()
            root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
            while time.monotonic()<deadline:
                state=receipt.snapshot()
                if not state['active']:raise RuntimeError('Owned engine exited; do not relaunch automatically')
                controller.submit(0,0,1)
                windows=api.discover()
                owned=[w for w in windows if w.pid==process.pid]
                if len(windows)==1 and len(owned)==1:
                    try:
                        hb,mtime,_=read_session_heartbeat(root,maximum_wait=.1)
                        validate_heartbeat(hb,time.time()-mtime,mutating=True)
                        if (int(hb['window_handle'].removesuffix('ULL'))!=owned[0].hwnd
                                or hb.get('track_id')!=a.track or hb.get('car_id')!='bmw_z4_gt3'
                                or hb.get('is_online') is not False
                                or hb.get('camera_active') is not False
                                or hb.get('capabilities',{}).get(capability) is not True):
                            raise ValueError('Waiting for the requested new owned producer')
                        if api.user.GetForegroundWindow()==owned[0].hwnd:
                            raise RuntimeError('Engine foreground despite hidden launch; no desktop fallback')
                        with NativeTeacherPhysicsReader(timeout=.15) as reader:
                            coupled=reader.receive()
                        validate_thermal_native(coupled)
                        identity=native_identity()
                        if identity['track']!=a.track or identity['car_model']!='bmw_z4_gt3':
                            raise ValueError('Waiting for matching native track and car')
                        write_json(run/'ready.json',dict(session=hb,native=coupled.authority_record(),
                            native_identity=identity))
                        result.update(status='completed_background_initial_engine',pid=process.pid,
                            session=hb,process=receipt.snapshot())
                        break
                    except (FileNotFoundError,PermissionError,ValueError,TimeoutError) as e:
                        observations.append(dict(host_clock=time.perf_counter(),kind=type(e).__name__,reason=str(e)))
                time.sleep(.05)
            write_json(run/'startup-observations.json',observations)
            if result['status']=='failed':raise TimeoutError('Owned engine live but ready state unverified; do not relaunch')
    except Exception as e:
        result.update(error=type(e).__name__,reason=str(e))
    finally:
        if controller and controller.owned:write_json(run/'neutral-release.json',controller.close())
        if controller:result['actuator_messages_sent']=controller.submit_attempts
        if receipt:
            result['final_process']=receipt.snapshot();receipt.close()
        result['termination_reason']='One reviewed cold-launch observation completed; later transitions require CSP' if result['status']!='failed' else 'Explicit launch/preflight/readiness failure; no UI/relaunch/termination fallback'
        write_json(run/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
