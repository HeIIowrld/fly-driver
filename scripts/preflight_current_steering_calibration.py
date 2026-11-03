"""Read-only current steering calibration readiness; never acquires or commands vJoy."""
import argparse,json,time
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.calibrate_ac_sdk_steering import validate_stationary_calibration_state

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
cfg=json.loads(Path('configs/ac-axis-probe.json').read_text(encoding='utf8'))
prior=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-after-restart-steering-calibration')
old=json.loads((prior/'result.json').read_text(encoding='utf8')); current_hash=sha256(cfg['profile_path'])
run=begin_run(a.output,dict(program_sha256=sha256(__file__),config_sha256=sha256('configs/ac-axis-probe.json'),
 prior_calibration_sha256=sha256(prior/'result.json'),read_only=True,actuator_messages_sent=0))
status,mtime=read_published_json(Path(__import__('os').environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
native=WindowsACWindow();windows=native.discover();devices=inspect_vjoy_controllers();sdk=load_vjoy_sdk(cfg['dll_path'],cfg['dll_sha256'])
with NativePhysicsReader(timeout=.5) as r: body=r.receive()
ai=NativeTeacherStatusReader()
try: flag=ai.snapshot()
finally: ai.close()
fields=dict(body.fields); errors=[]
try: validate_stationary_calibration_state(status,time.time()-mtime,flag,fields,status['generation'])
except ValueError as e: errors.append(str(e))
checks={'one_engine':len(windows)==1,'background':len(windows)==1 and native.user.GetForegroundWindow()!=windows[0].hwnd,
 'vjoy_free':sdk.GetVJDStatus(cfg['device_id'])==1,
 'device_identity':len(devices['vjoy_devices'])==1 and devices['candidate_ac_joy_index']==0 and devices['vjoy_devices'][0]['instance_guid']==cfg['instance_guid'],
 'pinned_profile_match':current_hash==cfg['profile_sha256'],
 'prior_completed_profile_match':old.get('status')=='completed' and old.get('profile_sha256')==current_hash,
 'stationary_full_brake_guard':not errors}
result={'status':'completed_read_only_preflight','generation':status['generation'],'packet_id':body.packet_id,
 'body':{k:fields[k] for k in ('speedKmh','gas','brake')},'native_ai':flag['builtin_ai_controlled'],
 'current_profile_sha256':current_hash,'checks':checks,'guard_errors':errors,
 'calibration_ready':all(checks.values()),'moving_steering_calibrated':False,'actuator_messages_sent':0,'ui_actions_sent':0}
write_json(run/'result.json',result);print(json.dumps(result))
