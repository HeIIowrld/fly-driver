"""Replace one pinned idle CSP camera source, preserving its lifecycle API.

Only a paused offline engine with released controls may be changed. Verify the
new script generation and capability; no desktop input or automatic UI fallback.
"""
import argparse
import json
import os
import time
from contextlib import closing
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader, repeated_snapshot
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.control_ac_session import (
    IMOLA_OWNED_COLLECTION_SOURCE_SHA256, IMOLA_OWNED_MMF_SOURCE_SHA256, IMOLA_OWNED_MOTION_SOURCE_SHA256,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis-config',required=True)
    p.add_argument('--expected-installed-sha256',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--candidate',default='integrations/ac_csp/candidates/FlyBrainCamera-source-body-witness-v1.lua')
    p.add_argument('--imola-owned-mmf',action='store_true')
    a=p.parse_args()
    source=Path(a.candidate)
    axis=json.loads(Path(a.axis_config).read_text(encoding='utf8'))
    run=begin_run(a.output,dict(arguments=vars(a),seed=780203,candidate_sha256=sha256(source)))
    result=dict(status='failed',ui_actions_sent=0,actuator_messages_sent=0,installed=False)
    try:
        capability='source_body_render_witness_v1'
        if a.imola_owned_mmf:
            capabilities={IMOLA_OWNED_MMF_SOURCE_SHA256:'imola_stationary_lifecycle_v1',
                          IMOLA_OWNED_MOTION_SOURCE_SHA256:'imola_owned_motion_v1',
                          IMOLA_OWNED_COLLECTION_SOURCE_SHA256:'imola_owned_collection_v1'}
            capability=capabilities.get(sha256(source))
            if capability is None:
                raise ValueError('Exact reviewed Imola owned MMF candidate required')
        api=WindowsACWindow();windows=api.discover()
        if len(windows)!=1:
            raise ValueError('Exactly one original engine required')
        window=windows[0]
        target=Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
        status_path=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        with closing(NativeTeacherStatusReader()) as ai, NativePhysicsReader(timeout=.15) as reader:
            def guard():
                refuse_leased_transport('camera source deployment',status_path=status_path)
                hb,mtime=read_published_json(status_path)
                sample=repeated_snapshot(reader.mapping);flag=ai.snapshot()
                if (not 0<=time.time()-mtime<=2 or not hb['paused'] or hb['is_online']
                        or hb['camera_active'] or hb['expert_probe_active'] or hb['ai_controlled']
                        or hb.get('error') or hb.get('pending_receipt_id')
                        or flag['builtin_ai_controlled'] or sdk.GetVJDStatus(axis['device_id'])!=1
                        or abs(sample.fields['speedKmh'])>=.1 or sample.fields['gas']>.02
                        or sample.fields['brake']<.98 or api.user.GetForegroundWindow()==window.hwnd
                        or int(hb['window_handle'].removesuffix('ULL'))!=window.hwnd
                        or a.imola_owned_mmf and (hb['track_id']!='imola' or hb['car_id']!='bmw_z4_gt3')
                        or sha256(axis['profile_path'])!=axis['profile_sha256']):
                    raise ValueError('Paused stationary offline original released profile required')
                return hb
            before=guard();write_json(run/'before.json',before)
            if sha256(target)!=a.expected_installed_sha256:
                raise ValueError('Installed camera source differs')
            if a.execute:
                (run/'installed-before.lua').write_bytes(target.read_bytes())
                (run/'candidate.lua').write_bytes(source.read_bytes())
                temporary=target.with_suffix('.lua.source-body-partial')
                if temporary.exists():
                    raise ValueError('Existing deployment partial requires inspection')
                if (guard()['generation']!=before['generation']
                        or sha256(target)!=a.expected_installed_sha256):
                    raise ValueError('Engine generation or installed source changed before deployment')
                temporary.write_bytes(source.read_bytes());os.replace(temporary,target)
                if sha256(target)!=sha256(source):
                    raise ValueError('Installed source readback differs')
                result.update(installed=True,installed_sha256=sha256(target))
                deadline=time.perf_counter()+20
                while time.perf_counter()<deadline:
                    hb,mtime=read_published_json(status_path)
                    if (0<=time.time()-mtime<=2 and hb['generation']!=before['generation']
                            and hb['capabilities'].get(capability) is True
                            and hb['track_id']==before['track_id'] and hb['car_id']==before['car_id']
                            and hb['paused'] and not hb['ai_controlled'] and not hb['camera_active']
                            and int(hb['window_handle'].removesuffix('ULL'))==window.hwnd):
                        write_json(run/'after.json',hb)
                        result['status']='deployed_new_generation_verified'
                        break
                    time.sleep(.05)
                else:
                    result['status']='installed_runtime_not_verified'
            else:
                result['status']='read_only_preflight'
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    write_json(run/'result.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
