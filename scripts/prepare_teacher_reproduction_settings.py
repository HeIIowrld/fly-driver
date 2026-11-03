"""Back up and prepare upstream teacher reproduction settings in a paused simulator.

No driving/lifecycle/UI requests. Changes take effect only after an explicit API
engine restart; raw physics and installed runtime extension are independently read.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import validate_heartbeat
from scripts.record_ac_wgc import physics_record
import time


def edit_ini(payload, changes):
    text = payload.decode('utf-8-sig')
    section = None
    seen = set()
    lines = []
    for line in text.splitlines(keepends=True):
        match = re.match(r'\s*\[([^\]]+)\]', line)
        if match:
            section = match[1]
        key = re.match(r'(\s*)([^=;]+?)(\s*)=(.*?)(\r?\n)?$', line)
        if key and (section, key[2]) in changes:
            pair = (section, key[2])
            if pair in seen:
                raise ValueError('Duplicate INI key')
            seen.add(pair)
            line = key[1]+key[2]+key[3]+'='+str(changes[pair])+(key[5] or '')
        lines.append(line)
    if seen != set(changes):
        raise ValueError('Required INI keys missing')
    return ''.join(lines).encode('utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis-config', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    axis = json.loads(Path(a.axis_config).read_text(encoding='utf8'))
    directory = Path(axis['profile_path']).parent
    files = {
        directory/'controls.ini': {('STEER', 'FF_GAIN'): 1},
        directory/'assists.ini': {('ASSISTS', k): v for k, v in dict(AUTO_BLIP=0,
            STABILITY_CONTROL=0, AUTO_SHIFTER=1, ABS=0, TRACTION_CONTROL=0,
            AUTO_CLUTCH=1, FUEL_RATE=0, TYRE_WEAR=0, TYRE_BLANKETS=1).items()},
        directory/'race.ini': {('RACE', 'PENALTIES'): 1,
            ('LAP_INVALIDATOR', 'ALLOWED_TYRES_OUT'): 2},
    }
    run = begin_run(a.output, dict(arguments=vars(a), seed=780102,
        reason='Match public teacher INSTALL settings; preserve original tyre setup values',
        controls_configuration=axis, student_optimizer_updates=0, ui_actions_sent=0))
    originals = {path:path.read_bytes() for path in files}
    if sha256(axis['profile_path']) != axis['profile_sha256']:
        raise ValueError('Reviewed controls profile changed')
    prepared = {path:edit_ini(originals[path], values) for path, values in files.items()}
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])

    def guard():
        hb, modified = read_published_json(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
        validate_heartbeat(hb, time.time()-modified, mutating=True)
        if (hb['paused'] is not True or hb['is_online'] or hb['camera_active']
                or hb['car_id'] != 'bmw_z4_gt3' or hb['track_id'] != 'monza'
                or sdk.GetVJDStatus(axis['device_id']) != 1):
            raise ValueError('Paused offline AI/camera-off released-controller session required')
        with NativePhysicsReader(timeout=.15) as reader:
            sample = reader.receive()
        if abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02 or sample.fields['brake'] <= .98:
            raise ValueError('Independent native stationary brake hold required')
        return hb, sample

    hb, sample = guard()
    write_json(run/'before-state.json', dict(session=hb, native=physics_record(sample)))
    changes = []
    for path in files:
        (run/(path.name+'.before')).write_bytes(originals[path])
        (run/(path.name+'.prepared')).write_bytes(prepared[path])
        changes.append(dict(path=str(path), before_sha256=hashlib.sha256(originals[path]).hexdigest(),
            prepared_sha256=hashlib.sha256(prepared[path]).hexdigest(),
            changes={section+'.'+key:value for (section,key),value in files[path].items()}))
    write_json(run/'settings-plan.json', changes)
    extension = Path('assets/external/ac-python33-socket-20261102/_socket.pyd')
    provenance = json.loads(extension.with_name('provenance.json').read_text())
    if sha256(extension) != provenance['sha256']:
        raise ValueError('Pinned extension hash differs')
    target = Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/python/flybrain_telemetry/runtime33/_socket.pyd')
    if target.exists() and sha256(target) != provenance['sha256']:
        raise ValueError('Never overwrite an unreviewed extension')
    if a.execute:
        guard()
        for path in files:
            if path.read_bytes() != originals[path]:
                raise ValueError('Configuration changed before commit')
        for path in files:
            temporary = path.with_name(path.name+'.flybrain-reproduction.partial')
            if temporary.exists():
                raise ValueError('Unreviewed settings partial exists')
            temporary.write_bytes(prepared[path])
            os.replace(temporary, path)
            if path.read_bytes() != prepared[path]:
                raise ValueError('Settings replacement readback differs')
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(extension.read_bytes())
        if sha256(target) != provenance['sha256']:
            raise ValueError('Extension deployment readback differs')
    axis['profile_sha256'] = hashlib.sha256(prepared[directory/'controls.ini']).hexdigest()
    axis['reproduction_settings_evidence'] = str(run/'settings-plan.json')
    write_json(run/'axis-config.json', axis)
    write_json(run/'result.json', dict(status='prepared_for_engine_restart' if a.execute else 'read_only',
        configuration_changes=changes, extension_target=str(target), extension_sha256=provenance['sha256'],
        runtime_activation_verified=False, source='upstream INSTALL at e1af2fa',
        actuator_messages_sent=0, ui_actions_sent=0, student_optimizer_updates=0))


if __name__ == '__main__':
    main()
