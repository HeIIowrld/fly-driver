"""Describe actual thermal/inertial coverage of qualified teacher requests.

This is a teacher-source diagnostic sidecar, NOT the actor-tick native19 join.
Raw source bytes, declared units and episode clocks remain separately auditable.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,PHYSICS_LAYOUT
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',nargs='+',required=True);p.add_argument('--qualification',nargs='+',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    if len(a.recording)!=len(a.qualification):raise ValueError('One qualification per episode required')
    run=begin_run(a.output,dict(arguments=vars(a),seed=780513,program_sha256=sha256(__file__),
        scope=__doc__,actor_input=False,optimizer_updates=0,vehicle_control_sent=False))
    reports=[];sources=[]
    for i,(recording,qualification) in enumerate(zip(a.recording,a.qualification,strict=True)):
        root=Path(recording);q=Path(qualification);qr=read(q/'result.json');qm=read(q/'manifest.json')['config']
        if qr['status']!='qualified_bounded_camera_body_alignment' or not qr['training_admissible']:
            raise ValueError('Complete independent camera/body qualification required')
        for name,key in [('result.json','source_result_sha256'),('camera/result.json','camera_result_sha256'),('teacher/result.json','teacher_result_sha256')]:
            if sha256(root/name)!=qm[key]:raise ValueError('Qualified collection changed')
        result=read(root/'teacher/result.json');admission=read(root/'teacher/thermal-start-admission.json')
        if result['status']!='curriculum_completed' or admission['actor_input'] is not False:
            raise ValueError('Completed thermal teacher-only episode required')
        events=root/'teacher/events.jsonl'
        if sha256(events)!=result['events_sha256']:raise ValueError('Teacher event source changed')
        proof=read(q/'lap-proof/result.json')
        if sha256(q/'lap-proof/result.json')!=qr['lap_proof_sha256']:raise ValueError('Lap proof changed')
        begin,end=proof['crossings'][0]['host_time'],proof['crossings'][-1]['host_time']
        last=None;values=[];times=[];packets=[];receipts=[];hashes=[];full=[];source_times=[]
        with events.open(encoding='utf-8') as stream:
            for line in stream:
                e=json.loads(line)
                if e['kind']=='raw_pair':last=e
                elif e['kind']=='teacher_request':
                    if last is None:raise ValueError('No preceding original source input for teacher request')
                    n=last['native'];raw=base64.b64decode(n['prefix_base64'],validate=True)
                    if len(raw)!=PHYSICS_PREFIX.size or hashlib.sha256(raw).hexdigest()!=n['prefix_sha256']:
                        raise ValueError('Original native prefix/hash changed')
                    decoded=iter(PHYSICS_PREFIX.unpack(raw))
                    f={name:[next(decoded) for _ in range(width)] for name,_,width in PHYSICS_LAYOUT}
                    if f['packetId'][0]!=n['packet_id'] or last['host_time']>e['host_time']:
                        raise ValueError('Request/source order differs')
                    vector=f['accG']+f['wheelSlip']+f['wheelLoad']+f['tyreCoreTemperature']+f['wheelsPressure']
                    if not np.isfinite(vector).all():raise ValueError('Nonfinite native source sensations')
                    values.append(vector);times.append(e['host_time']);packets.append(n['packet_id'])
                    receipts.append(n['received_monotonic_seconds']);hashes.append(n['prefix_sha256'])
                    source_times.append(last['host_time']);full.append(begin<=e['host_time']<=end)
        x=np.asarray(values);t=np.asarray(times);m=np.asarray(full,dtype=bool)
        if len(x)!=result['teacher_requests'] or not m.any() or np.any(np.diff(t)<=0):
            raise ValueError('Complete advancing teacher requests and full-lap mask required')
        temperature=x[:,11:15];early=t<=t[0]+5
        def coverage(mask):
            selected=x[mask]
            return dict(requests=int(mask.sum()),core_temperature_c_minimum=selected[:,11:15].min(0).tolist(),
                core_temperature_c_maximum=selected[:,11:15].max(0).tolist(),
                acceleration_native_g_minimum=selected[:,:3].min(0).tolist(),
                acceleration_native_g_maximum=selected[:,:3].max(0).tolist(),
                wheel_slip_native_minimum=selected[:,3:7].min(0).tolist(),
                wheel_slip_native_maximum=selected[:,3:7].max(0).tolist(),
                wheel_load_native_minimum=selected[:,7:11].min(0).tolist(),
                wheel_load_native_maximum=selected[:,7:11].max(0).tolist())
        target=run/f'teacher-source-native-{i}.npz'
        np.savez_compressed(target,source_values=x,request_host_time=t,native_packet=packets,
            native_receipt=receipts,native_prefix_sha256=hashes,source_event_host_time=source_times,
            full_lap_request_mask=m)
        reports.append(dict(recording=str(root),qualification=str(q),target_initial_temperature_c=admission['target_initial_temperature_c'],
            actual_start_temperature_c=admission['actual_start_temperature_c'],all_requests=coverage(np.ones(len(x),bool)),
            first_five_seconds=coverage(early),full_lap=coverage(m),laps=proof['laps'],
            sidecar_sha256=sha256(target),sidecar_role='Diagnostic teacher-source bytes, not actor-tick join/normalized training inputs',
            actor_tick_join_verified=False,extended_native19_training_admissible=False,
            camera_body_training_admissible=True,exact_gpu_exposure_timestamp_verified=False))
        sources.append(dict(recording=str(root),events_sha256=sha256(events),qualification_sha256=sha256(q/'result.json'),
            thermal_admission_sha256=sha256(root/'teacher/thermal-start-admission.json')))
    write_json(run/'result.json',dict(status='completed_thermal_driving_source_sensation_audit',episodes=reports,
        source_receipts=sources,source_vector_order=['native_accG_xyz3','wheelSlip4','wheelLoad4','tyreCoreTemperature4','wheelsPressure4'],
        units=dict(acceleration='Original native G, no axis conversion in this diagnostic',wheel_slip='Native scalar; not asserted ratio/angle/friction utilization',
            wheel_load='Original native value; installed offline SDK load N contract',temperature='Native core Celsius',pressure='Native units, no PSI/kPa claim'),
        training_admissible=False,actor_qualified=False,student_optimizer_updates=0,
        termination_reason='Original request/source bytes, hashes and actual condition coverage reconstructed; no actor join, new visual preprocessing or training claim'))


if __name__=='__main__':main()
