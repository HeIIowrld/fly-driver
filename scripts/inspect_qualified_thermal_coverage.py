"""Measured native thermal support on qualified full-lap actor supervision only."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',required=True);p.add_argument('--qualification',required=True);p.add_argument('--actor-data',required=True);p.add_argument('--old-extension-contract',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    recording=Path(a.recording);qualification=Path(a.qualification);actor=Path(a.actor_data);qr=read(qualification/'result.json');ar=read(actor/'result.json');am=read(actor/'manifest.json')['config']
    if not qr['training_admissible'] or ar['status']!='completed_qualified_actor_preprocessing' or am['qualification_sha256']!=sha256(qualification/'result.json') or sha256(actor/'actor-only.npz')!=ar['arrays_sha256']:raise ValueError('Completed same-source qualified actor data required')
    path=recording/'camera/native-body.jsonl'
    if sha256(path)!=qr['native_body_sha256']:raise ValueError('Qualified native source changed')
    with np.load(actor/'actor-only.npz') as z:packets=z['body_packet'][z['fit_mask']].copy();times=z['decision_time'][z['fit_mask']].copy()
    needed=set(map(int,packets));samples={}
    for line in path.open(encoding='utf-8'):
        row=json.loads(line);auth=row['authority'];packet=auth['packet_id']
        if packet not in needed or packet in samples:continue
        raw=base64.b64decode(auth['prefix_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('Native bytes changed')
        samples[packet]=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
    if set(samples)!=needed:raise ValueError('Supervised body packet missing')
    temperatures=np.asarray([samples[int(i)].raw19[11:15] for i in packets]);old=read(a.old_extension_contract)['thermal_training_support_c'];outside=(temperatures<np.asarray(old['minimum']))|(temperatures>np.asarray(old['maximum']))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),qualification_sha256=sha256(qualification/'result.json'),actor_result_sha256=sha256(actor/'result.json'),old_contract_sha256=sha256(a.old_extension_contract),seed=None,optimizer_updates=0,vehicle_control_sent=False))
    write_json(out/'result.json',dict(status='completed_qualified_thermal_coverage',qualified_full_lap_labels=len(packets),unique_body_packets=len(needed),temperature_minimum_c=temperatures.min(0).tolist(),temperature_maximum_c=temperatures.max(0).tolist(),temperature_percentiles_c=np.percentile(temperatures,[5,50,95],axis=0).tolist(),labels_any_wheel_outside_previous_training_support=int(outside.any(1).sum()),previous_training_support_c=old,first_to_last_supervised_seconds=float(times[-1]-times[0]),initial_setting_is_not_measured_temperature=True,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,termination_reason='Original qualified native bytes selected by exact full-lap supervised actor packets; actual thermal coverage, not learned adaptation or autonomous performance.'))

if __name__=='__main__':main()
