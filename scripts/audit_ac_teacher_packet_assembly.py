"""Wrap public records in synthetic SDK/native containers to audit teacher assembly."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, decode_physics_prefix
from flybrain_drive.ac_sac_teacher import CHANNELS, TeacherReferenceCurve
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.ac_teacher_observation import PrivilegedTeacherObservationBuilder
from flybrain_drive.ac_transport import decode_telemetry
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.numeric_pickle import load_numeric_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(Path('configs/ac-sac-replay.json').read_text())
    geometry_config = json.loads(Path('configs/ac-teacher-geometry-result.json').read_text())[
        'geometry_configuration']
    record = Path(config['record_path'])
    if sha256(record) != config['record_sha256']:
        raise ValueError('Public state source hash differs')
    run = begin_run(args.output, dict(scope=__doc__, source_sha256=sha256(record),
        synthetic_transport_and_native_wrappers=True, actual_sdk_observation_validation=False))
    geometry = PrivilegedTrackGeometry(geometry_config['track_csv'],
        geometry_config['reference_csv'], track_sha256=geometry_config['track_sha256'],
        reference_sha256=geometry_config['reference_sha256'])
    builder = PrivilegedTeacherObservationBuilder(geometry,
        TeacherReferenceCurve(geometry_config['reference_csv']), profile=config['profile'])
    states = load_numeric_pickle(record.read_bytes())['states']
    ratios = [row['LapDist'] / row['NormalizedSplinePosition'] for row in states
              if row['NormalizedSplinePosition'] > 1e-6]
    length = float(np.median(ratios))
    observations = []
    offset, native_offsets = 0, {}
    for name, _, width in PHYSICS_LAYOUT:
        native_offsets[name] = offset
        offset += width * 4
    for index, row in enumerate(states):
        extra_keys = (set(CHANNELS) - {'speed', 'gap', 'angular_velocity_y'}) | {
            'world_position_x', 'world_position_y', 'NormalizedSplinePosition'}
        extra = {key: float(row[key]) for key in extra_keys}
        extra['track_length_m'] = length
        payload = {key: float(row[key]) for key in ['speed', 'steerAngle', 'accStatus',
                                                   'brakeStatus', 'angular_velocity_y']}
        payload.update(steps=index + 1, timestamp_ac=float(row['timestamp_ac']),
            producer_session_id='public-record-in-synthetic-transport',
            privileged_teacher=dict(protocol='flybrain.ac.privileged_teacher.v1',
                                    actor_input=False, raw=extra))
        host_time = 100 + index * .04
        packet = decode_telemetry(json.dumps(payload).encode(), host_time)
        raw = bytearray(308)
        struct.pack_into('<i', raw, 0, index + 1)
        struct.pack_into('<f', raw, native_offsets['heading'], -row['yaw'])
        struct.pack_into('<i', raw, native_offsets['numberOfTyresOut'],
                         int(row['numberOfTyresOut']))
        native = decode_physics_prefix(bytes(raw), host_time + .001)
        result = builder.build(packet, native)
        if result.observation is not None:
            observations.append(result.observation)
    cached_root = Path('runs/ac-teacher-replay-20261027')
    relative = 't21b/runs/teacher-replay/privileged-teacher-replay.npz'
    cached = cached_root / relative
    receipt = json.loads((cached_root / 'collection-receipt.json').read_text())
    if sha256(cached) != receipt['files_sha256'][relative]:
        raise ValueError('Earlier verified observations differ')
    with load_array(cached, allow_pickle=False) as data:
        error = float(np.abs(observations - data['teacher_observations']).max())
    np.savez_compressed(run / 'privileged-assembled-observations.npz',
                        observations=np.array(observations))
    report = dict(public_states=len(states), assembled_observations=len(observations),
        warmup_rows_excluded=3, max_observation_absolute_error=error,
        track_length_reconstructed_from_public_values_m=length,
        reconstruction_matched=error < 1e-12,
        synthetic_transport_and_native_wrappers=True, actual_sdk_live_verified=False,
        actor_observations_created=False, actuator_messages_sent=0, goal_achieved=False)
    write_json(run / 'result.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
