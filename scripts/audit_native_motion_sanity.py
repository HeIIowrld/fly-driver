"""Receipt-time native motion sanity; no simulation-clock or latency certification."""

import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path


def measurements(source):
    source = Path(source)
    data = (source/'events.jsonl').read_bytes()
    original = json.loads((source/'result.json').read_text())
    digest = hashlib.sha256(data).hexdigest()
    if digest != original['events_sha256']:
        raise ValueError('Original event hash changed')
    rows = [json.loads(line) for line in data.splitlines()]
    samples = [r for r in rows if r['kind'] == 'sample']
    xyz = [r['native_graphics']['coordinates_native_xyz_m'] for r in samples]
    clock = [r['native']['received_monotonic_seconds'] for r in samples]
    speed = [r['native']['fields']['speedKmh']/3.6 for r in samples]
    path = sum(math.hypot(b[0]-a[0], b[2]-a[2])
               for a, b in zip(xyz[:-1], xyz[1:], strict=True))
    integral = sum(.5*(a+b)*(v-u) for a, b, u, v in zip(
        speed[:-1], speed[1:], clock[:-1], clock[1:], strict=True))
    velocity = []
    for row in samples:
        native = row['native']
        raw = base64.b64decode(native['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != native['prefix_sha256']:
            raise ValueError('Original native bytes changed')
        vector = struct.unpack_from('<3f', raw, 32)
        velocity.append(math.sqrt(sum(v*v for v in vector)))
    return dict(events_sha256=digest, samples=len(samples),
        native_horizontal_path_sum_m=path,
        native_horizontal_endpoint_displacement_m=math.hypot(
            xyz[-1][0]-xyz[0][0], xyz[-1][2]-xyz[0][2]),
        speed_host_receipt_trapezoid_m=integral, path_minus_receipt_integral_m=path-integral,
        maximum_velocity_norm_vs_speed_m_s=max(abs(a-b)
            for a, b in zip(velocity, speed, strict=True)),
        maximum_receipt_interval_seconds=max(b-a
            for a, b in zip(clock[:-1], clock[1:], strict=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--verify-result', required=True)
    args = parser.parse_args()
    observed = measurements(args.source)
    result = Path(args.verify_result)
    saved = json.loads(result.read_text())
    for key, value in observed.items():
        if saved[key] != value:
            raise ValueError('Stored motion sanity is not exactly reproducible: '+key)
    code = Path(__file__).read_bytes()
    (result.parent/'executed-program.py').write_bytes(code)
    receipt = dict(status='passed_exact_motion_sanity_recomputation',
        source_run=args.source, program_sha256=hashlib.sha256(code).hexdigest(),
        verified_result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(), **observed,
        optimizer_updates=0, runtime_mappings_opened=0, actuator_messages_sent=0,
        seed=None, deterministic_arithmetic=True, scope=__doc__)
    (result.parent/'reproducibility.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
