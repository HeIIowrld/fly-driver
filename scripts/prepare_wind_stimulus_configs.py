"""Create isolated offline wind test configurations; no live simulator modification."""

import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.wind_stimulus_config import isolated_wind_copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source)
    original = source.read_bytes()
    original_sha256 = sha256(source)
    configs = [(0, 0), (0, 36), (90, 36), (180, 36), (270, 36)]
    copies = [(direction, speed, isolated_wind_copy(original, direction_deg=direction,
                                                    speed_kmh=speed))
              for direction, speed in configs]
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_sha256': original_sha256, 'conditions': configs, 'scope': __doc__,
        'live_configuration_changed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    records = []
    for direction, speed, content in copies:
        path = run/f'wind-{direction:03d}-{speed:02d}.ini'
        path.write_bytes(content)
        records.append({'direction_deg': direction, 'speed_kmh': speed,
                        'file': path.name, 'sha256': sha256(path)})
    if sha256(source) != original_sha256:
        raise ValueError('Original simulator configuration changed during preparation')
    result = {'status': 'completed_isolated_wind_stimulus_preparation',
              'configs': records, 'source_sha256': original_sha256,
              'live_configuration_changed': False, 'simulator_restarted': False,
              'simulator_honors_config_verified': False, 'wind_source_qualified': False,
              'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
