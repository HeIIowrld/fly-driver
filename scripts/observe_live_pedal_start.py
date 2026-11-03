"""Record read-only paused native position and source timing for current AC session."""

import argparse
import json

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_transit_observer_v1 import observe_paused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-pid', type=int, required=True)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--lua-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = {'engine_pid': args.engine_pid, 'generation': args.generation, 'epoch': args.epoch,
            'lua_sha256': args.lua_sha256, 'observation_timeout_seconds': 2.0,
            'native_graphics_pair_minimum_seconds': .3}
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'plan': plan,
                                  'read_only': True, 'actor_input': False})
    try:
        result = observe_paused(plan, run / 'observer')
        write_json(run / 'result.json', {'status': 'completed', **result})
        print(json.dumps(result['observer']), flush=True)
    except Exception as error:
        write_json(run / 'result.json', {'status': 'failed', 'error': type(error).__name__,
                                         'reason': str(error)})
        raise


if __name__ == '__main__':
    main()
