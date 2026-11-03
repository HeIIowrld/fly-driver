"""Summarize saved one-second Windows GPU counters; no causal attribution."""

import argparse
import csv
import json
import re
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--pid', type=int, action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.csv)
    input_hash = sha256(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        records = list(reader)
    if not records or any(len(row) != len(header) for row in records):
        raise ValueError('Complete original typeperf table required')
    report = {}
    for pid in args.pid:
        indices = [i for i, name in enumerate(header) if re.search(
            rf'pid_{pid}_', name, flags=re.IGNORECASE)]
        usage = [i for i in indices if 'Utilization Percentage' in header[i]]
        memory = [i for i in indices if 'Dedicated Usage' in header[i]]
        def values(columns):
            return [float(row[i]) for row in records for i in columns if row[i].strip()]
        busy, mem = values(usage), values(memory)
        report[str(pid)] = {'engine_counter_columns': len(usage),
            'memory_counter_columns': len(memory),
            'single_engine_max_percent': max(busy) if busy else None,
            'dedicated_counter_max_mib': max(mem)/1024**2 if mem else None}
    if sha256(path) != input_hash:
        raise ValueError('Counter file changed during analysis')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'counter_sha256': input_hash, 'requested_pids': args.pid,
        'sampling_interval_seconds': 1, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'completed_gpu_counter_summary', 'sample_count': len(records),
        'first_timestamp': records[0][0], 'last_timestamp': records[-1][0],
        'processes': report, 'one_second_sampling_cannot_identify_147ms_cause': True,
        'actor_deployed': False}
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
