"""Summarize bounded teacher evidence without promoting diagnostic data to training."""
import base64
import hashlib
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    roots = [Path('D:/CodexResearchRuns/flybrain-teacher-shared-live-20261102-' + n)
             for n in ('01', '02', '03')]
    run = begin_run('D:/CodexResearchRuns/flybrain-teacher-shared-audit-20261102-01',
                    dict(script_sha256=sha256(__file__), sources=[str(p) for p in roots],
                         actor=False, optimizer_updates=0))
    summaries = []
    for root in roots:
        result = json.loads((root / 'result.json').read_text(encoding='utf8'))
        path = root / 'observations-and-requests.jsonl'
        if sha256(path) != result['events_archive_sha256']:
            raise ValueError('Event archive hash differs')
        events = [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]
        feedback = [x for x in events if x['kind'] == 'feedback']
        requests = [x for x in events if x['kind'] == 'teacher_request']
        intervals, previous = [], None
        for event in events:
            if event['kind'] == 'teacher_history_reset':
                previous = None
            if event['kind'] == 'raw_pair':
                wire = base64.b64decode(event['sdk']['wire_base64'], validate=True)
                if hashlib.sha256(wire).hexdigest() != event['sdk']['wire_sha256']:
                    raise ValueError('SDK raw evidence hash differs')
                timestamp = json.loads(wire)['timestamp_ac']
                if previous is not None:
                    intervals.append(timestamp - previous)
                previous = timestamp
        summaries.append(dict(run=str(root), status=result['status'],
            termination_reason=result.get('reason'), teacher_requests=len(requests),
            measured_feedback=len(feedback), initial_gap_m=feedback[0]['gap_m'] if feedback else None,
            final_gap_m=feedback[-1]['gap_m'] if feedback else None,
            max_speed_kmh=max((abs(x['speed_m_s'])*3.6 for x in feedback), default=None),
            out_of_track_feedback=sum(x['out_of_track'] for x in feedback),
            source_intervals_above_80ms=[x for x in intervals if x > .08],
            result_sha256=sha256(root / 'result.json'), events_sha256=sha256(path),
            training_admissible=result['training_admissible'],
            physical_exposure_sync_verified=result.get('vision', {}).get('physical_exposure_sync_verified')))
    write_json(run / 'result.json', dict(status='completed_evidence_audit', runs=summaries,
        optimizer_updates=0, actor_deployed=False, training_admissible=False,
        inference='Observed short on-track feedback is not a lap or synchronized training proof.'))


if __name__ == '__main__':
    main()
