"""Summarize original sequenced owner timings, including failed requests."""
import argparse
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.shadow);result=read(root/'result.json');events=read(root/'client-request-events.json');stages=read(root/'client-stage-events.json');owner=read(root/'owner-terminal.json')
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),original_sha256={f:sha256(root/f) for f in ['result.json','client-request-events.json','client-stage-events.json','owner-terminal.json']},vehicle_control_sent=False,optimizer_updates=0))
    rows=[]
    for event in events:
        selected=[s for s in stages if s['request_id']==event['request_id']]
        row=dict(event,request_duration_seconds=event['finished']-event['started'],stages=selected)
        if selected:
            row['send_to_first_stage_seconds']=selected[0]['host_time']-event['send_finished']
            row['last_stage_to_observed_finish_seconds']=event['finished']-selected[-1]['host_time']
            row['stage_intervals']=[dict(start=x['stage'],end=y['stage'],seconds=y['host_time']-x['host_time']) for x,y in zip(selected,selected[1:])]
        rows.append(row)
    write_json(out/'requests.json',rows)
    write_json(out/'result.json',dict(status='completed_original_owner_timing_summary',shadow_status=result['status'],owner_status=owner['status'],requests=len(rows),failed_requests=[r for r in rows if r['status']=='failed'],maximum_ordinary_request_seconds=max((r['request_duration_seconds'] for r in rows if r['timeout_seconds']==.05),default=None),requests_sha256=sha256(out/'requests.json'),owner_outputs=owner['outputs'],ticker_error=owner['ticker_error'],vehicle_control_sent=False,actual_driving_improvement_verified=False))

if __name__=='__main__':main()
