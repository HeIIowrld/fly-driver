"""Summarize archived SAC labels without promoting a replay-only dataset to training."""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding='utf8'))
    if sha256(__file__)!=plan['program_sha256']:raise ValueError('program mismatch')
    for path,digest in plan['sources'].items():
        if sha256(path)!=digest:raise ValueError('source mismatch: '+path)
    run=begin_run(args.output,dict(plan_sha256=sha256(args.plan),program_sha256=sha256(__file__),actor=False))
    result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0)
    try:
        meta=json.loads(Path(plan['dataset']).read_text(encoding='utf8'))
        original=json.loads(Path(plan['recording_result']).read_text(encoding='utf8'))
        episode=Path(plan['episode'])
        if meta['episodes'][0]['sha256'] != sha256(episode):raise ValueError('episode provenance mismatch')
        if meta.get('experimental_offline_replay_only') is not True or original.get('training_admissible') is not False:
            raise ValueError('expected original replay-only admission state')
        with load_array(episode,allow_pickle=False) as z:
            actions=z['action'].copy();valid=z['action_valid'].copy();views=z['obs_view_rgb'].shape
        if actions.shape!=(200,3) or valid.shape!=actions.shape or not np.isfinite(actions).all():
            raise ValueError('unexpected actions or validity')
        stats={}
        for i,name in enumerate(('steer','throttle','brake')):
            x=actions[valid[:,i],i]
            stats[name]={'valid':int(len(x)),'minimum':float(x.min()),'median':float(np.median(x)),
                'p95':float(np.percentile(x,95)),'maximum':float(x.max()),
                'absolute_over_005':int((np.abs(x)>.05).sum())}
        result.update(status='completed',samples=len(actions),stored_view_shape=list(views),action_statistics=stats,
            labels=meta['teacher_label_contract'],original_training_admissible=False,
            experimental_offline_replay_only=True,physical_exposure_sync_verified=False,
            scope='Previously executed speed-limited SAC requests, one short episode; counts do not prove teacher policy correctness or actuation timing. No fit.')
    except Exception as exc:result.update(error=type(exc).__name__,reason=str(exc))
    write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
