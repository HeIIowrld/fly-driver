"""Inspect fixed body adapter columns; no optimization or control."""
import argparse,json,os
from pathlib import Path
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();base=Path(os.environ['LOCALAPPDATA'])/'flybrain/runtime-runs';ck=base/'20261101-speed-wheel-turn-control-fit/speed-body-adapter.pt';r=json.loads((ck.parent/'result.json').read_text(encoding='utf8'));source=base/'20261102-isolated-body-components-1/result.json'
 if sha256(ck)!=r['checkpoint_sha256'] or json.loads(source.read_text(encoding='utf8'))['status']!='completed_offline_unified_stationary_replay':raise ValueError('verified source and adapter required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(ck),intervention_result_sha256=sha256(source),actor=False,training=False));payload=torch.load(ck,map_location='cpu',weights_only=True);state=payload['model'];idx=state['indices'].tolist();w=state['projection.weight'];names=['pitch','roll','load_0','load_1','load_2','load_3','tyre_0','tyre_1','tyre_2','tyre_3','speed','wheel_0','wheel_1','wheel_2','wheel_3']
 if payload['actor_qualified'] is not False or idx!=[0,1,8,9,10,11,12,13,14,15,16,17,18,19,20] or tuple(w.shape)!=(201,30):raise ValueError('exact declared adapter structure required')
 columns={name:dict(body_index=idx[i],value_max_abs=float(w[:,i].abs().max()),availability_max_abs=float(w[:,i+15].abs().max())) for i,name in enumerate(names)}
 result=dict(status='completed_adapter_column_audit',columns=columns,actor_qualified=False,training=False,interpretation='Only this checkpoint and adapter; zero weights imply no effect through this adapter, not all possible circuits')
 write_json(run/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
