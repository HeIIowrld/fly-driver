"""Replay preserved CPU worker frames through the original held-input computation."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.flyvis_process import FlyvisEngine

def main():
    source=Path('D:/CodexResearchRuns/flybrain-separate-visual-parity-cpu-20261102-03')
    record=json.loads((source/'result.json').read_text());partial=json.loads((source/'partial-status.json').read_text())
    datafile=source/'partial-observations.npz'
    if record['status']!='failed' or record['reason']!='ValueError: visual processing gap too large' or record['frames']!=2 or sha256(source/'partial-status.json')!=record['partial_status_sha256'] or sha256(datafile)!=partial['observations_sha256'] or partial['source_sha256']!=record['visual_worker_sha256']:raise ValueError('Original partial evidence differs')
    cfg=json.loads(Path('configs/flight-stream-batched-v1-20261029.json').read_text());camera=json.loads(Path(cfg['camera_config']).read_text())
    if sha256(cfg['camera_config'])!=cfg['camera_config_sha256'] or sha256(Path(cfg['author_model_dir'])/'best_chkpt')!=cfg['author_weights_sha256']:raise ValueError('FlyVis lineage differs')
    run=begin_run('D:/CodexResearchRuns/flybrain-separate-visual-partial-replay-cpu-20261102-01',dict(program_sha256=sha256(__file__),source_sha256=sha256(datafile),source_result_sha256=sha256(source/'result.json'),worker_source_sha256=record['visual_worker_sha256'],seed=863991,device='cpu',optimizer_updates=0))
    with np.load(datafile,allow_pickle=False) as z:pixels=z['pixels'].copy();scenes=z['scenes'].copy();times=z['visual_snapshots'].copy()
    if pixels.shape!=(2,4,180,320,3) or scenes.shape!=(2,1,576) or not 0<times[1]-times[0]<=.15:raise ValueError('Exact valid preserved observations required')
    rng=np.random.default_rng(863991);pool=FlightScenePool(rng.random((721,2)),dict(rows=4,columns=4,sigma_image_uv=.2))
    engine=FlyvisEngine(camera,cfg['author_model_dir'],cfg['cache_root'],copy.deepcopy(pool),device='cpu')
    state=engine.state;held=None;previous=None;differences=[]
    with torch.no_grad(),torch.device('cpu'):
        for image,scene,now in zip(pixels,scenes,times,strict=True):
            current=torch.as_tensor(np.stack([engine.projector.sample(view) for view in image]),device='cpu')
            if previous is not None:
                for dt,count in held_input_batches(float(now-previous)):
                    while count:
                        n=min(count,16);state=engine.network.simulate(held[:,None,None,:].expand(-1,n,-1,-1),dt=dt,initial_state=state,as_states=True)[-1];count-=n
            motion=state.nodes.activity[:,engine.indices]
            expected=engine.pool(torch.cat((current[:,None,:],motion),dim=1)[None]).cpu().numpy()
            differences.append(float(np.max(np.abs(expected-scene))))
            held=current.detach().clone();previous=now
    result=dict(status='completed_partial_cpu_recorded_visual_replay' if max(differences)<=2e-5 else 'failed',max_abs_difference=max(differences),differences=differences,frames=2,original_interval_seconds=float(times[1]-times[0]),device='cpu',source_run_status='failed',interpretation='two preserved synthetic frames only; original third frame rejected; not GPU or live qualification')
    write_json(run/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise ValueError('Replay mismatch')
if __name__=='__main__':main()
