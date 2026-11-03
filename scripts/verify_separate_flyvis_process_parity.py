"""Compare a visual process with the preexisting held-input calculation on synthetic RGB."""
import argparse, copy, gc, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.flyvis_process import FlyvisProcess, FlyvisEngine

def main():
    cfg=json.loads(Path('configs/flight-stream-batched-v1-20261029.json').read_text())
    camera=json.loads(Path(cfg['camera_config']).read_text())
    if sha256(cfg['camera_config'])!=cfg['camera_config_sha256'] or sha256(Path(cfg['author_model_dir'])/'best_chkpt')!=cfg['author_weights_sha256']: raise ValueError('Visual lineage changed')
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--device',choices=('cpu','cuda'),required=True);args=parser.parse_args()
    run=begin_run(args.output,dict(program_sha256=sha256(__file__),visual_worker_sha256=sha256('src/flybrain_drive/flyvis_process.py'),seed=863991,synthetic_rgb=True,optimizer_updates=0,device=args.device))
    rng=np.random.default_rng(863991)
    pool=FlightScenePool(rng.random((721,2)),dict(rows=4,columns=4,sigma_image_uv=.2))
    images=[tuple(rng.integers(0,256,size=(180,320,3),dtype=np.uint8) for _ in range(4)) for _ in range(3)]
    worker=None
    results=[]
    try:
        worker=FlyvisProcess(camera,cfg['author_model_dir'],cfg['cache_root'],copy.deepcopy(pool),timeout=120.,device=args.device)
        worker.timeout=120. if args.device=='cpu' else 10.
        for i,pixels in enumerate(images):
            results.append(worker.step(i,pixels))
            # Persist before requesting stop: a teardown failure must not erase
            # the observed output or its exact input and integration interval.
            np.savez(run / 'partial-observations.npz',
                pixels=np.stack([np.stack(p) for p in images[:i+1]]),
                scenes=np.stack([r['scene'] for r in results]),
                visual_snapshots=np.array([r['visual_snapshot'] for r in results],dtype=np.float64))
            write_json(run/'partial-status.json',dict(status='observed_not_qualified',frames=i+1,
                observations_sha256=sha256(run/'partial-observations.npz'),worker_pid=worker.process.pid,
                source_sha256=sha256('src/flybrain_drive/flyvis_process.py')))
        terminal=worker.close()
    except BaseException as exc:
        write_json(run/'result.json',dict(status='failed',stage='visual_worker',error=type(exc).__name__,
            reason=str(exc),worker_pid=worker.process.pid if worker else getattr(exc,'pid',None),worker_alive=worker.process.is_alive() if worker else getattr(exc,'alive',None),
            frames=len(results),partial_status_sha256=sha256(run/'partial-status.json') if results else None,
            visual_worker_sha256=sha256('src/flybrain_drive/flyvis_process.py')))
        raise
    engine=FlyvisEngine(camera,cfg['author_model_dir'],cfg['cache_root'],copy.deepcopy(pool),device=args.device)
    state=engine.state; held=None; previous=None; differences=[]
    with torch.no_grad(),torch.device(args.device):
        for pixels,res in zip(images,results,strict=True):
            sampled=np.stack([engine.projector.sample(image) for image in pixels])
            current=torch.as_tensor(sampled,device=args.device)
            now=res['visual_snapshot']
            if previous is not None:
                for dt,count in held_input_batches(now-previous):
                    while count:
                        n=min(count,16)
                        state=engine.network.simulate(held[:,None,None,:].expand(-1,n,-1,-1),dt=dt,initial_state=state,as_states=True)[-1];count-=n
            engine._synchronize()
            motion=state.nodes.activity[:,engine.indices]
            expected=engine.pool(torch.cat((current[:,None,:],motion),dim=1)[None]).to('cpu').numpy()
            engine._synchronize()
            differences.append(float(np.max(np.abs(expected-res['scene']))))
            held=current.detach().clone();previous=now
    # The baseline holds the last recurrent state separately from engine.
    # Release those references before recording its cleanup receipt.
    del state, held, current, motion
    engine.close();del engine;gc.collect()
    if args.device=='cuda':
        torch.cuda.empty_cache();torch.cuda.synchronize()
    result=dict(baseline_device_released=True,device=args.device,status='completed_synthetic_visual_process_parity' if max(differences)<=2e-5 else 'failed',max_abs_difference=max(differences),differences=differences,frames=len(images),terminal=terminal,visual_worker_sha256=sha256('src/flybrain_drive/flyvis_process.py'),interpretation='synthetic pixels, identical original visual intervals and pretrained network; not live timing or biological validation')
    write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise ValueError('Parity failed')
if __name__=='__main__':
    main()
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0)
