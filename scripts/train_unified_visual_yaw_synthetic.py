"""Train one existing visual boundary on controlled synthetic yaw; never vehicle control."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_stimulus(root,seed):
    result,config=read(root/'result.json'),read(root/'manifest.json')['config']
    path=root/'responses.npz'
    if (result['status'] != 'completed_coherent_four_view_yaw'
            or sha256(path) != result['responses_sha256'] or config['texture_seed'] != seed
            or result['cases'] != [0.,-.23,.23] or config['stationary_seconds'] != .4):
        raise ValueError('Exact independently declared synthetic texture source required')
    with np.load(path,allow_pickle=False) as f:
        retina,motion,times=f['retina'].copy(),f['motion'].copy(),f['times'].copy()
    if retina.shape != (3,41,4,721) or motion.shape != (3,41,4,8,721):
        raise ValueError('Complete original four-view synthetic responses required')
    return retina,motion,times


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    fit_source=base/'20261101-flyvis-coherent-four-view-yaw'
    dev_source=base/'20261101-flyvis-coherent-yaw-texture-863102'
    training=base/'20261029-t1437-embodied-joint-training'
    opponent=base/'20261031-opponent-temporal-graph-trained'
    context=base/'20261031-historical-train00-temporal-context-onepass'
    fit=base/'20261030-body-feedback-brake-steering-anchored'
    plan=read(base/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model']
    tr,op,cr,br=[read(x/'result.json') for x in (training,opponent,context,fit)]
    checkpoint=fit/'body-feedback-adapter.pt'
    if (sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']
            or sha256(context/'context-boundary.pt') != cr['checkpoint_sha256']
            or sha256(checkpoint) != br['checkpoint_sha256']
            or br['checkpoint_sha256'] != '4030ac7e53c9175a517adb3d588d9c5be4746f818ddc09762c6ee524771e2315'):
        raise ValueError('Fixed original graph and adapter lineage required')
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),seed=863009,
        fit_source_sha256=sha256(fit_source/'responses.npz'),dev_source_sha256=sha256(dev_source/'responses.npz'),
        optimizer='Adam lr=.01, 12 full-episode updates',fit_texture_seed=863101,dev_texture_seed=863102,
        objective='last-10 mean turn, relative to stationary: yaw -0.23 -> +.01; yaw +0.23 -> -.01',
        static_anchor='MSE to initial stationary last-10 mean',body_yaw=0,
        frozen_graph_sha256=sha256(plan['checkpoint']),actor_deployed=False))
    result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_deployed=False,
        physiological_reproduction_verified=False,only_synthetic_stimulus=True)
    try:
        torch.set_num_threads(1);torch.manual_seed(863009)
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        spec=read(training/'manifest.json')['config']['plan']
        source=read(Path(spec['source'])/'result.json')
        visual=EmbodiedVisualFlightCircuit(spec['source_spec'],torch.load(source['checkpoint'],
            map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
        visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
        physical=PhysicalManeuverCircuit(plan['spec'],torch.load(plan['checkpoint'],
            map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
        temporal=OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,
            visual.backbone.scene_command.scale,len(visual.backbone.vision_ports))
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model'])
        boundary=PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports))
        boundary.load_state_dict(torch.load(context/'context-boundary.pt',map_location='cpu',weights_only=True)['model'])
        model=UnifiedFlightSensorimotor(physical,visual.backbone,temporal,boundary)
        adapter=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
        for key in ('indices','scales'):
            if not torch.equal(getattr(model.body_feedback,key),adapter[key]):
                raise ValueError('Original body adapter contract changed')
        model.body_feedback.load_state_dict(adapter)
        model.motor.backbone.circuit.sparse_backend='cached_csr_gpu'
        model=model.to('cuda').eval().requires_grad_(False)
        model.temporal.opponent_projection.requires_grad_(True)
        parameter=model.temporal.opponent_projection.weight
        initial=parameter.detach().clone()
        optimizer=torch.optim.Adam([parameter],lr=.01)
        command=torch.tensor([[0.,10.]],device='cuda')
        goals=torch.tensor(racing_goal(100,1),device='cuda')
        view_valid=torch.ones(1,4,dtype=torch.bool,device='cuda')
        body=np.zeros(21,dtype=np.float64);body[16]=10.
        body_valid=np.ones(21,dtype=bool)

        def scenes(root,seed):
            retina,motion,times=load_stimulus(root,seed)
            image=np.concatenate((retina[:,:,:,None],motion),axis=3)
            with torch.no_grad():
                pooled=model.scene_pool(torch.tensor(image.reshape(-1,4,9,721),
                    dtype=torch.float32,device='cuda')).reshape(3,41,576)
            return pooled,times

        def rollout(images,times,learn):
            cases=[]
            for c in range(3):
                model.reset_episode();out=[]
                for i,t in enumerate(times):
                    now=1.+float(t)
                    observation=NavigationBodyObservation('synthetic',i,now-.01,now,
                        body.copy(),body_valid.copy(),'synthetic-fixture')
                    ctx=per_view_temporal_body_context((None,)*4,observation,
                        render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
                    method=model.training_decision if learn else model.decision
                    intent,_=method(decision_time=now,command=command,racing_goals=goals,
                        scene_features=images[c,i:i+1],scene_valid=view_valid,body_context=ctx)
                    out.append(intent[0])
                cases.append(torch.stack(out))
            return torch.stack(cases)

        fit_x,fit_time=scenes(fit_source,863101)
        baseline=rollout(fit_x,fit_time,False).detach()
        old=np.load(base/'20261101-unified-flight-coherent-four-view-yaw/outputs.npz',allow_pickle=False)
        np.testing.assert_allclose(baseline.cpu().numpy(),old['outputs'][0],rtol=0,atol=2e-6)
        old.close()
        static_anchor=baseline[0,-10:,0].mean().detach()
        torch.cuda.reset_peak_memory_stats()
        losses=[]
        for epoch in range(12):
            optimizer.zero_grad(set_to_none=True)
            pred=rollout(fit_x,fit_time,True)
            means=pred[:,-10:,0].mean(1)
            objective=(means[1]-means[0]-.01).square()+(means[2]-means[0]+.01).square()
            loss=objective+(means[0]-static_anchor).square()
            if not torch.isfinite(loss):raise ValueError('Finite synthetic objective required')
            loss.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError('Finite visual gradient required')
            grad=float(parameter.grad.norm().detach().cpu())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            write_json(run/'progress.json',dict(update=epoch+1,loss=losses[-1],gradient_norm=grad))
            result['optimizer_updates']=epoch+1
            del pred,loss
        final=rollout(fit_x,fit_time,False).detach().cpu().numpy()
        dev_x,dev_time=scenes(dev_source,863102)  # never opened for calibration or optimization
        dev_final=rollout(dev_x,dev_time,False).detach().cpu().numpy()
        trained=parameter.detach().clone()
        with torch.no_grad():parameter.copy_(initial)
        dev_initial=rollout(dev_x,dev_time,False).detach().cpu().numpy()
        with torch.no_grad():parameter.copy_(trained)
        def metrics(x):
            values=x[:,-10:,0].mean(1); delta=values[1:]-values[0]
            return dict(mean_turn=values.tolist(),relative_turn=delta.tolist(),
                signed_directions_correct=bool(delta[0]>0 and delta[1]<0),
                target_mse=float(np.mean((delta-np.array([.01,-.01]))**2)))
        path=run/'visual-temporal-boundary.pt'
        torch.save(dict(schema='flybrain.experimental.synthetic_yaw_boundary.v1',
            model=model.temporal.state_dict(),fit_texture_seed=863101,actor_qualified=False),path)
        np.savez_compressed(run/'outputs.npz',fit_before=baseline.cpu().numpy(),fit_after=final,
                            dev_before=dev_initial,dev_after=dev_final,fit_times=fit_time,dev_times=dev_time)
        result.update(status='completed_synthetic_visual_yaw_fit',optimizer_updates=12,
            training_loss=losses,fit_before=metrics(baseline.cpu().numpy()),fit_after=metrics(final),
            dev_before=metrics(dev_initial),dev_after=metrics(dev_final),
            parameter_change_norm=float((trained-initial).norm().cpu()),
            checkpoint_sha256=sha256(path),outputs_sha256=sha256(run/'outputs.npz'),
            peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),development_used_to_fit=False,
            checkpoint_actor_qualified=False,model_graph_frozen=True)
    except Exception as e:
        result.update(error=type(e).__name__,reason=str(e))
    finally:
        write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise RuntimeError(result['reason'])


if __name__=='__main__':
    main()
