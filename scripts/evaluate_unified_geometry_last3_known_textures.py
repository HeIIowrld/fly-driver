"""Evaluate same fixed graph with last-three synthetic stationary reference; no fit or control."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


class GeometryVisualPort(nn.Module):
    """Offline-only hypothesis: causal geometry signal modulates original visual ports."""
    def __init__(self, base, ports):
        super().__init__()
        self.base=base
        self.projection=nn.Linear(6,ports,bias=False)
        nn.init.zeros_(self.projection.weight)
        self.current=None

    def forward(self,traces,body,valid):
        if self.current is None or self.current.shape != (1,6) or not torch.isfinite(self.current).all():
            raise ValueError('Explicit finite causal geometry input required')
        return self.base(traces,body,valid)+.5*torch.tanh(self.projection(self.current))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_stimulus(root,seed,amplitude):
    result,config=read(root/'result.json'),read(root/'manifest.json')['config']
    path=root/'responses.npz'
    if (result['status'] != 'completed_coherent_four_view_yaw'
            or sha256(path) != result['responses_sha256'] or config['texture_seed'] != seed
            or result['cases'] != [0.,-amplitude,amplitude] or config['stationary_seconds'] != .4):
        raise ValueError('Exact independently declared synthetic texture source required')
    with np.load(path,allow_pickle=False) as f:
        retina,motion,times=f['retina'].copy(),f['motion'].copy(),f['times'].copy()
    if (retina.shape != (3,41,4,721) or motion.shape != (3,41,4,8,721)
            or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12)
            or not np.all(retina[:,:11] == retina[:,0:1])):
        raise ValueError('Complete original four-view synthetic responses required')
    return retina,motion,times


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--body-yaw-mode',choices=('fixed','matched','opposite'),default='fixed')
    a=p.parse_args()
    base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    fit_sources=[base/'20261101-flyvis-coherent-four-view-yaw',
        base/'20261101-flyvis-coherent-yaw-texture-863103',
        base/'20261101-flyvis-coherent-yaw-texture-863104']
    fit_seeds=[863101,863103,863104]
    dev_source=base/'20261101-flyvis-coherent-yaw-texture-863108-speed013'
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
    candidate=base/'20261101-unified-synthetic-geometry-yaw-axis-fit'
    trained_path=candidate/'geometry-visual-boundary.pt'
    if sha256(trained_path) != 'a9ed4f224b7dd00731ad51adcbd91f3bbe10812c83dd6cdfc31db950834927d8':
        raise ValueError('Exact unqualified trained projection required')
    run=begin_run(a.output,dict(evaluation_checkpoint_sha256=sha256(trained_path),program_sha256=sha256(__file__),seed=863009,
        fit_source_sha256=[sha256(p/'responses.npz') for p in fit_sources],dev_source_sha256=sha256(dev_source/'responses.npz'),
        optimizer='none; fixed-checkpoint evaluation only',fit_texture_seeds=fit_seeds,dev_texture_seed=863108,
        objective='last-10 mean turn, relative to stationary: yaw -0.23 -> +.01; yaw +0.23 -> -.01',
        static_anchor='MSE to initial stationary last-10 mean',body_yaw_mode=a.body_yaw_mode,
        geometry='fixed camera ray-weighted yaw axis only; causal stationary last3 of first11 reference; zero until ready',
        geometry_camera_sha256=sha256('configs/ac-csp-exterior-front-v3-retina.json'),
        geometry_registration_sha256=sha256(base/'20261031-pace-train-flyvis-replay/retina-registration.npz'),
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
        model.view_context=GeometryVisualPort(model.view_context,len(model.motor.backbone.vision_ports))
        adapter=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
        for key in ('indices','scales'):
            if not torch.equal(getattr(model.body_feedback,key),adapter[key]):
                raise ValueError('Original body adapter contract changed')
        model.body_feedback.load_state_dict(adapter)
        model.motor.backbone.circuit.sparse_backend='cached_csr_gpu'
        model=model.to('cuda').eval().requires_grad_(False)
        model.view_context.projection.requires_grad_(True)
        parameter=model.view_context.projection.weight
        initial=parameter.detach().clone()
        command=torch.tensor([[0.,10.]],device='cuda')
        goals=torch.tensor(racing_goal(100,1),device='cuda')
        view_valid=torch.ones(1,4,dtype=torch.bool,device='cuda')
        body=np.zeros(21,dtype=np.float64);body[16]=10.
        body_valid=np.ones(21,dtype=bool)
        camera_file=Path('configs/ac-csp-exterior-front-v3-retina.json')
        registration=base/'20261031-pace-train-flyvis-replay/retina-registration.npz'
        with np.load(registration,allow_pickle=False) as f:
            pool=VisualSelfMotionPool.from_registration(read(camera_file),{k:f[k].copy() for k in f.files})

        def scenes(root,seed,amplitude=.23):
            retina,motion,times=load_stimulus(root,seed,amplitude)
            image=np.concatenate((retina[:,:,:,None],motion),axis=3)
            with torch.no_grad():
                pooled=model.scene_pool(torch.tensor(image.reshape(-1,4,9,721),
                    dtype=torch.float32,device='cuda')).reshape(3,41,576)
            features=pool.features(motion)
            reference=features[:,8:11].mean(1,keepdims=True)
            geometry=features-reference
            geometry[:,:11]=0
            geometry[:,:,[0,1,3,4,5]]=0
            return pooled,times,torch.tensor(geometry,dtype=torch.float32,device='cuda')

        def rollout(images,times,geometry,learn):
            cases=[]
            for c in range(3):
                model.reset_episode();out=[]
                for i,t in enumerate(times):
                    now=1.+float(t)
                    sign={'fixed':0.,'matched':1.,'opposite':-1.}[a.body_yaw_mode]
                    body[4]=sign*[0.,-.23,.23][c] if t>.4 else 0.
                    observation=NavigationBodyObservation('synthetic',i,now-.01,now,
                        body.copy(),body_valid.copy(),'synthetic-fixture')
                    ctx=per_view_temporal_body_context((None,)*4,observation,
                        render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
                    method=model.training_decision if learn else model.decision
                    model.view_context.current=geometry[c,i:i+1]
                    intent,_=method(decision_time=now,command=command,racing_goals=goals,
                        scene_features=images[c,i:i+1],scene_valid=view_valid,body_context=ctx)
                    out.append(intent[0])
                cases.append(torch.stack(out))
            return torch.stack(cases)

        saved=torch.load(trained_path,map_location='cpu',weights_only=True)
        if saved['actor_qualified'] is not False or saved['fit_texture_seeds'] != fit_seeds:
            raise ValueError('Original non-actor fit lineage required')
        trained=saved['model']['weight'].to('cuda')
        if trained.shape != parameter.shape or not torch.isfinite(trained).all():
            raise ValueError('Exact finite projection dimensions required')
        known=[(863102,base/'20261101-flyvis-coherent-yaw-texture-863102'),
               (863105,base/'20261101-flyvis-coherent-yaw-texture-863105'),
               (863106,base/'20261101-flyvis-coherent-yaw-texture-863106'),
               (863107,base/'20261101-flyvis-coherent-yaw-texture-863107')]
        outputs=[]; reports=[]
        for seed,path in known:
            x,t,g=scenes(path,seed)
            with torch.no_grad(): parameter.copy_(initial)
            baseline=rollout(x,t,g,False).detach().cpu().numpy()
            with torch.no_grad(): parameter.copy_(trained)
            changed=rollout(x,t,g,False).detach().cpu().numpy()
            def score(values):
                means=values[:,-10:,0].mean(1); delta=means[1:]-means[0]
                return dict(relative_turn=delta.tolist(),
                    signed_directions_correct=bool(delta[0]>0 and delta[1]<0),
                    target_mse=float(np.mean((delta-np.array([.01,-.01]))**2)))
            reports.append(dict(seed=seed,role='known_development',
                responses_sha256=sha256(path/'responses.npz'),before=score(baseline),after=score(changed)))
            outputs.append(np.stack([baseline,changed]))
            write_json(run/'progress.json',dict(completed=len(outputs),total=len(known)))
        np.savez_compressed(run/'outputs.npz',outputs=np.stack(outputs),times=t)
        result.update(status='completed_known_texture_evaluation',optimizer_updates=0,
            reports=reports,checkpoint_sha256=sha256(trained_path),
            outputs_sha256=sha256(run/'outputs.npz'),known_development_only=True,
            checkpoint_actor_qualified=False,model_graph_frozen=True,body_yaw_mode=a.body_yaw_mode,
            historical_per_view_body_context='explicitly missing',
            reference_policy='synthetic stationary last3 of first11 only')
    except Exception as e:
        result.update(error=type(e).__name__,reason=str(e))
    finally:
        write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise RuntimeError(result['reason'])


if __name__=='__main__':
    main()
