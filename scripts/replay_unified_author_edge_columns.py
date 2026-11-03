"""Controlled FlyVis full-column edges through one flight state; offline only."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--columns',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--coherent-yaw',action='store_true',help='Replay synthetic four-view coherent yaw instead of author edges')
    p.add_argument('--body-yaw-mode',choices=('fixed','matched','opposite'),default='fixed',
                   help='Synthetic body yaw relative to coherent visual yaw; no driving')
    a = p.parse_args()
    if not a.coherent_yaw and a.body_yaw_mode != 'fixed':
        raise ValueError('Synthetic body yaw modes require coherent-yaw stimuli')
    columns = Path(a.columns)
    proof = read(columns/'result.json')
    if a.coherent_yaw:
        source = columns/'responses.npz'
        if (proof['status'] != 'completed_coherent_four_view_yaw'
                or proof['cases'] != [0.,-.23,.23] or sha256(source) != proof['responses_sha256']):
            raise ValueError('Exact original coherent four-view synthetic source required')
        source_hash = proof['responses_sha256']
        with np.load(source,allow_pickle=False) as f:
            retinal,motion,times = f['retina'].copy(),f['motion'].copy(),f['times'].copy()
        if (retinal.shape != (3,41,4,721) or motion.shape != (3,41,4,8,721)
                or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12)):
            raise ValueError('Three complete coherent four-view histories required')
    else:
        source = columns/'columns.npz'
        if (proof['status'] != 'completed_author_edge_full_columns'
                or proof['cases'] != [[0,0],[0,180],[1,0],[1,180]]
                or sha256(source) != proof['columns_sha256']):
            raise ValueError('Exact full-column author-edge source required')
        source_hash = proof['columns_sha256']
        with np.load(source,allow_pickle=False) as f:
            retinal,motion = f['retina'].copy(),f['motion'].copy()
            if float(f['dt']) != .005:
                raise ValueError('Original author stimulus timestep required')
        if retinal.shape != (4,485,721) or motion.shape != (4,485,8,721):
            raise ValueError('Four whole original 721-column histories required')
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    training = base/'20261029-t1437-embodied-joint-training'
    opponent = base/'20261031-opponent-temporal-graph-trained'
    context = base/'20261031-historical-train00-temporal-context-onepass'
    fit = base/'20261030-body-feedback-brake-steering-anchored'
    plan = read(base/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model']
    tr,op,cr,br = [read(x/'result.json') for x in (training,opponent,context,fit)]
    checkpoint = fit/'body-feedback-adapter.pt'
    if (sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']
            or sha256(context/'context-boundary.pt') != cr['checkpoint_sha256']
            or sha256(checkpoint) != br['checkpoint_sha256']
            or br['checkpoint_sha256'] != '4030ac7e53c9175a517adb3d588d9c5be4746f818ddc09762c6ee524771e2315'):
        raise ValueError('Original graph and unqualified experimental checkpoints required')
    run = begin_run(a.output,dict(program_sha256=sha256(__file__),column_sha256=source_hash,
        model_sha256=sha256(plan['checkpoint']), visual_sha256=sha256(tr['checkpoint']),
        temporal_sha256=op['checkpoint_sha256'],context_sha256=cr['checkpoint_sha256'],
        body_sha256=br['checkpoint_sha256'], seed=863009, frame_stride=8,
        synthetic_body_yaw_mode=a.body_yaw_mode,
        scope=('synthetic coherent four-view yaw, constant body/command, no vehicle' if a.coherent_yaw
               else 'synthetic author edges in one view, constant body/command, no vehicle')))
    result = dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,
        actor_deployed=False,physiological_reproduction_verified=False)
    try:
        torch.set_num_threads(1); torch.manual_seed(863009)
        if not torch.cuda.is_available(): raise RuntimeError('GPU required')
        spec = read(training/'manifest.json')['config']['plan']
        source = read(Path(spec['source'])/'result.json')
        visual = EmbodiedVisualFlightCircuit(spec['source_spec'],torch.load(source['checkpoint'],
            map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
        visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
        physical = PhysicalManeuverCircuit(plan['spec'],torch.load(plan['checkpoint'],
            map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
        temporal = OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,
            visual.backbone.scene_command.scale,len(visual.backbone.vision_ports))
        temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model'])
        boundary = PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports))
        boundary.load_state_dict(torch.load(context/'context-boundary.pt',map_location='cpu',weights_only=True)['model'])
        model = UnifiedFlightSensorimotor(physical,visual.backbone,temporal,boundary)
        trained = torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
        for key in ('indices','scales'):
            if not torch.equal(getattr(model.body_feedback,key),trained[key]):
                raise ValueError('Historical adapter channel or scale mismatch')
        model.body_feedback.load_state_dict(trained)
        model.motor.backbone.circuit.sparse_backend='cached_csr_gpu'
        model = model.to('cuda').eval().requires_grad_(False)
        command = torch.tensor([[0.,10.]],device='cuda')
        goals = torch.tensor(racing_goal(100,1),device='cuda')
        valid = torch.ones(1,4,dtype=torch.bool,device='cuda')
        body_values = np.zeros(21,dtype=np.float64); body_values[16] = 10.
        body_mask = np.ones(21,dtype=bool)
        sample_frames = np.arange(41) if a.coherent_yaw else np.arange(0,485,8)
        stimulus_times = times if a.coherent_yaw else sample_frames*.005
        outputs = np.zeros((1 if a.coherent_yaw else 4,3 if a.coherent_yaw else 4,len(sample_frames),2))
        with torch.no_grad():
            for view in range(outputs.shape[0]):
                for case in range(outputs.shape[1]):
                    model.reset_episode()
                    if not a.coherent_yaw:
                        fixed = np.concatenate((retinal[case,0,None],motion[case,0]),axis=0)
                    for j,frame in enumerate(sample_frames):
                        now = 1.+float(stimulus_times[j])
                        if a.coherent_yaw and stimulus_times[j] > .4:
                            sign = {'fixed':0.,'matched':1.,'opposite':-1.}[a.body_yaw_mode]
                            body_values[4] = sign*proof['cases'][case]
                        else:
                            body_values[4] = 0.
                        body = NavigationBodyObservation('synthetic',j,now-.01,now,
                            body_values.copy(),body_mask.copy(),'synthetic-fixture')
                        ctx = per_view_temporal_body_context((None,)*4,body,
                            render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
                        if a.coherent_yaw:
                            image = np.concatenate((retinal[case,frame,:,None],motion[case,frame]),axis=1)
                        else:
                            image = np.repeat(fixed[None],4,axis=0)
                            image[view] = np.concatenate((retinal[case,frame,None],motion[case,frame]),axis=0)
                        scene = model.scene_pool(torch.as_tensor(image[None],dtype=torch.float32,device='cuda'))
                        intent,_ = model.decision(decision_time=now,command=command,racing_goals=goals,
                            scene_features=scene,scene_valid=valid,body_context=ctx)
                        outputs[view,case,j] = intent.cpu().numpy()[0]
                write_json(run/'progress.json',dict(completed_views=view+1,total_views=outputs.shape[0]))
        path = run/'outputs.npz'
        np.savez_compressed(path,outputs=outputs,stimulus_seconds=stimulus_times)
        if a.coherent_yaw:
            delta=outputs[0,2]-outputs[0,1]
            result.update(status='completed_coherent_yaw_unified_graph',cases=proof['cases'],
                views=['front','left','right','rear'],observations=len(sample_frames),
                output_sha256=sha256(path), opposite_yaw_max_abs=abs(delta).max(0).tolist(),
                opposite_yaw_signed_mean_after_onset=delta[stimulus_times>=.4].mean(0).tolist(),
                pre_onset_max_abs=abs(delta[stimulus_times<=.4]).max(0).tolist(),
                synthetic_body_yaw_mode=a.body_yaw_mode, constant_command=True,
                all_views_move_together=True,stimuli_synthetic=True,
                direction_sign_or_closed_loop_correctness_verified=False)
            write_json(run/'result.json',result)
            print(json.dumps(result))
            return
        pairs = {}
        for v,name in enumerate(('front','left','right','rear')):
            pairs[name] = {}
            for c,label in ((0,'dark'),(2,'bright')):
                delta = outputs[v,c]-outputs[v,c+1]
                pairs[name][label] = dict(max_abs=abs(delta).max(0).tolist(),
                    mean_signed_after_one_second=delta[stimulus_times>=1].mean(0).tolist())
        result.update(status='completed_controlled_author_edges_unified_graph',
            cases=proof['cases'], views=['front','left','right','rear'], observations=len(sample_frames),
            output_sha256=sha256(path),opposite_direction_difference=pairs,
            first_5ms_frame_then_original_40ms_samples=True,constant_body_and_command=True,
            other_views_held_at_initial_visual_response=True,stimuli_synthetic=True,
            direction_sign_or_closed_loop_correctness_verified=False)
    except Exception as e:
        result.update(error=type(e).__name__,reason=str(e))
    finally:
        write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status'] == 'failed': raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
