"""Synthetic road bend x lateral position x translation through author T4/T5.

Functional component stimuli only. No scanned track, measured neural recording,
native sensor, physical camera clock or driving qualification is produced.
"""
import argparse,copy,gc,hashlib,itertools,json,os,shutil,sys,time,yaml
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.motion_per_view_flyvis_stream import MotionPerViewFlyvisStream
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import assemble_student,read
from flybrain_drive.synthetic_antialiased_planar_road_visual import AntialiasedSyntheticPlanarRoadVisual
from flybrain_drive.synthetic_yaw_body import planar_pose,yaw_profile


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1)
    for name,expected in plan['source_sha256'].items():
        if sha256(name)!=expected:raise ValueError('Sealed stimulus source changed '+name)
    cfg=read('configs/flight-stream-batched-v1-20261029.json');md=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root']);camera_file=Path('configs/ac-csp-exterior-front-v3-retina.json');geometry_file=Path('runs/thermal-driving-20261103/spatial-recovery-01/pool-geometry.npz')
    if sha256(md/'best_chkpt')!=cfg['author_weights_sha256'] or sha256(camera_file)!=plan['camera_sha256'] or sha256(geometry_file)!=plan['geometry_sha256']:raise ValueError('Frozen author/camera/pool geometry changed')
    with np.load(geometry_file) as z:geometry={k:z[k].copy() for k in ('old','inverse','grid8_narrow')}
    model,_=assemble_student();pool=model.scene_pool.to('cuda').eval().requires_grad_(False);np.testing.assert_array_equal(pool.weight.cpu().numpy(),geometry['old']);camera=read(camera_file);road=AntialiasedSyntheticPlanarRoadVisual(camera,width_m=plan['road_width_m'],bend_start_m=plan['bend_start_m'],curvature=plan['curvature'],samples_per_axis=plan['samples_per_axis']);single=[]
    for j in range(4):
        view=copy.copy(road);view.rays=[road.rays[j]];view.origins=[road.origins[j]];single.append(view)
    os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
    import flyvis
    with torch.device('cuda'):
        nc=yaml.safe_load((md/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**nc).to('cuda');network.load_state_dict(torch.load(md/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False);nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{j}' for k in ('T4','T5') for j in 'abcd']);projector=FlyvisCameraProjector(camera,u,v)
    cases=[dict(bend=b,lateral_m=0.,speed_mps=s,yaw_rate_target_rad_s=w) for s,w,b in itertools.product(plan['speeds_mps'],plan['yaw_rate_targets_rad_s'],plan['bends'])];out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),plan=plan,plan_sha256=sha256(a.plan),cases=cases,author_weights_sha256=cfg['author_weights_sha256'],process_pid=os.getpid(),seed=plan['texture_seed'],optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope=__doc__));shutil.copyfile(__file__,out/'executed-entrypoint.py');arrays={k:[] for k in ('scene','static','motion','coarse')};rgb_hashes=[];clock=np.asarray([[100+i*plan['frame_dt']+j*plan['view_dt'] for j in range(4)] for i in range(plan['frame_count'])]);samples=[];pose_data=[];minimum_entry_margin=float('inf');result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False);started=time.perf_counter()
    try:
        for k,case in enumerate(cases):
            with torch.device('cuda'):prior=network.steady_state(1.,.005,batch_size=4,value=.5)
            stream=MotionPerViewFlyvisStream(network,prior,pool,projector,indices,geometry,episode_id=f'synthetic-case-{k}',generation='synthetic-rotation-v1',device='cuda');values={key:[] for key in arrays};hashes=[];case_poses=[]
            for i,clocks in enumerate(clock):
                pixels={};poses=[]
                for j,name in enumerate(VIEW_ORDER):
                    elapsed=float(clocks[j]-100.);bend=case['bend'] if i>=plan['branch_frame'] else 0;pose=planar_pose(elapsed,case['speed_mps'],case['yaw_rate_target_rad_s'],onset=plan['rotation_onset_s'],duration=plan['rotation_duration_s']);longitudinal,lateral,yaw,_=map(float,pose);poses.append(pose)
                    origin=camera['views'][name]['origin_body_m'];margin=plan['bend_start_m']-(longitudinal+float(np.cos(yaw))*origin[0]-float(np.sin(yaw))*origin[1]);minimum_entry_margin=min(minimum_entry_margin,margin)
                    if margin<=0:raise ValueError('Rotated synthetic camera entered bend; pre-entry condition lost')
                    pixels[name]=single[j].render(bend=bend,forward_m=longitudinal,lateral_m=lateral,yaw_rad=yaw,texture_seed=plan['texture_seed'])[0]
                case_poses.append(np.asarray(poses));obs=stream.process(pixels=pixels,render_start_lowers=tuple(map(float,clocks)),camera_received=float(max(clocks)+.007),capture_id=i,generation='synthetic-rotation-v1');stream.prepare_held_prefix();hashes.append(list(obs.rgb_sha256))
                for key,value in [('scene',obs.scene),('static',obs.static),('motion',obs.motion_features),('coarse',obs.coarse_motion_reference)]:values[key].append(value.copy())
                if i in (plan['branch_frame'],plan['frame_count']-1):samples.append((k,i,pixels['front'].copy()))
            for key in arrays:arrays[key].append(np.asarray(values[key]))
            rgb_hashes.append(hashes);pose_data.append(np.asarray(case_poses));write_json(out/'progress.json',dict(completed_cases=k+1,total=len(cases),elapsed_seconds=time.perf_counter()-started))
        maximum_prefix=0.
        for k in range(0,len(cases),2):
            assert cases[k]['yaw_rate_target_rad_s']==cases[k+1]['yaw_rate_target_rad_s']
            assert cases[k]['lateral_m']==cases[k+1]['lateral_m'] and cases[k]['speed_mps']==cases[k+1]['speed_mps'] and cases[k]['bend']==-cases[k+1]['bend']
            assert rgb_hashes[k][:plan['branch_frame']]==rgb_hashes[k+1][:plan['branch_frame']]
            for key in arrays:maximum_prefix=max(maximum_prefix,float(abs(arrays[key][k][:plan['branch_frame']]-arrays[key][k+1][:plan['branch_frame']]).max()))
        if maximum_prefix>2e-6:raise ValueError('Opposite bend pair lacks common visual prefix')
        np.savez_compressed(out/'features.npz',**{k:np.asarray(v) for k,v in arrays.items()},render_lowers=clock,camera_receipt=clock.max(1)+.007,case_bend=np.array([c['bend'] for c in cases]),case_lateral_m=np.array([c['lateral_m'] for c in cases]),case_speed_mps=np.array([c['speed_mps'] for c in cases]),case_yaw_rate_target_rad_s=np.array([c['yaw_rate_target_rad_s'] for c in cases]),render_pose=np.asarray(pose_data),view_body_sample_time=clock-.001,decision_body_sample_time=clock.max(1)+.011);write_json(out/'rgb-hashes.json',rgb_hashes)
        sheet=Image.new('RGB',(640,204*len(cases)));draw=ImageDraw.Draw(sheet)
        for k,i,rgb in samples:
            col=0 if i==plan['branch_frame'] else 1;sheet.paste(Image.fromarray(rgb),(col*320,k*204+24));draw.text((col*320+3,k*204+4),f'{cases[k]} frame {i}',fill='white')
        sheet.save(out/'stimulus-sheet.png');result.update(status='completed_visual_body_rotation_stimuli',minimum_camera_bend_entry_margin_m=minimum_entry_margin,body_motion_is_analytic_hypothesis=True,cases=len(cases),frames_per_case=plan['frame_count'],state_resets=len(cases),opposite_pair_prefix_max_difference=maximum_prefix,features_sha256=sha256(out/'features.npz'),rgb_hashes_sha256=sha256(out/'rgb-hashes.json'),stimulus_sheet_sha256=sha256(out/'stimulus-sheet.png'),synthetic_camera_times=True,actual_native_body_samples=False,pre_entry_only=True,termination_reason='Independent bend/yaw-pulse/speed factorial with integrated planar trajectory rendered at each synthetic view time through unchanged author per-view T4/T5 stream; common straight visual prefixes verified, no fitting or actual driving')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    try:
        # The final speculative prefix may still own outstanding GPU work.
        # Drain it and release local tensor owners before normal interpreter
        # shutdown; an immediate Windows ExitProcess retained the first tool.
        torch.cuda.synchronize()
        if 'stream' in locals():stream.prefix=None;del stream
        if 'prior' in locals():del prior
        del network,pool,model,indices,nodes,receptors
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        result['explicit_gpu_drain_and_release_completed']=True
    except Exception as error:result.update(status='failed',gpu_cleanup_error=str(error))
    result['elapsed_seconds']=time.perf_counter()-started;write_json(out/'result.json',result);print(json.dumps(result),flush=True);return 1 if result['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
