"""Evaluate road-pose confounds in recorded opposite-corner interventions.

All world/road values stay in a separate diagnostic artifact, never actor data.
"""
import argparse,base64,hashlib,json,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,PHYSICS_LAYOUT
from flybrain_drive.diagnostic_road_pose import project_road_pose,paired_pose_bounds
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('actor-data','interventions','plan','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();plan=read(a.plan);actor=Path(a.actor_data);d,c,provenance=load_episode(actor);iv=Path(a.interventions);ir=read(iv/'result.json');im=read(iv/'manifest.json')['config']
    if ir['status']!='completed_motion_spatial_same_body_interventions' or im['entry']['arrays_sha256']!=provenance['arrays_sha256']:raise ValueError('Exact original intervention receiver episode required')
    cfg=read(actor/'manifest.json')['config'];recording=Path(cfg['arguments']['recording']);teacher=recording/'teacher';tr=read(teacher/'result.json');source=teacher/'tcp-source-packets.jsonl'
    if sha256(source)!=tr['tcp_source_archive_sha256']:raise ValueError('Original TCP archive changed')
    geo=read('configs/ac-teacher-geometry-result.json')['geometry_configuration']
    if sha256(geo['track_csv'])!=geo['track_sha256']:raise ValueError('Diagnostic track geometry changed')
    track=np.genfromtxt(geo['track_csv'],delimiter=',',names=True);center=np.column_stack([(track['left_border_x']+track['right_border_x'])/2,(track['left_border_y']+track['right_border_y'])/2])
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),road_pose_source_sha256=sha256('src/flybrain_drive/diagnostic_road_pose.py'),actor_arrays_sha256=provenance['arrays_sha256'],interventions_result_sha256=sha256(iv/'result.json'),source_tcp_sha256=sha256(source),track_sha256=geo['track_sha256'],plan=plan,plan_sha256=sha256(a.plan),seed=None,optimizer_updates=0,actor_input=False,vehicle_control_sent=False,actor_qualified=False,scope='Exact nearest road-center segment and original source-copy bracketing endpoints. Diagnostic sampled-pose intervals only, not guaranteed continuous exposure pose or an independently correct counterfactual action. No actor/teacher data mutation.'));shutil.copyfile(__file__,out/'executed-entrypoint.py')
    result=dict(status='failed',actor_input=False,vehicle_control_sent=False,actor_qualified=False,optimizer_updates=0)
    try:
        times=[];positions=[];yaws=[];speeds=[];native_packets=[];producer=None;copy_spans=[]
        for line in source.open(encoding='utf8'):
            row=json.loads(line);wire=base64.b64decode(row['wire_base64'],validate=True)
            if hashlib.sha256(wire).hexdigest()!=row['wire_sha256']:raise ValueError('Original TCP wire bytes differ')
            x=json.loads(wire);namespace=x['privileged_teacher'];native=x['source_native_snapshot']
            if namespace['actor_input'] is not False or namespace['protocol']!='flybrain.ac.privileged_teacher.v1' or native['protocol']!='flybrain.ac.source_native.v1' or not native['repeated_copy_equal']:raise ValueError('Explicit teacher-only source-native snapshot required')
            if producer is None:producer=x['producer_session_id']
            if x['producer_session_id']!=producer:raise ValueError('Original source episode changed')
            raw=base64.b64decode(native['prefix_base64'],validate=True);values=iter(PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]));fields={k:tuple(next(values) for _ in range(n)) for k,_,n in PHYSICS_LAYOUT};hz=native['qpc_frequency_hz'];t=native['qpc_finished_ticks']/hz;span=(native['qpc_finished_ticks']-native['qpc_started_ticks'])/hz
            if not 0<=span<=plan['maximum_source_copy_seconds'] or t>row['received_monotonic_seconds'] or times and t<=times[-1]:raise ValueError('Original monotonic bounded source-copy clocks required')
            extra=namespace['raw'];times.append(t);positions.append([extra['world_position_x'],extra['world_position_y']]);yaws.append(-fields['heading'][0]);speeds.append(fields['speedKmh'][0]/3.6);native_packets.append(fields['packetId'][0]);copy_spans.append(span)
        times=np.asarray(times);positions=np.asarray(positions);yaws=np.asarray(yaws);speeds=np.asarray(speeds);ids=np.unique(d['camera_index']);first=np.array([np.flatnonzero(d['camera_index']==i)[0] for i in ids]);bounds=d['render_lowers'][first];before=np.searchsorted(times,bounds.min(1),side='right')-1;after=np.searchsorted(times,bounds.max(1),side='left');in_range=(before>=0)&(after<len(times));safe_before=np.clip(before,0,len(times)-1);safe_after=np.clip(after,0,len(times)-1);gaps=times[safe_after]-times[safe_before];admissible=in_range&(gaps<=plan['maximum_pose_bracket_seconds']);needed=np.unique(np.r_[safe_before,safe_after]);road=project_road_pose(positions[needed],yaws[needed],center);lookup={int(k):j for j,k in enumerate(needed)};brackets=np.asarray([[np.r_[road[lookup[int(k)],:2],speeds[k]] for k in pair] for pair in zip(safe_before,safe_after)]);distance=np.asarray([[road[lookup[int(k)],2] for k in pair] for pair in zip(safe_before,safe_after)])
        if not np.isfinite(brackets).all() or not np.isfinite(distance).all():raise ValueError('Finite diagnostic source pose required')
        moving=speeds[needed]>5;aligned=float(np.mean(abs(road[moving,1])<.5)) if moving.any() else 0.
        if aligned<plan['minimum_moving_tangent_agreement']:raise ValueError('Road tangent orientation/ground coordinate compatibility failed; no automatic flip')
        event_map=np.searchsorted(ids,d['camera_index']);reports=[];all_deltas=[];all_valid=[];windows=ir['windows'];limits=np.asarray([plan['maximum_offset_difference_m'],plan['maximum_heading_difference_rad'],plan['maximum_speed_difference_mps']])
        def comparison(receiver,donor_window):
            indices=np.asarray(receiver['pre_indices'],int);w=receiver['receiver'];target=donor_window['receiver']['time']+d['decision_time'][indices]-w['time'];donor_events=np.searchsorted(d['decision_time'],target,side='right')-1
            if np.any(donor_events<0) or np.any(target>d['decision_time'][-1]):raise ValueError('Donor original source clock outside episode')
            rj=event_map[indices];dj=event_map[donor_events];delta=paired_pose_bounds(brackets[rj],brackets[dj]);valid=admissible[rj]&admissible[dj];passes=valid&np.all(delta<=limits,axis=1)
            return dict(donor_event=donor_window['receiver']['event'],donor_time=donor_window['receiver']['time'],receiver_pre_events=len(indices),source_bracket_valid_events=int(valid.sum()),pose_matched_events=int(passes.sum()),entire_window_pose_matched=bool(np.all(passes)),maximum_observed_endpoint_difference=delta.max(0).tolist(),median_observed_endpoint_difference=np.median(delta,axis=0).tolist(),receiver_pose_event_indices=rj.tolist(),donor_pose_event_indices=dj.tolist()),delta,valid
        for receiver in windows:
            w=receiver['receiver'];original=next(x for x in windows if x['receiver']['event']==w['donor_event']);original_report,delta,valid=comparison(receiver,original);candidates=[comparison(receiver,z)[0] for z in windows if z['receiver']['direction']!=w['direction']];best=min(candidates,key=lambda z:max(np.asarray(z['maximum_observed_endpoint_difference'])/limits));reports.append(dict(window=receiver['window'],receiver_event=w['event'],receiver_direction=w['direction'],original_donor=original_report,opposite_candidates=candidates,best_diagnostic_donor=best,any_entire_window_pose_matched=any(z['entire_window_pose_matched'] for z in candidates)));all_deltas.extend(delta);all_valid.extend(valid)
        np.savez_compressed(out/'diagnostic-road-poses.npz',camera_indices=ids,event_to_camera=event_map,source_times=times,source_positions=positions,source_yaws=yaws,source_speeds_mps=speeds,source_native_packets=np.asarray(native_packets),camera_pose_bracket_source_indices=np.column_stack([safe_before,safe_after]),pose_brackets=brackets,nearest_center_distance=distance,pose_bracket_admissible=admissible,pose_bracket_span_seconds=gaps,projected_source_indices=needed,projected_source_road_pose=road)
        delta=np.asarray(all_deltas);valid=np.asarray(all_valid);matched=valid&np.all(delta<=limits,axis=1)
        result.update(status='completed_precorner_road_pose_diagnostic',source_packets=len(times),camera_groups=len(ids),source_bracket_admissible_groups=int(admissible.sum()),moving_tangent_agreement=aligned,maximum_source_copy_seconds=max(copy_spans),windows=reports,paired_windows=len(reports),original_entire_windows_pose_matched=sum(z['original_donor']['entire_window_pose_matched'] for z in reports),any_opposite_entire_windows_pose_matched=sum(z['any_entire_window_pose_matched'] for z in reports),original_pre_events=len(delta),original_source_bracket_valid_events=int(valid.sum()),original_pose_matched_pre_events=int(matched.sum()),original_maximum_endpoint_difference=delta.max(0).tolist(),original_median_endpoint_difference=np.median(delta,axis=0).tolist(),arrays_sha256=sha256(out/'diagnostic-road-poses.npz'),actor_truth_inputs_added=False,continuous_exposure_pose_verified=False,counterfactual_action_correctness_verified=False,termination_reason='Original source TCP bytes/clocks projected to road-center diagnostic coordinates; opposite-corner road-pose/speed confounds and strict sampled-bracket matching support measured without actor changes or invented matches')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    write_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='windows'}))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
