"""Independent byte/clock/projection and original-donor confound audit."""
import argparse,base64,hashlib,json,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.diagnostic);r=read(root/'result.json');m=read(root/'manifest.json')['config'];plan=m['plan'];d,c,provenance=load_episode(m['arguments']['actor_data']);iv=read(Path(m['arguments']['interventions'])/'result.json');cfg=read(Path(m['arguments']['actor_data'])/'manifest.json')['config'];source=Path(cfg['arguments']['recording'])/'teacher/tcp-source-packets.jsonl'
    assert r['status']=='completed_precorner_road_pose_diagnostic' and sha256(root/'diagnostic-road-poses.npz')==r['arrays_sha256'] and sha256(source)==m['source_tcp_sha256'] and provenance['arrays_sha256']==m['actor_arrays_sha256']
    with np.load(root/'diagnostic-road-poses.npz') as z:x={k:z[k].copy() for k in z.files}
    times=[];positions=[];yaws=[];speeds=[];packets=[]
    for line in source.open(encoding='utf8'):
        row=json.loads(line);wire=base64.b64decode(row['wire_base64'],validate=True);assert hashlib.sha256(wire).hexdigest()==row['wire_sha256'];packet=json.loads(wire);native=packet['source_native_snapshot'];raw=base64.b64decode(native['prefix_base64'],validate=True);values=PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]);offset=0;fields={}
        for name,_,width in PHYSICS_LAYOUT:fields[name]=values[offset:offset+width];offset+=width
        assert packet['privileged_teacher']['actor_input'] is False and native['repeated_copy_equal'] is True
        times.append(native['qpc_finished_ticks']/native['qpc_frequency_hz']);positions.append([packet['privileged_teacher']['raw'][k] for k in ('world_position_x','world_position_y')]);yaws.append(-fields['heading'][0]);speeds.append(fields['speedKmh'][0]/3.6);packets.append(fields['packetId'][0])
    for name,expected in [('source_times',times),('source_positions',positions),('source_yaws',yaws),('source_speeds_mps',speeds),('source_native_packets',packets)]:np.testing.assert_array_equal(x[name],expected)
    np.testing.assert_array_equal(x['camera_indices'][x['event_to_camera']],d['camera_index']);geo=read('configs/ac-teacher-geometry-result.json')['geometry_configuration'];assert sha256(geo['track_csv'])==m['track_sha256'];track=np.genfromtxt(geo['track_csv'],delimiter=',',names=True);center=np.column_stack([(track['left_border_x']+track['right_border_x'])/2,(track['left_border_y']+track['right_border_y'])/2])
    if np.linalg.norm(center[-1]-center[0])<1e-8:center=center[:-1]
    vectors=np.roll(center,-1,axis=0)-center;length2=np.einsum('ij,ij->i',vectors,vectors);maximum_projection_error=0.
    for j,k in enumerate(x['projected_source_indices']):
        point=x['source_positions'][k];relative=point-center;fraction=np.clip(np.einsum('ij,ij->i',relative,vectors)/length2,0,1);difference=relative-fraction[:,None]*vectors;distance2=np.einsum('ij,ij->i',difference,difference);segment=int(np.argmin(distance2));v=vectors[segment];delta=difference[segment];yaw=x['source_yaws'][k]-np.arctan2(v[1],v[0]);expected=np.array([(v[0]*delta[1]-v[1]*delta[0])/np.sqrt(length2[segment]),np.arctan2(np.sin(yaw),np.cos(yaw)),np.sqrt(distance2[segment]),segment,fraction[segment]]);maximum_projection_error=max(maximum_projection_error,float(abs(expected-x['projected_source_road_pose'][j]).max()))
    assert maximum_projection_error<=1e-9
    lookup={int(k):j for j,k in enumerate(x['projected_source_indices'])};limits=np.array([plan['maximum_offset_difference_m'],plan['maximum_heading_difference_rad'],plan['maximum_speed_difference_mps']]);matched=valid_count=0;all_delta=[];window_counts=[]
    for original,report in zip(iv['windows'],r['windows']):
        w=original['receiver'];indices=np.array(original['pre_indices']);donor_times=w['donor_time']+d['decision_time'][indices]-w['time'];donor_events=np.searchsorted(d['decision_time'],donor_times,side='right')-1;ri=x['event_to_camera'][indices];di=x['event_to_camera'][donor_events];assert ri.tolist()==report['original_donor']['receiver_pose_event_indices'] and di.tolist()==report['original_donor']['donor_pose_event_indices'];counts=0
        for receiver,donor in zip(ri,di):
            endpoints=[];good=True
            for camera in (receiver,donor):
                events=np.flatnonzero(d['camera_index']==x['camera_indices'][camera]);clocks=d['render_lowers'][events[0]];before=np.flatnonzero(x['source_times']<=min(clocks))[-1];after=np.flatnonzero(x['source_times']>=max(clocks))[0];np.testing.assert_array_equal([before,after],x['camera_pose_bracket_source_indices'][camera]);good &= x['source_times'][after]-x['source_times'][before]<=plan['maximum_pose_bracket_seconds'];values=np.array([np.r_[x['projected_source_road_pose'][lookup[k],:2],x['source_speeds_mps'][k]] for k in (before,after)]);np.testing.assert_array_equal(values,x['pose_brackets'][camera]);endpoints.append(values)
            differences=[]
            for left in endpoints[0]:
                for right in endpoints[1]:
                    diff=left-right;diff[1]=np.arctan2(np.sin(diff[1]),np.cos(diff[1]));differences.append(abs(diff))
            delta=np.max(differences,axis=0);all_delta.append(delta);valid_count+=int(good);counts+=int(good and np.all(delta<=limits))
        assert counts==report['original_donor']['pose_matched_events'];matched+=counts;window_counts.append(counts)
    delta=np.asarray(all_delta);assert matched==r['original_pose_matched_pre_events'] and valid_count==r['original_source_bracket_valid_events'];np.testing.assert_allclose(delta.max(0),r['original_maximum_endpoint_difference'],atol=1e-12,rtol=0);np.testing.assert_allclose(np.median(delta,axis=0),r['original_median_endpoint_difference'],atol=1e-12,rtol=0)
    candidate_count=0;any_matched=0
    for original,report in zip(iv['windows'],r['windows']):
        receivers=np.asarray(original['pre_indices']);w=original['receiver'];expected_donors={z['receiver']['event'] for z in iv['windows'] if z['receiver']['direction']!=w['direction']};assert {z['donor_event'] for z in report['opposite_candidates']}==expected_donors;whole=[]
        for candidate in report['opposite_candidates']:
            donor=next(z['receiver'] for z in iv['windows'] if z['receiver']['event']==candidate['donor_event']);assert donor['time']==candidate['donor_time'];stamp=donor['time']+d['decision_time'][receivers]-w['time'];donor_events=np.searchsorted(d['decision_time'],stamp,side='right')-1;ri=x['event_to_camera'][receivers];di=x['event_to_camera'][donor_events];differences=x['pose_brackets'][ri,:,None,:]-x['pose_brackets'][di,None,:,:];differences[:,:,:,1]=np.arctan2(np.sin(differences[:,:,:,1]),np.cos(differences[:,:,:,1]));distance=np.max(abs(differences),axis=(1,2));valid=x['pose_bracket_admissible'][ri]&x['pose_bracket_admissible'][di];passes=valid&np.all(distance<=limits,axis=1);assert candidate['pose_matched_events']==int(passes.sum()) and candidate['source_bracket_valid_events']==int(valid.sum()) and candidate['entire_window_pose_matched']==bool(passes.all());np.testing.assert_allclose(candidate['maximum_observed_endpoint_difference'],distance.max(0),atol=1e-12,rtol=0);whole.append(bool(passes.all()));candidate_count+=1
        assert report['any_entire_window_pose_matched']==any(whole);any_matched+=int(any(whole))
    assert any_matched==r['any_opposite_entire_windows_pose_matched']
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),diagnostic_result_sha256=sha256(root/'result.json'),source_tcp_sha256=m['source_tcp_sha256'],actor_arrays_sha256=provenance['arrays_sha256'],actor_input=False,optimizer_updates=0,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_precorner_road_pose_audit',source_packets_verified=len(times),projected_source_points_verified=len(x['projected_source_indices']),projection_max_difference=maximum_projection_error,original_pairs_verified=len(iv['windows']),original_pre_events_verified=len(delta),original_source_bracket_valid_events=valid_count,original_pose_matched_events=matched,original_window_matching_counts=window_counts,all_opposite_candidates_verified=candidate_count,any_opposite_entire_windows_pose_matched=any_matched,actor_truth_inputs_added=False,vehicle_control_sent=False,optimizer_updates=0,termination_reason='Original TCP byte fields, clocks, camera source bracket identities and independent exhaustive nearest-segment projections/original and all alternative donor endpoint statistics checked; road-center asset approximation and sampled bracket limitations remain'))


if __name__=='__main__':main()
