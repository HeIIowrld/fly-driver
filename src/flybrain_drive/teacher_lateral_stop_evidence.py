"""Independent privileged lateral/stop replay; never an actor qualification."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
from .ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX
from .ac_teacher_geometry import PrivilegedTrackGeometry
from .ac_geometric_teacher import GeometricRecoveryTeacher
from .graph import sha256
from .sac_unified_student import read


def audit_lateral_stop(teacher_root,events,tr,tm):
    root=Path(teacher_root);args=tm['arguments'];path=args.get('low_speed_lateral_calibration')
    if not path:return None
    contract=read(root/'teacher-lateral-contract.json');fit=read(path)
    assert contract['actor_input'] is False and contract['privileged_route'] is True
    assert sha256(path)==contract['fit_sha256']==tm['low_speed_lateral_fit_sha256']
    assert contract['current_profile_sha256']==tm['axis_configuration']['profile_sha256']
    geo=read('configs/ac-teacher-geometry-result.json')['geometry_configuration']
    geometry=PrivilegedTrackGeometry(geo['track_csv'],geo['reference_csv'],track_sha256=geo['track_sha256'],reference_sha256=geo['reference_sha256'])
    teacher=GeometricRecoveryTeacher(geometry.line,fit['curvature_per_canonical_steering'],maximum_speed_kmh=20.)
    latest=None;requests=[];feedback=[];checks=[];close_time=None
    for event in events:
        kind=event['kind']
        if kind=='raw_pair':
            for item,tag in [(event['sdk'],'wire'),(event['native'],'prefix')]:
                value=base64.b64decode(item[tag+'_base64'],validate=True);assert hashlib.sha256(value).hexdigest()==item[tag+'_sha256']
            sdk=json.loads(base64.b64decode(event['sdk']['wire_base64']));raw=base64.b64decode(event['native']['prefix_base64']);values=PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]);fields={};index=0
            for name,_,width in PHYSICS_LAYOUT:
                fields[name]=values[index:index+width];index+=width
            state=dict(sdk['privileged_teacher']['raw']);state.update(speed=sdk['speed'],brakeStatus=sdk['brakeStatus'],yaw=-fields['heading'][0],numberOfTyresOut=fields['numberOfTyresOut'][0])
            state['gap']=geometry.features(state['world_position_x'],state['world_position_y'],state['yaw'])['gap']
            latest=(fields['packetId'][0],state)
        elif kind in ('teacher_request','teacher_stop_request'):
            assert latest is not None
            expected=float(teacher.request(latest[1])[0]*args['steering_request_scale'])
            control=event['request']['requested_control'];assert abs(control['steering']-expected)<=1e-10
            checks.append(abs(control['steering']-expected))
            if kind=='teacher_stop_request':
                assert control['throttle']==0 and control['brake']==1 and 0<=event['source_capture_age_before_actuation_ms']<=50
                requests.append(event)
        elif kind=='teacher_stop_feedback':
            assert event['packet']==latest[0] and event['tyres_out']==latest[1]['numberOfTyresOut']==0
            np.testing.assert_allclose([event['speed_kmh'],event['brake']],[latest[1]['speed']*3.6,latest[1]['brakeStatus']],atol=1e-12,rtol=0)
            feedback.append(event)
        elif kind=='close':close_time=event['host_time']
    stop=tr.get('terminal_stop',{});complete=stop.get('status')=='completed_teacher_feedback_stop'
    if complete:
        assert len(requests)==len(feedback)==stop['requests'] and len(feedback)>=3
        assert len({e['packet'] for e in feedback[-3:]})==3
        assert all(e['speed_kmh']<=.5 and e['brake']>=.98 for e in feedback[-3:])
        assert feedback[-1]['host_time']<=close_time and stop['elapsed_seconds']<=3
    return dict(status='audited_teacher_only_lateral_and_stop',completed=complete,normal_and_stop_steering_replays=len(checks),maximum_steering_difference=max(checks) if checks else None,
        stop_requests=len(requests),stationary_feedback_verified=3 if complete else 0,
        historical_curvature_transfer=True,current_curvature_refit_verified=False,actor_input=False)
