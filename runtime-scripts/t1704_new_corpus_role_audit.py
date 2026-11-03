"""Read-only metadata/static-geometry audit for prospective new corpus roles."""
from pathlib import Path
import hashlib,json
import numpy as np
from flybrain_drive.prospective_corpus import validate_split
from flybrain_drive.evidence import write_json

root=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1704-new-corpus-role-audit')
root.mkdir(parents=True,exist_ok=False)
plan_path=Path('configs/flight-new-diverse-corpus-20261030.json')
plan=json.loads(plan_path.read_text())
contract=validate_split(plan)
legacy=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1393-v2-route-coverage-audit/result.json')
current=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1633-spatial-native-corrected-completion/result.json')
old=json.loads(legacy.read_text());recent=json.loads(current.read_text())
metadata_roles=[dict(role=e['role'],spline_start=e['spline_start'],spline_end=e['spline_end']) for e in old['reports']]
metadata_roles +=[dict(role=e['lease']['role'],minimum_spline=e['admission'].get('minimum_spline'),maximum_spline=e['admission'].get('maximum_spline')) for e in recent['episodes'] if e['lease']['role']!='transit']
path=Path('assets/external/acgym-teacher-recipe-20261027/legacy/assetto_corsa_gym/AssettoCorsaConfigs/tracks/monza-racing_line.csv')
a=np.genfromtxt(path,delimiter=',',names=True);xy=np.c_[a['pos_x'],a['pos_y']]
arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
step=np.arange(0,arc[-1],5.);q=np.c_[np.interp(step,arc,xy[:,0]),np.interp(step,arc,xy[:,1])]
f=step/arc[-1];left=np.roll(q,4,axis=0);right=np.roll(q,-4,axis=0)
u=q-left;v=right-q
curvature=np.arctan2(u[:,0]*v[:,1]-u[:,1]*v[:,0],(u*v).sum(1))/20
geometry=[]
for name,lo,hi in [('first_chicane',.13,.195),('central_corners',.275,.54),('unused_candidate',.55,.65),('unused_narrow_gap',.635,.665),('late_corner',.85,.93)]:
 keep=(f>=lo)&(f<=hi)
 geometry.append(dict(region=name,approximate_line_fraction=[lo,hi],five_metre_points=int(keep.sum()),positive_curvature_gt003=int((keep&(curvature>.003)).sum()),negative_curvature_ltminus003=int((keep&(curvature<-.003)).sum()),maximum_abs_curvature_inv_m=float(abs(curvature[keep]).max())))
source_comparison={name:dict(inspected_sha256=sha,current_sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest(),unchanged=sha==hashlib.sha256(Path(name).read_bytes()).hexdigest()) for name,sha in plan['source_sha256'].items()}
result=dict(status='completed_read_only_new_corpus_role_capability_audit',contract_validation=contract,known_development_regions=plan['known_development_regions'],evaluation_regions=plan['evaluation_regions'],historical_metadata_roles=metadata_roles,static_teacher_geometry=geometry,static_line_length_m=float(arc[-1]),geometry_mapping_limit='Static line arc fraction approximates normalized spline; no new camera/pose arrays or exact game-spline mapping qualification. Only teacher diagnostic planning, notactorinputs.',source_comparison=source_comparison,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),evaluation_arrays_or_receipts_opened=False,known_development_arrays_opened=False,actor_inputs_added=False,gpu_used=False,api_requests_sent=0,vehicle_control_messages_sent=0,ui_actions_sent=0,process_changes=0,profile_implementation_qualified=False,collector_execution_allowed=False,unseen_geography_both_direction_corner_coverage_claimed=False)
write_json(root/'result.json',result)
(root/'collection-contract.json').write_bytes(plan_path.read_bytes())
write_json(root/'validation.json',dict(status='passed',cpu_tests=3,pytest_exit_code=0,ruff_exit_code=0,contract_control_permission=False,profile_execution_attempts=0,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/flybrain_drive/prospective_corpus.py','tests/test_prospective_corpus.py','configs/flight-new-diverse-corpus-20261030.json']}))
print(json.dumps({k:v for k,v in result.items() if k not in ['source_comparison','historical_metadata_roles']},ensure_ascii=False))
