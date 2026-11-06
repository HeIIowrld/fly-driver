"""Record final full-range port trial without promoting a research candidate."""
import os,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.control_ac_session import read_session_heartbeat

ROOT=Path('runs/full-range-port-learning-20261104')
CHARTER='초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.'

def main():
 full_init=[read(Path('runs/full-routing-initialization-20261104')/f'initial-episode{i}-01/result.json') for i in [0,1]];assert all(r['optimizer_updates']==0 for r in full_init);assert read(ROOT/'process-inventory-final-01.json')==[];p=read(ROOT/'plan-01.json');old=read(ROOT/'initial-preservation-01.json')
 for key in ['preserved_production_and_checkpoint_sha256','preserved_prior_experiment_source_sha256']:
  assert all(sha256(f)==h for f,h in old[key].items())
 assert all(sha256(f)==h for f,h in p['source_sha256'].items())
 v=read(ROOT/'validation-01/result.json');assert v['gradient_validated'] and v['synthetic_native_grad_is_none'] and v['synthetic_native_optimizer_change']==0
 probe=read(ROOT/'full-routing-train-probe-01/result.json');assert probe['optimizer_updates']==0 and probe['reserved_pixels_read'] is False;ranges=read(ROOT/'current-range-audit-01.json');opt=read(ROOT/'optimizer-audit-01.json');ev=read(ROOT/'evaluation-audit-01.json');runtime=read(ROOT/'runtime-validation-01/result.json');native=read(ROOT/'native-feature-ablation-01/result.json')
 assert runtime['status']=='completed_full_range_ports_runtime_validation' and all(r['maximum_prediction_difference']<2e-6 and r['maximum_muscle_difference']<2e-6 for r in runtime['models'])
 assert native['maximum_identity_difference']<2e-6 and '17 passed' in (ROOT/'unit-tests-02.log').read_text()
 warm=[read(ROOT/f'initial-episode{i}-01/result.json') for i in [0,1]];assert sum(r['yaw_both_correct'] for r in warm)==sum(r['speed_both_correct'] for r in warm)==30
 with np.load(ROOT/'visual_only-fit-01/optimizer-trajectory.npz') as a,np.load(ROOT/'visual_native-fit-01/optimizer-trajectory.npz') as b:initial_difference=float(abs(a['before'][0]-b['before'][0,:1288]).max())
 assert initial_difference==0
 for arm,model in ev['models'].items():
  result=read(ROOT/f'{arm}-evaluation-01/result.json');model['train_pre_steering_reduction_from_warm_percent']=[100*(1-result['actual_recordings'][i]['metrics']['pre']['new']['rmse'][0]/warm[i]['metrics']['pre']['warm']['rmse'][0]) for i in [0,1]]
 feature_effects={key:max(r['groups'][key]['maximum_action_difference'] for r in native['reports']) for key in ['native_change_off','acceleration_reference','temperature_reference']}
 summary=dict(status='completed_full_range_port_learning_trial',models=ev['models'],optimizer=opt['models'],validation=v,runtime=runtime,native_feature_maximum_action_differences=feature_effects,learned_gate_current_range=ranges,local_full_routing_train_probe=probe,initial_visual_parameter_difference=initial_difference,fixed_full_routing_train_initialization=full_init,source_revision=old['source_revision'],fixed_full_routing_initialization_sha256={str(f).replace(chr(92),'/'):sha256(f) for f in Path('runs/full-routing-initialization-20261104').rglob('*.json')},real_events_per_model=32597,updates_per_model=395,actual_evaluation_events_per_model=48866,body_command_cases_per_model=45,retimed_development_image_groups_per_model=30,synthetic_image_groups_per_model=18,actor_promoted=False,actual_driving_improvement_verified=False,actual_tyre_adaptation_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active',next_work='Train a predeclared fixed gate=1 full-routing input from verified continuous TRAIN initialization; require original fine steering/pedal preservation, visual semantics and all bidirectional commands jointly. Current learned-gate models remain unqualified.')
 write_json(ROOT/'summary-01.json',summary)
 next_plan=read(p['reserved_plan']);next_plan['texture_seed']=915320137;next_plan['role']='future_unconsumed_registration_followup';write_json(ROOT/'next-reserved-plan-01.json',next_plan)
 rows=[]
 for arm,m in ev['models'].items():
  reduction=' / '.join(f'{x:+.3f}%' for x in m['steering_pre_rmse_reduction_percent']);yaw=sum(x['yaw'] for x in m['command_counts'].values());speed=sum(x['speed'] for x in m['command_counts'].values());rows.append(f"| {arm} | {reduction} | {yaw}/45 | {speed}/45 | {m['reserved_direction_counts']['fine']}/18 | {m['real_image_direction_counts']['fine']}/30 | {m['real_image_target_counts']['fine']}/30 |")
 report='# 양쪽 시각 입력의 작동점 등록과 전체 이력 학습 — 2026-11-04\n\n'+CHARTER+'\n\n'
 report+='기존 좁은 logit 보정의 한쪽 입력 제한을 넘도록 네 비행 입력 포트에 전체 범위 등록을 연결하고 실제 학습했다. gate=.01의 양쪽 초기값은 TRAIN만으로 정했고 냉간·고온 전체 연속 이력 32,597결정 및 30개 구간의 양방향 회전·속도 명령을 통과했다. 이 초기값은 공학적 작동점이며 실측 tonic/시냅스/생리학 검증이 아니다.\n\n'
 report+='공동 후보의 진입 전 조향 RMSE는 원래 fine보다 냉간 40.813%·고온 65.570%·개발 38.377% 컸다. 양방향 명령은 45/45이고 실제 영상 순서는 30/30이지만, 새 합성 영상 방향은 0/18이었다. 새 시각 등록 학습분을 제거할 때 실제 영상 margin .002 통과가 19→23/30으로 늘어 이번 학습이 그 시각 구분을 강화했다고 말할 수 없다. 두 후보는 미승격이다.\n\n'
 report+='시각 조건은 새 투사 1,280개+bias 4개+gate 4개, 공동 조건은 여기에 native 몸 투사 9,246개를 더해 총 10,534개다. 비교 조건의 용량은 같지 않다. 이전 visual_native 파일을 두 조건의 동일 부모로 사용했다. 부모 시각 codec, 화학 그래프, 원래 시각·시간 경로와 body21, 근육 생성 및 물리 부호를 고친 affine 판독기는 고정했다. native19에는 가속도·slip·load·타이어 core 온도·pressure와 마스크가 들어가며 기존 단위/등록 한계를 유지한다. 경로·지도·world pose·절대 heading·교사 조작은 actor 입력으로 넣지 않았다.\n\n'
 report+='각 조건은 자체 수치 상태로 실제 32,597결정을 순서대로 처리하고 실제 감독 301회, 같은 prefix 명령 30회, TRAIN 영상 교환 10회, 합성 영상 54회로 총 395회 갱신했다. detach는 수치 리셋이 아니며 부모 prefix 주입은 0회다. 영상 재배치/합성에서는 native gradient를 None으로 두어 Adam momentum을 포함한 native 변경이 0이었다. 독립 감사가 전체 Adam 제안, 파라미터별 투영 범위, 1,020 window의 상태 연결을 확인했다.\n\n'
 report+='투사/bias/gate/native의 Adam 학습률은 .001/.001/.005/.001이다. 한 번 갱신 norm 상한은 .0015/.002/.01/.00025, 투사 총 norm 2·bias 총 norm 2·gate .01~1·native 총 변화 .05/원소 변화 .02를 적용했다. 같은 feature의 시각 전류 갱신 상한은 약 .00881이며 차량 조향/안전 상한은 아니다.\n\n'
 report+='역전파는 실제 prefix에서 유한 차분으로 검증했다. 첫 bias 검사에서는 작은 차분이 FP32 loss 해상도에 묻혀 실패했다. 실패 소스·계획·원본 결과를 validation-roundoff-01에 보존하고 bias만 세 개의 큰 차분 간격으로 재검사했다. .005/.001에서 상대 오차 약 5.01%/4.25%였고 판정 허용치 .1은 바꾸지 않았다. 새 등록·고정 부모·엄격 로더의 17검사를 통과했다.\n\n'
 report+='최종 가중치로 조건별 세 기록 48,866결정을 처음부터 재생한 결과다. 조향 수치는 원래 fine 대비 진입 전 RMSE 감소율이며 음수는 악화다. warm 초기값 대비 TRAIN 변화도 summary에 따로 기록했다. evaluation의 initial 필드는 예전 affine 비교이므로 새로운 gate=.01 초기값으로 읽으면 안 된다.\n\n| 조건 | 조향 감소율 냉간 / 고온 / 개발 | yaw | 속도 | 새 합성 방향 | 실제 영상 순서 | 실제 영상 margin .002 |\n|---|---|---|---|---|---|---|\n'+'\n'.join(rows)+'\n\n'
 report+='같은 몸과 동일 회로 prefix에서 개발 기록의 left/right 영상을 바꿨다. 두 left·세 right의 여섯 제공 쌍을 다섯 몸/prefix에 재배치한 30조합이며 30개 독립 코너가 아니다. 각 영상은 코너 회전 onset 전에 끝난다. visual_change_off는 이번 새 등록 학습분만 warm gate=.01 값으로 되돌린 것이며 전체 시각 제거가 아니다. native_change_off도 새 native 증분만 이전 pilot으로 되돌린다. 새 합성 seed915320111은 이번에 소비했으며 가설 몸/온도 자극이다. 실제 타이어 동역학 적응의 증거로 사용하지 않는다.\n\n'
 report+='순서 검사를 통과하는 것과 조향 오차 감소·실제 주행 개선은 별도다. 모든 후보는 연구용 미승격이고 실차/AC 학생 제어 0회다. 새 트랙 분리 시험이 없어 코너 암기를 입증하거나 배제하지 못했다. 몸의 회전·속도와 회로의 과거 이력만으로 조향이 어느 정도 설명될 수 있으므로 영상 기여가 낮다는 결과를 영상이 완전히 없다는 뜻으로 읽지 않는다.\n\n'
 report+='실제 native bytes·카메라 clock을 조인한 read-only runtime 128결정씩을 검사했다. full-range 전용 로더는 부모/hash/정규화/shape/dtype/유한성/변화 범위와 미승격 표시를 확인하며 실패할 때 호출자의 몸 가중치를 바꾸지 않는다. runtime parity는 과거 자료 재생이며 live 폐루프나 한계 주행 검증이 아니다. native 가속도/온도 clamp는 새 학습 증분의 모델 내 기여만 검사했다.\n\n'
 report+='학습된 gate를 고정한 전류 바깥구간 검사도 current-range-audit-01에 기록했다. target 전류를 임의의 [0,.2]로 허용한 넓은 구간이다. 0을 제외하면 그 gate에서 집단 우세 반전은 불가능하지만, 0을 포함한다는 사실만으로 가중치 실현이나 차량 제어를 입증하지 않는다. gate 자체는 다시 학습할 수 있다.\n\n'
 report+='추가로 네 TRAIN warm prefix에서 .25/.5/.75/1의 수동 gate를 검사했다. 이 검사는 독립적인 실제 TRAIN만 사용한 국소 gradient/몸 명령 probe이고 optimizer 갱신은 0회다. 전 이력을 새 gate로 형성한 초기값 검증이나 추가 driver 학습으로 세지 않는다.\n\n'
 report+='추가 전체 검사는 별도 runs/full-routing-initialization-20261104에 보존했다. gate=1로 냉간·고온 TRAIN 전체 이력을 다시 만들고 30개 양방향 명령을 검사했다. 수치 결과는 summary의 fixed_full_routing_train_initialization에 있다. 이는 입력 등록 초기값의 검사이며 새 driver 학습은 0회다.\n\n'
 report+='다음은 아래 결과의 실패 조건을 함께 해결하는 TRAIN 회로 보정이다. 코너 진입 전의 실제 영상 의미·세 기록 조향/가감속 오차·양방향 몸 명령을 함께 보존하고, 독립 코너/트랙과 실제 폐루프로 일반화 여부를 확인해야 한다. 변속·하중 이동·접지·타이어/셋업/차종/상대풍 적응, 빠른 유효 완주와 레이싱 라인/랩타임은 여전히 미검증이다. 새 seed915320137은 예약만 했고 미생성/미소비다.\n\n'
 report+='원본은 runs/full-range-port-learning-20261104의 plan, validation, initial-episode*, 두 fit/evaluation, optimizer/evaluation 감사, native-feature-ablation, runtime-validation, summary와 source-receipt-final-01.json이다. 기존 44파일/131소스를 보존했다. 설명 HTML·simulator 변경 API·desktop 입력은 0회다. 목표는 active다.\n'
 rp=Path('docs/full-range-port-learning-20261104.md');rp.write_text(report,encoding='utf-8')
 brief='공동 후보는 원래 fine 대비 진입 전 조향 오차가 냉간 40.813%·고온 65.570%·개발 38.377% 커 실패했다. 실제 영상 순서 30/30·양방향 45/45지만 새 합성 0/18이며 새 시각 학습분의 margin 통과도 23→19/30으로 줄었다. 양쪽 작동점 gate=.01을 전체 TRAIN 이력/30명령에서 확인하고 시각/시각+native 입력을 각 395회 학습했다. 결과와 조향 감소율은 연결된 표에 기록했다. 동일 몸/prefix 영상 교환·새 합성·45개 양방향 명령을 함께 평가했고 두 후보는 미승격이다. 17검사·원본 44파일/131소스 보존·학생 제어 0회·실제 한계 주행/타이어 적응 미검증·목표 active.'
 for name in ['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md']:
  path=Path(name);text=path.read_text(encoding='utf-8');first,rest=text.split('\n',1);link=('docs/' if name=='README.md' else '')+rp.name;extra=''
  if name=='docs/CONTEXT.md':
   rest=rest.replace('최신 원본은 `runs/visual-native-port-learning-20261104/summary-01.json`','이전 visual-native 단계 당시 최신 원본은 `runs/visual-native-port-learning-20261104/summary-01.json`',1)
   extra='\n최신 원본은 `runs/full-range-port-learning-20261104/summary-01.json`과 `source-receipt-final-01.json`이다. 이번 소유 실행은 종료됐다. 기존 narrow logit의 범위 제한을 넘는 full-range 등록은 이제 학습/평가까지 했지만 후보 미승격이다. 915320111은 이번에 생성/소비했고 새915320137은 예약만 했다. gate=1 TRAIN 전 이력 초기값 검사도 별도 full-routing-initialization 원본으로 완료했고 추가 driver 학습은 0회다. 다음은 전체 개방을 고정한 입력 경로에서 조향 오차·영상 의미·양방향 명령을 함께 유지하는 학습과 새 코너/트랙 평가다.\n\n'+CHARTER+'\n'
  path.write_text(first+'\n\n**2026-11-04 전체 범위 시각 포트 학습:** '+brief+' [원본·결과]('+link+').\n'+extra+'\n아래는 이전 단계의 당시 보고다.\n'+rest,encoding='utf-8')
 sources=list(p['source_sha256'])+['src/flybrain_drive/full_range_port_candidate.py','scripts/validate_full_range_runtime.py','scripts/audit_full_range_optimizer.py','scripts/audit_full_range_evaluation.py','scripts/ablate_full_range_native_features.py','scripts/audit_full_range_current_range.py','scripts/probe_full_routing_train.py','scripts/replay_full_routing_initialization.py','scripts/finalize_full_range_ports.py','tests/test_registered_current.py','tests/test_full_range_port_candidate.py'];sources=list(dict.fromkeys(sources))
 for name in sources:
  dest=ROOT/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
 docs=['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md',str(rp)]
 try:heartbeat=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
 except (OSError,ValueError,RuntimeError) as error:heartbeat=dict(read_failed=True,error=type(error).__name__)
 lua=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');assert sha256(lua)=='0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e'
 receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),**old,source_and_documents_sha256={f:sha256(f) for f in sources+docs},current_checkpoints_sha256={str(ROOT/f'{a}-fit-01/ports.pt').replace(chr(92),'/'):sha256(ROOT/f'{a}-fit-01/ports.pt') for a in ['visual_only','visual_native']},summary_sha256=sha256(ROOT/'summary-01.json'),heartbeat_readonly=heartbeat,installed_lua_sha256=sha256(lua),process_inventory_sha256=sha256(ROOT/'process-inventory-final-01.json'),reserved_textures=dict(consumed=915320111,reserved_unread=915320137),tests_passed=17,actor_promoted=False,actual_driving_improvement_verified=False,actual_tyre_adaptation_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active',next_work='Train a predeclared fixed gate=1 full-routing input from verified continuous TRAIN initialization; require original fine steering/pedal preservation, visual semantics and all bidirectional commands jointly. Current learned-gate models remain unqualified.')
 write_json(ROOT/'source-receipt-final-01.json',receipt);print(ROOT/'summary-01.json')
if __name__=='__main__':main()
