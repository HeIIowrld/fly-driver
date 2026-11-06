"""Publish bounded port learning, range failure and untrained registration prototype."""
import os,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from scripts.control_ac_session import read_session_heartbeat

ROOT=Path('runs/visual-native-port-learning-20261104')
def main():
 plan=read(ROOT/'plan-01.json');initial=read(ROOT/'initial-preservation-01.json');assert all(sha256(p)==h for p,h in initial['preserved_production_and_checkpoint_sha256'].items());assert all(sha256(p)==h for p,h in initial['preserved_prior_experiment_source_sha256'].items());assert all(sha256(p)==h for p,h in plan['source_sha256'].items());optimizer=read(ROOT/'optimizer-audit-01.json');evaluation=read(ROOT/'evaluation-audit-01.json');runtime=read(ROOT/'runtime-validation-01/result.json');feature=read(ROOT/'native-feature-ablation-01/result.json');ranges=read(ROOT/'reachable-range-audit-01.json');registration=read(ROOT/'registration-validation-01/result.json');descent=read(ROOT/'registration-descent-probe-01/result.json');grid=read(ROOT/'registration-tonic-grid-01/result.json');validation=read(ROOT/'validation-01/result.json');assert validation['gradient_validated'] and registration['gradient_validated'] and registration['neutral_maximum_difference']==0;assert runtime['status']=='completed_visual_native_ports_runtime_validation' and all(v['maximum_prediction_difference']==0 and v['maximum_muscle_difference']==0 and v['events']==128 for v in runtime['models']);assert feature['maximum_identity_difference']==0 and ranges['cannot_cross_zero_opponent_interval_fraction']==1.;assert '17 passed' in (ROOT/'tests-03.txt').read_text(encoding='utf-8');assert descent['feasible_nonzero_gate'] and descent['observed_descent'];assert not registration['new_driver_trained'] and registration['registration_optimizer_updates']==0;native_maxima={key:max(row['groups'][key]['maximum_action_difference'] for row in feature['reports']) for key in ['native_change_off','acceleration_reference','temperature_reference']}
 # Matched visual initialization and explicit native-only treatment.
 with np.load(ROOT/'visual_only-fit-01/optimizer-trajectory.npz') as v,np.load(ROOT/'visual_native-fit-01/optimizer-trajectory.npz') as n:initial_visual_difference=float(abs(v['before'][0]-n['before'][0,:1280]).max())
 assert initial_visual_difference==0;next_reserved=read(plan['reserved_plan']);next_reserved['texture_seed']=915320111;next_reserved['role']='future_full_range_registration_trial_reserved_only';write_json(ROOT/'next-reserved-plan-01.json',next_reserved)
 summary=dict(status='completed_visual_native_port_learning_and_registration_component_trial',models=evaluation['models'],optimizer=optimizer['models'],validation=validation,runtime=runtime,native_feature_maximum_action_differences=native_maxima,reachable_range=ranges,registration_component=registration,registration_bootstrap=descent,registration_tonic_grid_result_sha256=sha256(ROOT/'registration-tonic-grid-01/result.json'),paired_initial_visual_parameter_difference=initial_visual_difference,real_events_per_model=32597,updates_per_model=395,training_real_windows=301,training_command_windows=30,training_real_image_exchanges=10,training_synthetic_exchanges=54,actor_promoted=False,actual_driving_improvement_verified=False,actual_tyre_adaptation_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active',next_work='Calibrate bilateral nonzero registration operating point using TRAIN-only neutral/command tests and full continuous replay, then train full-range visual/native ports under fixed signed decoder. Preserve actual cold/hot steering/pedals, new image dependence and all45 bidirectional cases. New codec is component-tested, not trained. Seed915320111 reserved only.');write_json(ROOT/'summary-01.json',summary)
 charter='초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.';rows=[]
 for arm,v in evaluation['models'].items():
  reduction=' / '.join(f'{x:+.3f}%' for x in v['steering_pre_rmse_reduction_percent']);rows.append(f"| {arm} | {reduction} | {v['command_counts']['TRAIN']['yaw']+v['command_counts']['development']['yaw']}/45 | {v['reserved_direction_counts']['fine']}/18 | {v['real_image_direction_counts']['fine']}/30 | {v['real_image_target_counts']['fine']}/30 |")
 report='''# 시각·native 몸 감각 포트 공동 학습과 입력 작동점 제한 — 2026-11-04

'''+charter+'''

## 판정

기존 비행 회로에 들어가는 시각 전류와 native 몸 감각 포트를 실제로 학습했다. 시각만 보정하는 조건과 시각+몸 조건을 비교하고 두 조건 모두45구간의 양방향 명령 평균 방향을 유지했다. 그러나 원래 fine 모델의 조향 오차를 회복하지 못했고 새 합성 영상 방향도0/18이었다. **두 후보 미승격이며 실제 자율 주행·한계 주행·타이어 적응 개선은 입증하지 않았다.**

추가 검사에서 기존 보정기는 이번 합성 영상 후반192샘플 모두에서 DN 집단 전류의 한쪽 우세를 뒤집을 수 없는 범위임을 확인했다. 이는 임의 가중치에 대해서도 성립하는 입력 범위 제한이다. 전체 실패의 유일 원인이나 차량의 제어 불가능성을 증명한 것은 아니다. 이 제한을 넘어설 새 입력 등록 부품도 구현·검사했지만 **아직 새 부품을 학습한 운전 모델은 없다.**

## 무엇을 고정하고 학습했는가

시작점은 이전 공동 학습의 시각 보정 부모 `00d39c19…`와 여러 코너에서 양방향 의미가 맞았던 affine 근육 판독기 `f555048d…`다. 이 판독기는 이미 원래 fine보다 조향 기록 오차가 컸던 미적격 연구 후보이며 기존 운전자를 교체한 것이 아니다. 새 학습 전/후와 원래 fine 대비를 구분한다.

측정 비행 화학 연결·원래 시각/시간 경로·legacy body21 경로·근육 생성·판독기 전체를 고정했다. 시각 조건은 기존4포트 전류 보정1,280개만, 공동 조건은 여기에 기존 native19 투사의201×46=9,246개를 더해 총10,526개를 학습했다. 용량이 같은 비교가 아니므로 성능 차이만으로 몸 감각의 일반적 우월성이나 생리를 주장하지 않는다. 네 시각 포트와201 몸 포트의 대응은 공학적 학습 경계이며 새 측정 시냅스가 아니다.

몸 입력은 기존 v2 평가 계약에 따른 가속도·slip·load·tyre core 온도·pressure와 유효성 마스크다. 단위/등록 한계는 원래 계약을 유지한다. 차체 pitch/roll·회전·속도 등의 legacy 몸 되먹임도 유지한다. native19에는 정답 경로·world pose·절대 heading·교사 명령이 없다. 새 조향은 여전히 비행 그래프가 만든 여섯 근육 proxy를 통해 나온다. 판독기 계수는 고정했지만 근육이 바뀌므로 가속·제동 예측값은 달라질 수 있다.

## 학습과 검증 방법

각 조건은 실제 AC 기록32,597결정을 시간순으로 처리했다. 고정32결정 블록 중301개를 실제 감독 학습에 사용하고 나머지도 수치 회로 상태를 전진시켰다. 코너10개의 세 시점에서 후보 자신의 온라인 prefix로 중립/yaw±/speed±를 각각 실행해30회 학습했다. 이 prefix는 과거 갱신 당시 형성된 후보 상태이며 현재 가중치로 만든 정지 궤적이나 원래 부모 상태라고 부르지 않는다. 모든 분기는 동일 prefix·현재 가중치·같은 몸/영상에서 비교하고 실제 궤적의 수치 상태를 정확히 복귀했다. gradient detach는 수치 리셋이 아니다.

TRAIN의 실제 코너 영상 교환10회와 기존 합성 전 이력54회도 시각 학습에 사용했다. 총395회 갱신이다. 영상 교환은 같은 몸의 시계에 다른 실제 영상을 재배치한 순서 학습이며 물리적으로 가능한 반사실 차량 조작 정답이 아니다. 교사 부호는 장면 선택과 손실에만 쓰고 actor 입력에 넣지 않았다. 합성 몸/영상 재배치에서는 native projection의 gradient를 None으로 두어 Adam momentum까지 갱신하지 않았다. 독립 감사가 해당64회 모두 native 변경0을 확인했다.

시각/몸 한 번 갱신의 Frobenius norm 상한은.001/.00025, 원래 대비 총 변화 상한은.1/.01이었다. 둘 다 총 상한에 도달했다. native 인코딩과 tanh의 전역 기울기 상한으로 한 번 native 전류 변화의 벡터 norm은 약.00463 이하가 된다. 이것은 코드상 입력 전류 변화 범위이며 차량 조작 변화나 안전성을 보장하는 수치가 아니다. 원래 정규화·마스크·단위는 학습하지 않았다.

학습 전1,024결정 차이는 최대1.79e-7이었고 실제 시각/native gradient는 유한 차분과 일치했다(최소 상대 차이 약1.29%/.19%). native gradient를 먼저 실제 자료로 갱신한 뒤에도 합성/영상 재배치 optimizer 단계에서 변경0임을 별도 확인했다. 최종395단계 Adam 제안과 투영 범위, 각1,020 window의 수치 hash 연결을 독립 감사했다. 두 조건의 초기 시각 파라미터 차이는0이다.

## 최종 고정 가중치 결과

각 모델은 세 원본 기록48,866결정을 처음부터 다시 재생했다. 조향 수치는 원래 fine 대비 진입 전 RMSE 감소율이며 음수는 악화다.

| 조건 | 조향 감소율 냉간 / 고온 / 개발 | 양방향 yaw 평균 방향 | 새 합성 영상 방향 | 실제 영상 순서 | 실제 영상 목표 margin |
|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

속도 명령의 gas−brake proxy도 두 조건 모두45/45에서 양쪽 평균 방향이 맞았다. 이 방향 검사와 각 프레임의 반응 크기·실제 차량의 가속/yaw 이득 검증은 다르다. 학습 전 signed affine 시작점의 조향 감소율은 냉간-52.600%·고온-81.948%·개발-47.850%였다. 몸 보정은 이 나쁜 시작점 대비 오차를 약1% 줄였지만 원래 fine을 넘지 못했다. 시각만 학습한 조건보다도 약.8~1.0%의 작은 차이이며 미승격 판정을 바꾸지 않는다.

조향/가속/제동의 전체 및 진입 전 모든 축 RMSE는 summary와 evaluation-audit 원본에 있다. 예를 들어 공동 조건의 원래 fine 대비 전체 가속/제동 RMSE 감소율은 냉간 약3.39%/9.40%, 고온5.73%/9.24%, 개발6.57%/9.25%다. 이는 기록의 조작 예측 오차이며 실제 가감속·랩타임 개선이 아니다. 진입 전 가속 오차는 일부 악화했고 조향 실패도 남는다.

새 합성 seed915320087은6영상쌍×3가설 온도로 평가했고 이번에 소비했다. 실제 온도 적응 시험이 아니다. 개발 실제 영상은 두 left·세 right 제공 구간의6쌍을 다섯 수용 몸/회로 상태에서 교환한30조합이다. 30독립 코너/트랙이나 물리적 반사실 정답이 아니다. 각 제공 구간은32결정·약4카메라이며 yaw onset보다 마지막 결정이 약.73초 이르다. 두 모델 모두 기대 순서30/30이지만, 시각 학습분을 제거하면 목표 margin.002 통과 수는23→18/30이다. 이 제거는 새로운 시각 보정만 부모 값으로 되돌리고 나머지 시각과 prefix 이력은 유지한다.

## native 학습분의 인과적 기여

시각만 학습한 대조군은 native 학습분을 제거해도 합성/실제 영상 예측 차이가 정확히0이었다. 공동 조건에서는 차이가 각각 최대.001560/.001176이었다. 학습분을 제거하는 것은 기존 body21/native 피드백 전체를 제거하는 실험이 아니다.

45개 실제 구간에서 원래 native 피드백을 보존하고 새로 배운 증가분만 제거하거나, 그 증가분의 가속도/온도 입력만 TRAIN 기준값으로 바꿨다. 행동 차이 최대값은 전체 학습분 제거.001611, 가속도 기준값.000672, 온도 기준값.000350이었다. 같은 입력 그대로 조합한 identity 대조는0이다. 새 몸 경계가 실제로 입력에 반응함을 보여주지만, 효과는 작고 타이어 물성 적응·기울기 복원·한계 주행 성공의 증거가 아니다. 기준값 개입은 명시적 모델 진단이며 결측 센서를 교사로 복원한 것이 아니다.

## 기존 DN 보정기의 도달 범위

`reachable-range-audit-01.json`은 원래 DN 전류에 최대±2 logit 보정만 더하는 codec의 범위를 계산했다. 새 합성 영상 후반192샘플에서 두 종류 포트는 모두1e-5 미만이고, 허용 보정 전체를 적용해도 양·음 orientation 집단의 평균 전류 차이는 약[-.1980,-.1253]의 음수 범위였다. 따라서 임의의 보정 가중치로도 이번 샘플의 입력 집단 우세 방향을 반전할 수 없다. 더 큰 가중치 norm만 허용해도 logit 상한이 그대로면 이 제한은 남는다.

이는 입력 전류 범위에 대한 분석적 상한이며 전체 조향/차량의 제어 불가능성이나 실패의 유일 원인 증명이 아니다. 다른 시각/몸 경로와 전류 기준점도 작용한다. 그러나 닫힌 쪽 포트를 등록할 수 있는 작동점부터 확인해야 한다는 구체적 후속 근거다.

## 새 전체 범위 등록 부품 — 실행 검사와 학습을 구분

`FullRangeDNRegistration`을 구현했다. 기존 전류와 새 bounded 시각 투사의 convex 혼합으로 네 포트를 기존 계약0..0.2 안에서 독립적으로 등록한다. gate0은 부모를 정확히 유지하고 gate1은 기존 낮은 쪽 집단도 열 수 있다. 화학 연결·측정 구조나 새 body 센서를 추가하지 않았다. 이것은 공학적 작동점 등록 가설이며 생리학적 tonic을 실측했다는 뜻이 아니다.

17검사와 새 부품의1,024결정 중립 차이0을 확인했다. gate0의 투사 gradient가0인 단계 구조를 명시했다. 수동 gate.05에서 gate/투사 gradient는 유한 차분과 약.81%/1.34% 최소 상대 차이로 일치했다. 실제 유효 TRAIN 라벨 한 구간의 투영된 작은 gate 이동은 손실을 줄였지만, 처음에는 원래 낮은 두 포트의 gate gradient가0이었다.

네 TRAIN 작동점에서 수동 gate0/.005/.01/.02/.05를 검사했다. .005에는 일부 포트 투사 gradient가0이었고 .01 이상에서는 이 네 구간의 네 투사 모두 비영이었다. 이것은 기존 후보 prefix에서의 국소 검사다. 새 tonic으로 전체 궤적을 형성했을 때의 중립 안정성이나 모든 코너의 적합성을 보장하지 않는다. 다음 학습 전에는 이 비영 시작 작동점을 연속 재생·양방향 반응과 함께 교정해야 한다.

새 부품의 optimizer 학습은0회이고 새 driver checkpoint도 쓰지 않았다. 이번 두 학습 후보와 새 등록 부품을 혼동하지 않는다. 부품 실행/gradient 검증은 생리 기능·통합 차량 제어·자율 완주를 대체하지 않는다.

## 원본과 다음 작업

원본은 `runs/visual-native-port-learning-20261104/`의 plan·validation·두 fit/evaluation·optimizer/evaluation 감사·native-feature ablation·reachable-range audit·registration validation/probe/grid·runtime validation·최종 영수증이다. 독립 NumPy 재판독의 예측 차이는 최대 약3.2e-7이었다. 실제 native bytes/camera를 이용한128결정 runtime은 두 모델 모두 근육/출력 차이0이었으나, 이는 과거 자료의 read-only 실행이며 live 폐루프/전체 시간 보장 검증은 아니다.

다음은 TRAIN에서 양측 입력의 시작 작동점을 등록하고 전체 이력을 다시 형성한 뒤, 새 입력 경계와 native 몸 포트를 공동 학습하는 것이다. 기존 화학 연결과 정상 방향 판독기를 보존하면서 조향·가감속·여러 몸 상태의 명령·실제 영상 선행 판단을 함께 평가한다. 변속·하중/접지·타이어/셋업/차종·실제 상대풍 적응·빠른 유효 완주·레이싱 라인/랩타임을 포함한 전체 목표는 여전히 미완료다.

기존42파일/112소스와 원자료·성공·실패를 보존했다. 이번 학생 제어·simulator API 변경·desktop 입력은 모두0이며 모든 소유 프로세스는 종료됐다. 새915320111은 예약만 했고 영상은 미생성/미소비다. 목표는 active다.
'''
 report_path=Path('docs/visual-native-port-learning-and-operating-range-20261104.md');report_path.write_text(report,encoding='utf-8');brief='시각/시각+native 몸 포트를 각32,597결정·395회 학습하고97,732사건 고정 재생했다. 두 조건 모두양방향45/45·실제 영상순서30/30이지만 원래 대비 조향 크게 악화·합성0/18로 미승격이다. native의 가속도/온도 학습 기여는 작게 확인했다. 기존 logit 보정은 새 영상192샘플 전부에서 DN 집단 우세 방향 반전이 불가능했다. 전체 범위 등록 부품은17검사·중립1024 차이0·국소 gradient를 확인했으나 미학습이다. 원본42파일/112소스 보존·학생 제어0·실제 한계 주행/타이어 적응 미검증·목표active.'
 for name in ['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md']:
  path=Path(name);text=path.read_text(encoding='utf-8');first,rest=text.split('\n',1);link=('docs/' if name=='README.md' else '')+report_path.name;extra=''
  if name=='docs/CONTEXT.md':
   rest=rest.replace('최신 원본은 `runs/multicorner-command-20261104/summary-01.json`','이전 multicorner 단계 당시 최신 원본은 `runs/multicorner-command-20261104/summary-01.json`',1);rest=rest.replace('새915320087은 예약만 했으며 미생성/미소비다.','당시 새915320087은 예약/미소비였고 이후 visual-native 단계에서 소비됐다.',1);extra='\n최신 원본은 `runs/visual-native-port-learning-20261104/summary-01.json`과 `source-receipt-final-01.json`이다. 이번 모든 실행은 종료됐고 두 학습 후보는 미승격이다. 새 full-range DN 등록 부품은 실행/gradient만 검사했고 새 driver 학습0회다. 다음은 TRAIN의 양측 비영 작동점을 전체 연속 이력에서 검증한 뒤 새 등록 경계/native 몸 포트를 학습하는 것이다. gate0에서 낮은 쪽 gate gradient가0일 수 있으므로 무조건0초기화만으로 양쪽이 열릴 것으로 가정하지 않는다. 새915320111은 예약만 했으며 미생성/미소비다.\n\n'+charter+'\n'
  path.write_text(first+'\n\n**2026-11-04 시각·native 몸 포트 학습과 입력 범위 제한:** '+brief+' [원본·결과]('+link+').\n'+extra+'\n아래는 이전 단계의 당시 보고다.\n'+rest,encoding='utf-8')
 files=['scripts/visual_native_port_experiment.py','scripts/prepare_visual_native_ports.py','scripts/validate_visual_native_ports.py','scripts/fit_visual_native_ports.py','scripts/evaluate_visual_native_ports.py','src/flybrain_drive/visual_native_port_candidate.py','scripts/audit_visual_native_optimizer.py','scripts/audit_visual_native_evaluation.py','scripts/validate_visual_native_runtime.py','scripts/ablate_learned_native_features.py','scripts/audit_visual_port_reachable_range.py','src/flybrain_drive/full_range_dn_registration.py','tests/test_port_weight_projection.py','tests/test_visual_native_port_candidate.py','tests/test_full_range_dn_registration.py','scripts/validate_full_range_dn_registration.py','scripts/probe_registration_gate_descent.py','scripts/probe_registration_tonic_grid.py','scripts/finalize_visual_native_ports.py','README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md',str(report_path)]
 for name in files:
  if name.startswith(('src/','scripts/','tests/')):
   dest=ROOT/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
 try:heartbeat=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
 except (OSError,ValueError,RuntimeError) as error:heartbeat=dict(read_failed=True,error=type(error).__name__)
 lua=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');assert sha256(lua)=='0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e';receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),source_revision=initial['source_revision'],preserved_production_and_checkpoint_sha256=initial['preserved_production_and_checkpoint_sha256'],preserved_prior_experiment_source_sha256=initial['preserved_prior_experiment_source_sha256'],source_and_documents_sha256={name:sha256(name) for name in files},current_checkpoints_sha256={str(p).replace('\\','/'):sha256(p) for p in [ROOT/'visual_only-fit-01/ports.pt',ROOT/'visual_native-fit-01/ports.pt']},summary_sha256=sha256(ROOT/'summary-01.json'),next_reserved_plan_sha256=sha256(ROOT/'next-reserved-plan-01.json'),reserved_textures=dict(consumed=915320087,reserved_unread=915320111),heartbeat_readonly=heartbeat,installed_lua_sha256=sha256(lua),process_inventory_sha256=sha256(ROOT/'process-inventory-final-01.json'),observed_original_terminal_handles_exit0=[26417,65455,16746,47635,36983,40238,27698,75485,4431,16849,88086,50294],tests_passed=17,registration_optimizer_updates=0,new_registration_driver_trained=False,actor_promoted=False,actual_driving_improvement_verified=False,actual_tyre_adaptation_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active');assert all(sha256(p)==h for p,h in receipt['source_and_documents_sha256'].items());write_json(ROOT/'source-receipt-final-01.json',receipt);print(ROOT/'summary-01.json')
if __name__=='__main__':main()
