"""Publish multi-corner steering failures and verified body/command coverage."""
import os,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from scripts.control_ac_session import read_session_heartbeat
from scripts.multicorner_command_data import ROOT,collections

def main():
 plan=read(ROOT/'protocol-01.json');initial=read(ROOT/'initial-preservation-01.json');assert all(sha256(p)==h for p,h in initial['preserved_production_and_checkpoint_sha256'].items());assert all(sha256(p)==h for p,h in initial['preserved_prior_experiment_source_sha256'].items());assert all(sha256(p)==h for p,h in plan['source_sha256'].items());source=collections(plan);audit=read(ROOT/'independent-audit-01.json');runtime=read(ROOT/'runtime-validation-01/result.json');real=read(ROOT/'real-images-01/result.json');robust=read(ROOT/'robust_quadratic-fit-01/result.json');protected=read(ROOT/'protected_quadratic-fit-01/result.json');baselines=read(ROOT/'reserved-baselines-01.json');assert runtime['status']=='completed_multicorner_native_runtime_validation' and all(v['events']==128 and v['maximum_prediction_difference']<2e-6 and v['maximum_muscle_difference']==0 and v['maximum_estimated_output_body_age_seconds']<=.05 for v in runtime['models']);assert real['status']=='completed_shared_real_image_exchange' and sha256(ROOT/'real-images-01/responses.npz')==real['responses_sha256'];assert '16 passed' in (ROOT/'tests-02.txt').read_text(encoding='utf-8');assert protected['status']==robust['status']=='protected_feasible_solution_not_established';assert not (ROOT/'protected_quadratic-fit-01/decoder.pt').exists() and not (ROOT/'robust_quadratic-fit-01/decoder.pt').exists();native=read(ROOT/'native-contract-receipt-01.json');assert native['schema'].endswith('.v2') and native['assessed_acceleration_axes']==[True,True,True] and sha256(native['path'])==native['sha256']
 # Independently recompute every actual-image margin from stored predictions.
 with np.load(ROOT/'real-images-01/responses.npz') as z:
  for row in real['groups']:
   assert row['identity_muscle_difference']==0
   for arm,heads in row['arms'].items():
    for name,value in heads.items():
     pred=z[f"case{row['case']}_{arm}_{name}_prediction"];margin=float(-np.mean(pred[0,:,0]-pred[1,:,0])/2);assert abs(margin-value['signed_teacher_bend_margin'])<1e-12
 summary=dict(status='completed_multicorner_command_and_decoder_trial',models=audit['models'],parent_command_counts=audit['parent_command_counts'],body_coverage=audit['body_coverage'],reserved_baselines=baselines,real_image_exchange=real,protected_fit=protected,robust_fit=robust,runtime=runtime,native_contract=native,full_neural_events=audit['total_full_neural_events'],command_windows=45,training_windows=20,train_late_diagnostic_windows=10,development_windows=15,solver_refinement_train_windows=30,actor_promoted=False,actual_driving_improvement_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active',next_work='Jointly recalibrate existing visual and native-body flight-port boundaries under a fixed explicitly bidirectional decoder. Preserve all corner/body-state command semantics and cold/hot steering while improving actual image dependence. No decoder-only success claim. Seed915320087 reserved, pixels not generated.');write_json(ROOT/'summary-01.json',summary)
 charter='초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.';model_rows=[]
 for name,v in audit['models'].items():
  c=v['command_counts'];g=' / '.join(f'{x:+.3f}%' for x in v['steering_pre_rmse_reduction_percent']);model_rows.append(f"| {name} | {c['TRAIN']['yaw']}/20 | {c['TRAIN_diagnostic']['yaw']}/10 | {c['development']['yaw']}/15 | {g} | {v['reserved_direction_counts']['fine']}/18 |")
 image_rows=[]
 for row in real['groups']:
  image_rows.append('| '+str(row['case'])+' | '+' | '.join(f"{row['arms']['fine'][name]['signed_teacher_bend_margin']:+.7f}" for name in ['parent','affine','quadratic'])+' |')
 report='''# 여러 코너·몸 상태에서의 명령 검사와 근육 판독 학습 — 2026-11-04

'''+charter+'''

## 판정

첫 코너의 두 TRAIN 구간에 국한했던 명령 검사를 실제 기록의45구간으로 확대했다. 선형 판독기는 모든 구간에서 좌우 명령의 양쪽 평균 방향을 맞췄지만 조향 회귀와 영상 판단이 악화했다. 근육 상호작용 판독기는 고온·개발 진입 전 조향 오차를 조금 줄였으나 냉간·일부 명령·실제 영상 교환이 실패했다. **두 후보 모두 미승격이며 실제 한계 주행·온도 적응·자율 완주 개선을 입증하지 않았다.**

조건별 조향을 보호하는 추가 학습도 실제로 실행했다. 선언한2% TRAIN 개선과 모든 양방향 조건을 동시에 만족하는 해는 확인하지 못했다. 선언한 목표를 낮추거나 실패 후보를 배포하지 않았다. 학습을 진행할 수 없다는 뜻이나 수학적 불가능성 증명도 아니다.

## 보존한 구조와 새 부품

원래 rank4 시각 하행 보정 부모 `00d39c199c18df768c861632570b9627a3042d81f10a1ba37b03a718941d95dd`의 시각·몸 감각·측정 비행 연결·근육 생성 경로는 고정했다. 새 `MuscleInteractionVehicleReadout`은 여섯 근육 proxy만 받아 조향을 계산한다. 선형7계수와 선형+21개2차 곱의28계수를 비교했다. 몸 감각·지도·world pose·정답 조향·시간/코너 ID가 판독기 입력에 추가되지 않는다. 2차 곱은 자동차 몸의 출력 대응에 대한 공학적 가설이며 측정 날개 힘 법칙이 아니다.

수치 상쇄를 줄이기 위해 새 조향 판독만 float64로 계산하고 결과는 기존 float32 명령으로 돌려준다. 회로 상태는 기존 float32이며 리셋하지 않았다. 가감속은 원래 부모 판독의 계수와 출력이 정확히 같고 새로 학습하지 않았다. 따라서 페달의 방향 검사 통과나 기록 오차를 이번 가감속 개선으로 보고하지 않는다.

2차 곱의 기준·스케일은 TRAIN 두 기록의 유효5,659점에서만 산출했다. 같은 TRAIN 조향 inverse-tanh 목적·ridge1e-6·계수 범위±10에서 convex quadratic 회귀를 수행했다. 두 모델은7 대28계수로 용량이 다르다. source revision·원본·seed·설정·solver 종료 이유·가중치 hash는 각 run manifest/result에 있다. solver는 결정적이며 무작위 선택은 없었다.

## 실제 코너와 몸 감각 범위

기존 냉간·고온 TRAIN의 코너10개와 이미 소비된 동일 코스 개발의 코너5개를 사용했다. yaw onset은 기존의 몸 회전/교사 라벨 기반 진단 proxy이며 도로 기하 정답이라고 부르지 않는다. 각 onset의 -1.0초/-0.4초/+0.2초에서32결정씩을 선택했다. TRAIN 이른 진입/회전 중20구간을 학습하고 늦은 진입10구간은 첫 두 fit에서 진단만 했다. 개발15구간은 모든 fit에서 제외했다. 추가 보호 실험에서는 TRAIN 늦은10구간을 학습에 편입했다고 명시했다.

세 기록48,866결정을 각각 처음부터 연속 재생했다. 원래 부모의 전체 예측·근육과 차이는 모두0이었다. 각45구간에서 동일한 후보 회로 prefix와 동일 영상/몸을 복원해 중립·yaw±.01rad/s·속도±.01m/s를 실행했다. 중립 분기와 전체 궤적도 모두 차이0이다. 각 분기의 근육과 원래 body21/native19 값·유효성 마스크를 저장했다. 이것은 고정 기록에서의 명령 응답이며 실제 자동차의 조향→yaw 폐루프 효과가 아니다.

수집 구간의 실제 속도는 약23.56~71.63m/s, 몸 yaw는 약-.435~+.519rad/s였다. TRAIN 타이어 core 온도는 약49.69~98.67°C 범위였다. 실제 자세·가속도·하중·슬립·온도 등의 값을 원자료대로 사용하며 결측을 교사로 채우지 않았다. 가속도3축 유효성은 현재 v2 계약 `9f9b5b75…`의 대역폭 평가와 pitch/roll/speed 범위에 근거한다. 옛2축 제한과 혼동하지 않는다. 현재 수집 구간은 그3축이 모두 valid였다. 값이 다양하다는 사실은 타이어 적응이나 중력/생리 등록의 전역 타당성을 증명하지 않는다.

## 고정 판독 결과

명령 성공은 ±를 각각 중립과 비교한 평균 방향이 모두 맞는지다. 중앙차분 하나로 양쪽 반응을 대체하지 않는다. 최소 평균 logit 기울기.005는 작동 반응의 모델 guard이며 실차/실제 AC yaw 이득의 물리적 정답이 아니다. 학습 시에는 각 프레임의 반대 방향 변화도 제한했다. 프레임별 correct/wrong 비율과 실제 응답 크기는 원본을 따른다.

조향 수치는 원래 fine 대비 진입 전 RMSE 감소율이다. 음수는 악화다.

| 후보 | TRAIN 양방향 yaw | TRAIN 늦은 진입 | 개발 양방향 yaw | 조향 감소율 냉간 / 고온 / 개발 | 새 영상 방향 |
|---|---|---|---|---|---|
'''+ '\n'.join(model_rows)+'''

부모는45구간 모두 yaw의 양쪽 평균 방향을 동시에 맞추지 못했다. 속도 명령은 부모와 두 새 판독기 모두45/45에서 양쪽 평균 방향이 맞았다. 새 영상은6쌍×3가설 온도=18이며 seed781191을 이번에 소비했다. 같은 자극의 원래 fine은3/18, 공동 학습 부모는6/18이다. quadratic의 보정 제거는6/18, 운동 시각 제거는1/18이며 목표 margin.002 통과는0/18이다. 낮은 속도의 합성 장면·가설 온도는 실제 타이어 동특성이나 새 트랙 평가가 아니다.

## 실제 영상 교환과 몸 포트 개입

이전 단계와 같은 실제 좌·우 코너 영상 한 쌍을 같은 세 개발 몸/회로 prefix에서 교환했다. 제공 영상의 미래 교사 조향 부호는 장면 선택·진단 라벨에만 사용한다. 각 제공 구간은32결정·카메라4장이며 yaw onset보다 마지막 결정이 약.53초 이르다. 제공 구간의 속도 차4.38m/s가 남아 있고, 영상을 수용 몸 시계에 재배치한 진단이다. 실제로 가능한 반사실 조작의 정답이라고 부르지 않는다.

양의 margin은 제공 영상의 미래 코너 부호와 같은 순서다.

| 수용 몸/회로 구간 | 공동 학습 부모 | affine | quadratic |
|---|---|---|---|
'''+ '\n'.join(image_rows)+'''

quadratic은 두 구간에서 기대 순서를 뒤집었다. 기록의 RMSE가 조금 줄어도 영상 기반 운전이 좋아졌다고 판정하지 않은 이유다. 원래 영상을 그대로 제공하는 identity 근육 대조는 세 구간 모두 차이0이었다. 시각 전류 보정 제거와 native19 포트 추가분 제거도 prefix 이후에만 시행했다. native19 제거는 기존 body21 피드백 전체나 실제 타이어 온도 자체를 제거하는 실험이 아니다. 모든 개입의 근육과 예측을 보존했다. 이 작은 반응 차이로 온도/가속도 적응 성능을 주장하지 않는다.

## 조건별 보호 실험과 한계

`protected_quadratic-fit-01`은 모든30 TRAIN 명령 구간과 냉간/고온 all/pre 조향을 각각2% 개선하는 충분조건(logit 오차 상한)을 동시에 넣었다. solver가 조건 만족을 확립하지 못했고 checkpoint를 쓰지 않았다. solver 실패를 불가능성 증명으로 해석하지 않는다.

`robust_quadratic-fit-01`은 선형 방향 제약의 feasible 시작점에서 먼저 quadratic 회귀를 풀고, 실제 tanh 조향의 최악 조건별 MSE 비율을 직접 최적화했다. 모든30 TRAIN 방향 제약과 solver 수렴은 확인했지만 최악 조건의 RMSE는 원래의1.0110배였다. 고온 진입 전은.9617배였다. 선언한.98배 목표를 만족하지 못해 이 실험도 checkpoint를 쓰지 않았다. 실제 비선형 문제의 전역 최적성이나 목표 불가능성은 증명하지 않았다. 이 추가 학습들은 개발/예약 영상 데이터를 읽지 않았다. 후속 새 seed915320087은 예약만 했고 영상은 생성하지 않았다.

## 검증 원본과 다음 작업

원본은 `runs/multicorner-command-20261104/`의 protocol·세 episode collection·두 fit/evaluation·reserved stimuli/response·real-images·independent audit·runtime validation·두 보호 실험·최종 영수증이다. 독립 NumPy가 전체 예측·RMSE·명령 방향·학습 부등식·영상 margin을 검산했다. 새 module/엄격 loader16검사를 통과했다. 실제 native bytes와 같은 camera/body 결합의128결정 runtime을 두 판독기로 실행했고 근육 차이0, 출력 최대 차이는1.79e-7이었다. 이 짧은 runtime 검증은 폐루프나 전체 구간 timing 검증을 대체하지 않는다.

`yaw_responsive_frames`는 evaluator에 남은 미사용0 필드이며 측정 응답 수가 아니다. 감사한 total/yaw/speed와 각 분기의 correct/wrong 비율을 따른다(`evaluation-field-clarification-01.json`).

판독기만 바꾸면 명령 의미와 영상 의미가 서로 어긋날 수 있다는 근거가 생겼다. 다음은 명령의 양쪽 의미를 고정한 판독기 아래에서 기존 시각·native 몸 감각의 비행 포트 경계를 함께 보정하는 것이다. 실제 TRAIN 조건별 조향/페달 보존과 여러 코너의 양방향 명령·영상 선행 판단을 동시에 확인한다. 합성 몸 가설로 실제 타이어 적응을 학습했다고 부르지 않는다. 가감속·변속·기울기·가속도·하중·접지·타이어/셋업/차종·실제 상대풍 적응·빠른 유효 완주·라인/랩타임은 전체 목표로 계속 남는다.

기존40파일/96소스와 과거 성공·실패를 보존했다. 이번 학생 제어·simulator API 변경·desktop 입력은 모두0이며 모든 소유 프로세스는 종료됐다. 목표는 active다.
'''
 report_path=Path('docs/multicorner-command-and-muscle-interaction-20261104.md');report_path.write_text(report,encoding='utf-8');brief='실제15코너×3시점45구간에서 같은 prefix/영상/몸의 양방향 yaw·speed를 수집했고48,866사건 원본 parity0을 확인했다. 선형 yaw45/45는 조향·영상 악화, 근육 상호작용 yaw43/45는 고온1.989%·개발1.085% 개선/냉간3.748% 악화·새 영상7/18·실제 영상 교환2구간 역방향으로 두 후보 미승격이다. 모든30 TRAIN 명령과 조건별 조향2% 보호를 함께 만족하는 해도 미확립이다. 16검사·두 native128 최대차1.79e-7·기존40파일/96소스 보존·학생 제어0·실제 타이어 적응/한계 주행 미검증·목표active.'
 for name in ['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md']:
  path=Path(name);text=path.read_text(encoding='utf-8');first,rest=text.split('\n',1);link=('docs/' if name=='README.md' else '')+report_path.name;extra=''
  if name=='docs/CONTEXT.md':
   rest=rest.replace('최신 원본은 `runs/joint-dn-decoder-20261104/summary-01.json`','이전 joint 단계 당시 최신 원본은 `runs/joint-dn-decoder-20261104/summary-01.json`',1);rest=rest.replace('새781191은 예약됐으며 영상 미생성/미소비다.','당시 새781191은 예약/미소비였으며 이후 multicorner 단계에서 소비됐다.',1);extra='\n최신 원본은 `runs/multicorner-command-20261104/summary-01.json`과 `source-receipt-final-01.json`이다. 이번 실행은 모두 종료됐고 두 실제 가중치 후보는 미승격·두 보호 실험은 조건 만족 checkpoint를 쓰지 않았다. 다음은 양방향 명령과 실제 영상 의미를 함께 보존하는 시각/native 몸 감각 포트 공동 보정이다. 새915320087은 예약만 했으며 미생성/미소비다.\n\n'+charter+'\n'
  path.write_text(first+'\n\n**2026-11-04 여러 코너·몸 상태의 양방향 명령 학습:** '+brief+' [원본·결과]('+link+').\n'+extra+'\n아래는 이전 단계의 당시 보고다.\n'+rest,encoding='utf-8')
 for p,h in plan['source_sha256'].items():
  dest=ROOT/'executed-source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha256(dest)==h
 files=['src/flybrain_drive/muscle_interaction_vehicle_readout.py','src/flybrain_drive/muscle_interaction_candidate.py','scripts/multicorner_command_data.py','scripts/prepare_multicorner_command.py','scripts/collect_multicorner_command.py','scripts/collect_multicorner_visual_response.py','scripts/fit_multicorner_decoder.py','scripts/evaluate_multicorner_decoder.py','scripts/fit_protected_multicorner_decoder.py','scripts/fit_robust_multicorner_decoder.py','scripts/collect_multicorner_real_images.py','scripts/audit_multicorner_results.py','scripts/validate_multicorner_runtime.py','scripts/finalize_multicorner_command.py','tests/test_muscle_interaction_vehicle_readout.py','tests/test_muscle_interaction_candidate.py','README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md',str(report_path)]
 for name in files:
  if name.startswith(('src/','scripts/','tests/')):
   dest=ROOT/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
 try:heartbeat=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
 except (OSError,ValueError,RuntimeError) as error:heartbeat=dict(read_failed=True,error=type(error).__name__)
 lua=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');assert sha256(lua)=='0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e';receipt=dict(observed_utc=datetime.now(timezone.utc).isoformat(),source_revision=initial['source_revision'],preserved_production_and_checkpoint_sha256=initial['preserved_production_and_checkpoint_sha256'],preserved_prior_experiment_source_sha256=initial['preserved_prior_experiment_source_sha256'],source_and_documents_sha256={name:sha256(name) for name in files},current_checkpoints_sha256={str(p).replace('\\','/'):sha256(p) for p in [ROOT/'affine-fit-01/decoder.pt',ROOT/'quadratic-fit-01/decoder.pt']},summary_sha256=sha256(ROOT/'summary-01.json'),heartbeat_readonly=heartbeat,installed_lua_sha256=sha256(lua),process_inventory_sha256=sha256(ROOT/'process-inventory-final-01.json'),observed_original_terminal_handles_exit0=[67233,18086,84627,6675,21798,71968,25338,52433,24731,27156,24870,58425,81454,14887],tests_passed=16,reserved_textures=dict(consumed=781191,reserved_unread=915320087),actor_promoted=False,actual_driving_improvement_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active');assert all(sha256(p)==h for p,h in receipt['source_and_documents_sha256'].items());write_json(ROOT/'source-receipt-final-01.json',receipt);print(ROOT/'summary-01.json')
if __name__=='__main__':main()
