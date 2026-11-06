"""Publish audited offline outcomes without changing deployment authority."""
import os,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from scripts.control_ac_session import read_session_heartbeat

def main():
 root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');initial=read(root/'initial-preservation-01.json');assert all(sha256(p)==h for p,h in initial['preserved_production_and_checkpoint_sha256'].items());assert all(sha256(p)==h for p,h in initial['preserved_prior_experiment_source_sha256'].items());assert all(sha256(p)==h for p,h in plan['source_sha256'].items());validation=read(root/'validation-01/result.json');assert validation['gradient_validated'] and validation['neutral_maximum_difference']==0;audit=read(root/'independent-audit-01.json');runtime=read(root/'runtime-validation-01/result.json');assert runtime['status']=='completed_joint_native_runtime_parity';assert all(v['events']==128 and v['maximum_prediction_difference']<2e-6 and v['maximum_muscle_difference']<2e-6 and v['maximum_estimated_output_body_age_seconds']<=.05 for v in runtime['models']);exchange=read(root/'real-image-exchange-01/result.json');assert exchange['status']=='completed_real_pre_corner_image_exchange' and sha256(root/'real-image-exchange-01/predictions.npz')==exchange['predictions_sha256'];xp=read(root/'real-image-exchange-plan-01.json');assert xp['program_sha256']==sha256('scripts/evaluate_real_image_exchange.py') and xp['donors']==exchange['donors']
 with np.load(root/'real-image-exchange-01/predictions.npz') as z:
  for row in exchange['reports']:
   for arm in row['arms']:
    pred=z[f"rank{row['rank']}_case{row['source']['case']}_{arm}"];margin=float(-np.mean(pred[0,:,0]-pred[1,:,0])/2);assert abs(margin-row['arms'][arm]['signed_teacher_bend_margin'])<1e-12
   assert row['identity_donor_maximum_difference']<2e-6
 assert '14 passed' in (root/'tests-01.txt').read_text(encoding='utf-8');parent=read(root/'reserved-parent-01/result.json');warm=read(root/'warm-independent-audit-01.json');fits={f'rank{k}':read(root/f'rank{k}-fit-01/result.json') for k in [1,4]};summary=dict(status='completed_joint_visual_dn_decoder_trial',parent_reserved_direction_counts=parent['hypothesized_positive_bend_pairs'],warm=warm,models=audit['models'],fits=fits,real_image_exchange=exchange,runtime=runtime,validation=validation,actor_promoted=False,actual_driving_improvement_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active',next_work='Keep original visual/body/flight goal. Diagnose command-semantic versus recorded-action conflict and port operating points; train on actual pre-corner visual dependence while preserving steering, pedals and continuous body feedback. New course/physical tyre adaptation and limit laps remain unverified.');write_json(root/'summary-01.json',summary)
 child=read(root/'one-sided-refit-01/result.json');child_audit=read(root/'one-sided-independent-audit-01.json');decomposition=read(root/'yaw-decomposition-01/result.json');assert sha256(root/'one-sided-refit-01/one-sided-head.pt')==child['checkpoint_sha256'] and child_audit['mean_bilateral_direction_passed']==6;assert sha256(root/'yaw-decomposition-01/muscle-probes.npz')==decomposition['muscle_probes_sha256'];summary.update(one_sided_child=child,one_sided_child_audit=child_audit,yaw_decomposition=decomposition,next_work='Collect bidirectional yaw/speed probes across all TRAIN pre-corner signs/body/operating states; enforce separate +/- neutral-relative response and original steering/pedal preservation. First-corner-only constraints do not generalize to development case1. Seed781191 reserved, pixels not generated.');write_json(root/'summary-01.json',summary)
 rows=[]
 for name,v in audit['models'].items():
  gains=' / '.join(f'{x:+.2f}%' for x in v['steering_pre_rmse_reduction_percent']);counts=v['reserved_direction_counts'];rows.append(f"| {name} | {gains} | {counts['fine']}/18 | {counts['calibration_off_after_prefix']}/18 | {counts['motion_off_after_prefix']}/18 | {v['yaw_direction_gate']} / {v['speed_direction_gate']} |")
 exchange_lines=[]
 for row in exchange['reports']:
  exchange_lines.append(f"| rank{row['rank']} | {row['source']['case']} | {row['arms']['fine']['signed_teacher_bend_margin']:.8f} | {row['arms']['calibration_off_after_prefix']['signed_teacher_bend_margin']:.8f} | {row['identity_donor_maximum_difference']:.3g} |")
 charter='초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.'
 report='''# 시각 하행 포트와 차량 판독 공동 학습 — 2026-11-04

'''+charter+'''

## 판정과 범위

원래 측정 비행 연결·다른 시각·몸 감각 가중치는 보존하고 공학적 시각 하행 전류 보정과 근육→차량 판독을 함께 학습했다. 실제 냉간·고온 TRAIN 기록 합계32,597결정의 연속 상태를 사용해 두 조건 각147회 갱신했다. 이 중 실제93·합성54회이며 두 실제 구간에는 후보 자신의 같은 prefix에서 yaw/speed 명령 개입 손실도 추가했다. 이후 고정 후보로 세 기록 48,866사건씩을 처음부터 재생했다. 온라인 학습 중 출력과 최종 고정 가중치 재생을 구별한다.

이번 결과는 아래 조향/영상/명령 표를 함께 판단한다. 모든 후보 actor 미적격이며 실제 조향 개선·새 트랙 일반화·생리 기능·한계 주행을 입증하지 못했다. 기존 원본 및 실패 후보는 모두 보존했다. 학생 제어/API 변경/UI 입력은 0이다.

## 구현

`visual_dn_current_calibration.py`의 새 전류 보정은 기존 영상 특징320개만 사용한다. 출력은 기존 네 비행 명령 포트로 들어가고 기존 비행 그래프와 여섯 근육 proxy를 거쳐 차량 출력에 도달한다. 새 CNN이나 영상→차량 직접 우회를 추가하지 않았다. 이것은 측정 시냅스가 아닌 공학적 보정이다.

추가 보정은 rank1(320가중치) 또는 rank4(1,280가중치)다. 기존 원래 시각 투사는 고정되어 별도로 남아 있으므로 rank1 조건 전체 감각 입력이 1차원이라는 뜻이 아니다. 차량 판독은 원래 여섯 proxy의 affine 구조21개를 유지하며 양쪽 power 각각의 gas 기울기를 양수, brake 기울기를 음수로 투영한다. raw 좌우 대칭이나 common-power 조향 불변은 이 실험의 가정이 아니다. 총 학습 파라미터341 대1,301로 수가 다르며 용량을 맞춘 비교가 아니다.

두 조건은 같은 TRAIN-only 판독 warmup에서 시작했다(1,200회, 최저 TRAIN 목적 상태 선택). 본 공동 학습은 같은 seed·스케줄·학습률의 한 번 연속 통과 후 최종 상태를 사용했고 개발/예약 영상으로 후보를 선택하지 않았다. 기존 parent 상태 주입0회, 실제 수치 이력은 보존하고 gradient만 detach한다. 합성 에피소드와 두 명령 진단 후 원래 실수치 상태를 복귀했다. 독립 감사가 두 조건 각각1,020 window의 앞뒤 수치 hash 연결과147개 학습 기록을 확인했다.

기존 head를 그대로 둔 중립 경로1,024사건 차이는0이었다. 새 전류 보정의 실제 TRAIN neural gradient는 유한 차분과 .56%/.25% 상대 차이로 일치했다. 현재 모듈/엄격 loader의14검사를 통과했다. 원래 정규화·정확한 dtype/shape·파일 hash·부호 제약·미적격 상태를 검사한다. 두 후보의 실제 native bytes/camera128사건 runtime도 별도로 재생해 고정 재생과 비교했다. 이 짧은 runtime 검증은 폐루프 주행이나 전체 기록 runtime timing 검증을 대체하지 않는다.

## 최종 고정 재생

조향 수치는 기존 fine 대비 진입 전 RMSE 감소율이며 음수는 악화다. 개발 기록은 이미 소비된 동일 코스이며 새 코스가 아니다. 새 영상 seed781183도 이번에 소비됐다. 18쌍은 합성 속도·횡위치·가설 온도 조건이며 실제 타이어 적응이 아니다.

| 조건 | 진입 전 조향 감소율 냉간 / 고온 / 개발 | 새 영상 방향 | 보정 제거 | 운동 시각 제거 | yaw / speed 평균 방향 7구간 모두 통과 |
|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

원래 부모와 판독기-only warm 기준의 새 영상 방향은 각각3/18이다. 단순 방향 순서와 목표 margin .002는 구분한다. 각 후보의 목표 margin 통과 수와 전체 가감속 RMSE, 각 명령 ±분기의 중립 대비 변화·올바른 방향 비율은 summary 및 독립 감사 원본을 따른다. 중앙차분 평균 부호만으로 양쪽 명령이 모두 정상이라고 결론 내리지 않는다.

## 실제 영상 교환

개발 기록에서 모델 출력을 보지 않고 미래 교사 조향 부호로 좌·우 코너를 구분한 뒤, yaw가 아직 작은 진입 전32결정 구간 중 영상 제공 구간의 속도 차가 가장 작은 쌍을 선택했다. 교사 부호는 진단 장면 선택과 평가에만 쓰였다. actor 입력은 영상과 원래 몸 감각뿐이다.

좌·우 제공 영상은 각각 원래 코너 yaw onset보다 마지막 결정이 약 .532/.531초 이르다. 각32결정은 독립 카메라4장에 해당한다. 제공 구간의 평균 속도는56.28/51.91m/s로 차이4.38m/s이며 완벽한 동일 장면/속도 매칭이 아니다. 동일한 수용 몸 이력·동일한 후보 회로 prefix에서 영상(scene/static/fine motion)만 함께 교환했다. 원래 영상 그대로 제공하는 identity 대조도 수행했다.

| 후보 | 수용 구간 | 좌영상 조향−우영상 조향의 기대 부호 margin | 보정 제거 margin | identity 최대 차이 |
|---|---|---|---|---|
'''+ '\n'.join(exchange_lines)+'''

양의 margin은 제공 영상의 미래 교사 코너 부호와 같은 순서다. 네 포트 후보의 차이는 보정 제거 시의 약1.87~2.48배였다. 동일한 영상 제공 한 쌍을 세 수용 몸/회로 구간에 넣었으므로 세 독립 코너 쌍을 검증한 것은 아니다. 다른 시점의 영상을 현재 몸 시계에 의도적으로 재배치했으므로 이 수치를 물리적으로 옳은 반사실 조작·새 코너 일반화·실제 운전 성공이라고 부르지 않는다. 동일 몸에서 영상이 조향을 바꾸는지와 그 방향을 제한적으로 보는 진단이다.

## 중앙차분의 허점과 양방향 제약 재학습

두 공동 후보의 yaw 중앙차분은6/7구간에서 원하는 부호였지만 양의 왼쪽 명령을 중립과 따로 비교하면 올바른 방향의 사건 비율은 모두0이었다. 두 방향이 모두 양의 조향으로 움직이면서 크기 차이만 중앙차분을 음수로 만들었다. 따라서 중앙차분 방향 통과를 양방향 제어 기능으로 쓰지 않는다.

rank4 후보의 실제 yaw 근육을7구간에서 다시 수집해 원래 고정 재생 출력과 차이0을 확인했다. 공통 power의 조향 logit 계수 합은1751.37이며 양쪽 명령 변화에 기여하지만, 이 성분을 대수적으로 빼도 양의 왼쪽 명령은 여전히 반대 방향이다. 공통 power 제거 하나로 해결된다고 결론 내리지 않는다. TRAIN 두 구간의 반대 명령에 대한 정규화 근육 변화 cosine은 약-.44/-.45로 구분 가능한 방향이다.

후속 `one-sided-refit-01`은 이미 진단을 소비한 rank4 부모를 사용한다고 명시하고 새 질감781191을 예약한 뒤, 그 부모의 전체 고정 근육 이력에서 조향7계수만 학습했다. TRAIN 두 구간의 ±명령을 각각 중립과 비교한 프레임별 비오방향 제약과 평균 logit 기울기 .005를 넣고, TRAIN 실제 조향의 inverse-tanh 가중 회귀를 convex quadratic 문제로 풀었다. 추가 회로 재생이나 시각 학습으로 세지 않는다. 가감속 계수와 PyTorch 가감속 출력은 부모와 정확히 같다.

47회 solver iteration, 양의 최소 Hessian 고유값4.73e-6, float64/float32 최대 제약 위반1.65e-17/1.08e-9를 확인했다. 독립 PyTorch 재판독과 NumPy 출력의 최대 차이는2.98e-7이다. 고정 후7구간 중6구간은 양쪽 평균 반응이 맞았지만 개발 case1은 양쪽 모두 틀렸다. 진입 전 조향 RMSE는 원래 fine보다 냉간11.35%·고온15.45%·개발11.48% 악화해 이 자식도 거절했다. 새781191 영상은 생성하거나 소비하지 않았다. 첫 코너의 TRAIN 두 구간 제약만으로 다른 코너의 작동점까지 보장할 수 없다는 증거다. 이 자식은 별도 native runtime 검증을 하지 않았으며 위128사건 검증은 rank1/rank4 공동 후보에만 해당한다.

## 근거와 다음 작업

`runs/joint-dn-decoder-20261104/`에 plan·TRAIN warmup·두 fit·두 evaluation·원래 부모의 예약 영상 반응·실제 영상 교환·독립 NumPy 감사·작동점 감사·native runtime 검증·최종 보존 영수증이 있다. NumPy는 neural muscle 기록에서 출력을 독립 재판독하고 모든 RMSE·영상 방향·명령 차분을 검산한다.

TRAIN 작동점에서 원래 네 전류 중 두 종류가 36.4%/45.3% 사건에서1e-5보다 작았다. rank4 실제 보정 출력의 네 singular value는 모두0보다 크며 실제 최대 전류 변화는 .0816~.0916이다. 따라서 추가 포트가 전혀 작동하지 않았다는 설명은 맞지 않는다. 약한 작동점이 유일한 실패 원인이라는 주장도 하지 않는다. 제한된 logit 보정의 비대칭 가용 범위와 원래 회귀/명령 의미의 충돌을 분리해 다음 학습을 결정한다.

계획의 `maximum_same_observation_steering_change=.01`은 이 offline fitter가 소비하지 않은 예약 필드다. 해당 trust bound 통과를 주장하지 않는다(`plan-field-clarification-01.json`). 실제 적용한 것은 전류 logit·가중치 범위와 power 판독 부호 투영이다.

다음은 첫 코너에 한정한 TRAIN 두 구간을 넘어 TRAIN의 모든 좌·우 코너와 몸/회로 작동점에서 후보 자신의 prefix를 수집하고, 명령 ±를 중립과 각각 비교하는 것이다. 실제 진입 전 영상 의존성과 원래 조향 회귀를 함께 학습하고 물리 명령의 방향·충분한 gain·양쪽 반응을 보존한다. 새 영상·새 코너·동기화 실측 몸 감각을 분리 평가한다. 가감속·변속·기울기·가속도·하중·접지·타이어 온도/슬립·차종/셋업·실제 상대풍 적응과 빠른 유효 완주/라인/랩타임을 포함한 전체 목표는 진행 중이다.
'''
 report_path=Path('docs/joint-visual-dn-decoder-learning-20261104.md');report_path.write_text(report,encoding='utf-8');brief='시각 하행 전류와 근육 차량 판독을 두 조건 각32,597결정/147회 공동 학습하고97,732사건 고정 재생했다. 추가4축은 새 영상3→9/18·보정 제거3/18·목표0/18이며 냉간 진입 전 조향.713% 감소/고온17.258%·개발12.480% 악화다. 실제 영상 한 쌍의 동일 몸/회로 교환에서는 기대 방향 조향 차이가 보정 제거 대비1.87~2.48배다. yaw 중앙차분의 허점을 찾아 양방향 제약 조향7계수 재학습도 했으나 개발1구간과 조향 회귀가 실패했다. 14검사·중립1024 차이0·두 native128 parity0·기존36파일/68소스 보존·모두 미승격·학생 제어0·실제 한계 주행/타이어 적응 미검증·목표active.'
 for name in ['README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md']:
  path=Path(name);text=path.read_text(encoding='utf-8');first,rest=text.split('\n',1);link=('docs/' if name=='README.md' else '')+report_path.name;extra=''
  if name=='docs/CONTEXT.md':
   rest=rest.replace('최신 원본은 `runs/paired-muscle-decoder-20261104/summary-01.json`','이전 paired 단계 당시 최신 원본은 `runs/paired-muscle-decoder-20261104/summary-01.json`',1);extra='\n최신 원본은 `runs/joint-dn-decoder-20261104/summary-01.json`과 `source-receipt-final-01.json`이다. 이번 모든 소유 프로세스는 종료됐고 네 고정 후보(기준 warm·양방향 자식 포함)는 미승격이다. 다음은 TRAIN의 여러 좌우 코너/몸 작동점에서 후보 prefix와 양방향 명령 검사를 수집해 영상 선행 판단·명령 의미·기존 조향/페달 보존을 함께 학습하는 것이다. 새781191은 예약됐으며 영상 미생성/미소비다.\n\n'+charter+'\n'
  path.write_text(first+'\n\n**2026-11-04 시각 하행·판독 공동 학습:** '+brief+' [원본·결과]('+link+').\n'+extra+'\n아래는 이전 단계의 당시 보고다.\n'+rest,encoding='utf-8')
 for p,h in plan['source_sha256'].items():
  dest=root/'executed-source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha256(dest)==h
 try:heartbeat=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
 except (OSError,ValueError,RuntimeError) as error:heartbeat=dict(read_failed=True,error=type(error).__name__)
 lua=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua');assert sha256(lua)=='0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e';files=[Path(p) for p in plan['source_sha256']]+[report_path,Path('README.md'),Path('docs/CONTEXT.md'),Path('docs/research-plan.md'),Path('docs/implementation-status.md'),Path('src/flybrain_drive/joint_dn_decoder_candidate.py'),Path('scripts/validate_joint_dn_runtime.py'),Path('scripts/validate_joint_dn_decoder.py'),Path('scripts/audit_joint_dn_decoder.py'),Path('scripts/audit_joint_decoder_warmup.py'),Path('scripts/audit_dn_current_operating_points.py'),Path('scripts/evaluate_real_image_exchange.py'),Path('scripts/finalize_joint_dn_decoder.py'),Path('tests/test_joint_dn_decoder_candidate.py')];write_json(root/'source-receipt-final-01.json',dict(observed_utc=datetime.now(timezone.utc).isoformat(),source_revision=initial['source_revision'],preserved_production_and_checkpoint_sha256=initial['preserved_production_and_checkpoint_sha256'],preserved_prior_experiment_source_sha256=initial['preserved_prior_experiment_source_sha256'],source_and_documents_sha256={str(p).replace('\\','/'):sha256(p) for p in files},summary_sha256=sha256(root/'summary-01.json'),heartbeat_readonly=heartbeat,installed_lua_sha256=sha256(lua),process_inventory_sha256=sha256(root/'process-inventory-final-01.json'),tests_passed=14,actor_promoted=False,actual_driving_improvement_verified=False,student_control_sent=False,simulator_mutation_api_calls=0,desktop_automation_calls=0,goal_status='active'));print(root/'summary-01.json')
 receipt_path=root/'source-receipt-final-01.json';receipt=read(receipt_path)
 for name in ['scripts/decompose_joint_yaw_response.py','scripts/fit_one_sided_joint_decoder.py','scripts/audit_one_sided_joint_decoder.py']:
  receipt['source_and_documents_sha256'][name]=sha256(name);dest=root/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
 receipt.update(observed_original_terminal_handles_exit0=[26261,77290,18565,75993,40181,87283,15964,75054,69933,71458,76330,7544,61690],current_checkpoints_sha256={str(p).replace('\\','/'):sha256(p) for p in [root/'warm-fit-01/warm-head.pt',root/'rank1-fit-01/joint-dn-decoder.pt',root/'rank4-fit-01/joint-dn-decoder.pt',root/'one-sided-refit-01/one-sided-head.pt']},reserved_textures=dict(consumed=781183,reserved_unread=781191),one_sided_independent_audit_sha256=sha256(root/'one-sided-independent-audit-01.json'),yaw_decomposition_result_sha256=sha256(root/'yaw-decomposition-01/result.json'),plan_field_clarification_sha256=sha256(root/'plan-field-clarification-01.json'))
 assert all(sha256(p)==h for p,h in receipt['source_and_documents_sha256'].items());write_json(receipt_path,receipt)
if __name__=='__main__':main()
