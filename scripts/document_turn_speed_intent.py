"""Markdown report for split visual intents and the failed vehicle-decoder sign gate."""
from pathlib import Path
from flybrain_drive.sac_unified_student import read

def main():
 root=Path('runs/visual-turn-speed-20261104');r=read(root/'summary-01.json');text='''# 시각 조향·속도 의도 분리와 차량 판독 부호 반전 — 2026-11-04

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

**새 경로는 계산·학습·원본 재생을 완료했지만 정상적인 차량 속도 제어로 판정하지 않는다.** 고정 부모의 네 TRAIN 구간에서 높은 속도 요청→modeled power_L/R 증가→차량 가속 감소/제동 증가라는 반전을 확인했다. 기존 차량 판독기의 부호가 원인이 되는 직접 경로를 분리했고, 두 후보는 미승격이다. 학생 제어·시뮬레이터 변경/API/UI 조작은0이다. 실제 주행/한계 주행·생리 기능·새 실제 코너 검증은 미완료다.

## 새 시각·몸 감각 경로

기존320특징→1값→네 turn 입력을 보정하면서, 별도의320특징→속도 보정 입력을 추가했다. 현재 native19(가속도·타이어 slip/하중/온도/압력과 각각의 유효성 mask)는38개 계수를 통해 **양의 gain**만 조절한다. 새 속도 보정은 `10*tanh(retinal_projection*gain)` m/s이고 gain은 exp(±.5) 범위다. retinal projection이0이면 몸 감각과 관계없이 보정0이다. 몸 상태만으로 새 시각 속도 명령을 만들거나 그 부호를 뒤집지 못한다.

원하는 속도는 현재 몸 속도+보정을0~외부 racing ceiling으로 제한한다. 그 뒤 기존 `ManeuverCurrents`가 현재 몸 속도를 한 번 빼서 기존 비행 power 포트에 전류를 넣는다. 직접 차량 페달 출력 경로를 추가하지 않았다. 시각 손실/몸 속도 결측은 거부하고, 부분 native 결측은 mask로 표시한다. native 전체 결측이면 gain1이다. 정상 입력을 oracle로 보충하지 않는다.

이 경로는 공학적 차량 대응 가설이며 측정 fly tuning/마찰계수/최적 타이어 온도/고차 뇌를 구현한 것으로 부르지 않는다. 기존 측정 비행 그래프·신경 동역학·몸 안정화·다른 시각·차량 근육 판독과 정규화는 모두 고정했다. actor에는 영상과 허용된 몸 감각만 주며 위치·지도·교사 행동·코너 label은 손실/평가에만 쓴다.

합성 코너에는 assessed native sample이 없으므로 **새 속도 모듈의 native mask는 모두 invalid, gain1, body-gain gradient0**으로 두었다. 기존 고정 회로의 factorial 몸 입력은 가설 하중/휠/기준 압력 fixture다. 실제 타이어 센서 증거가 아니다. Adam의 body-gain momentum은 앞선 실제 gradient에서 이어질 수 있다. 실제 기록은 원본 native byte와 기존 assessed 유효성 계약을 그대로 사용했고, 학습 입력과 재생 입력의 native 값/mask가 정확히 같음을 감사했다.

## 검증과 학습 폭

- 새 모듈/후보 로더/기존 시각·몸 경계35검사 통과. 카메라/속도 결측 거부, native invalid 값 무시, 몸 감각만으로 속도 의도 생성 금지, 양의 gain/속도 범위/정규화 보호를 검사했다.
- 새 모듈이 중립인 원본 TRAIN1024결정과 새 native/camera 실행 경계128결정에서 원래 출력과 최대차0이다. 전체 실제 주행이나 생리 기능의 증거는 아니다.
- 초기 큰 speed 가중치 차분(.001/.0005)은 비선형/초과오차 보호 손실 때문에 gradient와 심하게 달랐다. 실패 결과를 보존했다. 같은 TRAIN 점에서 작은1e−5/2e−6 차분은 상대차 .0927%/.2794%, baseline 재시작 차이0이었다. 한 점/방향 검증이며 전역 gradient 정확성 보장은 아니다.
- turn 학습률 .001, speed1e−6, body-gain1e−4를 사용했다. speed/body-gain 업데이트 L2는 각2e−5/1e−3 이내로 제한하고 전체 가중치 경계도 적용했다. 고정 관측에서 목표 속도 변화의 계산상 상한은 .141m/s 미만이다. 실제 가속도·전체 recurrent 출력·다음 관측 변화의 상한이 아니다.

두 조건은 같은 원래 fine 부모 fc4a2997…·seed781149·실제32,597결정·선택93구간+합성54구간·147업데이트·손실을 사용했다. 후보 자신의 수치 상태를 연속 유지하며 gradient만 절단했고, 별도 합성 에피소드 뒤 실제 상태를 정확히 복원했다. 부모 상태 주입은0회다. turn_only는320개, turn_speed는678개를 학습하므로 **파라미터 수가 같은 비교가 아니다.** 새 경로를 끈 별도 개입으로 인과 기여를 확인한다.

## 최종 후보의 연속 기록 재생

부모 대비 RMSE 감소율(%)이며 음수는 악화다. 같은 BMW/Monza의 소비된 교사 기록 재생이고 실제 학생 조작 성과가 아니다.

| 조건 | 기록 | 전체 조향 | 진입 전 조향 | 전체 가속 | 전체 제동 | 보존 gate |
|---|---|---:|---:|---:|---:|---|
'''
 names=['냉간 TRAIN','고온 TRAIN','소비된 개발']
 for arm in ('turn_only','turn_speed'):
  for row in r['models'][arm]['actual']:
   v=row['rmse_reduction_percent'];text+=f"| {arm} | {names[row['episode']]} | {v['all'][0]:.4f} | {v['pre'][0]:.4f} | {v['all'][1]:.4f} | {v['all'][2]:.4f} | {row['preservation_gate']} |\n"
 text+='''
두 후보 모두 전체3축/진입 전 조향이 부모 RMSE1.02배 이내인 보존 gate는 통과했다. 이 gate는 개선·유효 완주·속도 의미의 부호 검증이 아니다. normal 두 후보97,732사건과 새 경로를 끈 개발16,269사건, 합계114,001사건을 원본 native/영상 시각으로 재생했다.

## 같은 몸에서의 새 영상과 새 경로 개입

fit 전 봉인한 새 texture781151에서2코너×3횡위치×2속도,3가설 온도18group을 비교했다. 동일 몸/공통 신경 prefix 뒤 입력만 바꾼 계산 자극이다. 온도 반복을 독립 도로18개로 세지 않는다. 역사적 조향 부호 가설의 margin>.0001을 양성, .002를 목표로 정했다. 물리적 최적 조향 label이나 새 실제 track 검증이 아니다.

| 모델 | 정상 방향/18 | motion 제거/18 | 새 속도 경로 제거/18 | 목표/18 |
|---|---:|---:|---:|---:|
'''
 for arm in ('parent','turn_only','turn_speed'):
  row=r['models'][arm]['reserved'];counts=row['direction_pairs'];text+=f"| {arm} | {counts['fine']} | {counts['motion_off_after_prefix']} | {counts['speed_off_after_prefix']} | {row['target_margin_pairs']} |\n"
 cut=r['development_speed_cut'];text+=f"\n같은 turn_speed 가중치를 유지하고 새 속도 경로만 처음부터 끈 개발 기록과 비교하면, 정상 경로의 전체/진입 전 조향 RMSE는 {cut['normal_vs_speed_off_rmse_reduction_percent']['all'][0]:.4f}%/{cut['normal_vs_speed_off_rmse_reduction_percent']['pre'][0]:.4f}% 감소했다. 출력 최대 차이(조향/가속/제동)는 {cut['maximum_axis_prediction_difference']}다. 이는 고정된 기록 입력에서 새 경로의 영향이며 실제 차가 더 잘 달렸다는 증거가 아니다. 새 질감은 소비됐고 목표 margin은 미달이다.\n"
 text+='''
## 판독기의 부호 반전: 이번 핵심 실패

고정 부모의 냉간2/고온2 TRAIN 구간에서 속도 요청을±.01/±.001m/s 바꾸고 같은 몸/영상/신경 prefix로 비교했다. 높은 요청은 내부 power 전류를 높이고 modeled power_L/R도 높였다. 그러나 최종 가속은 줄고 제동은 늘었다. 네 구간의 local d(gas−brake)/d(requested_speed)는 약−.0618~−.0778이었다. 유한차분과 autograd 벡터 차이는 작은 간격에서 최대.142%다.

이 반전의 약99.86~99.96%를 두 power proxy의 차량 판독 기여로 설명했다. 원본±출력/근육 배열에 별도 수식을 적용해 재검산했다. 여섯 값은 모델의 근육 proxy이며 실측 wing force가 아니다. 실제 차량 가속도/모든 상태의 회로 단조성을 측정한 결과도 아니다.

기존 차량 판독기는 각 입력을 양의 TRAIN scale로 정규화하고 linear→tanh/두 sigmoid를 적용한다. 고정된 power 두 열의 계수는 다음과 같다.

| 판독 logit | power_L | power_R |
|---|---:|---:|
| 가속 | −1.429436 | −1.157619 |
| 제동 | +.862485 | +1.258021 |

따라서 다른 근육을 고정하고 power proxy만 높이면 이 판독기는 가속을 감소시키고 제동을 증가시키는 방향이다(포화 상태에서는 반응이0일 수 있음). steering logit에도 common-power 직접 기여가 남아 있다. 내부 m/s encoding의 단위/전류 부호 통과와 차량 속도 제어의 의미 통과를 구별해야 한다. 새 경로의3→6/18 방향 반응이나 작은 RMSE 개선을 올바른 속도 제어·비행 조향 기능의 증거로 채택하지 않는다.

## 판정·원본·다음 수정

**두 후보는 미승격이고 speed-direction semantics gate는 false다.** 새로운 입력의 반응은 생겼지만 기존 차량 판독기의 방향 계약이 맞지 않았다. 다음은 측정 회로와 기존 결과를 보존하면서 공학적 근육 proxy→차량 판독기를 보정하는 것이다. common-power의 종방향 부호, 좌우 opponent 조향 및 대칭 조건을 명시하고, 원본 TRAIN 적합도와 실제 냉간/고온 보존·영상 개입을 함께 검사한다. 새 입력의 부호를 무조건 뒤집어 원하는 속도라고 다시 부르지 않는다.

원본은 `runs/visual-turn-speed-20261104/`의 `plan-01.json`, 초기 실패 `validation-01`, 수렴 확인 `gradient-resolution-01`, `runtime-neutral-01`, 두 fit/10개 평가/두 감사, `control-dimensions-01`, raw `speed-sign-fd-02/probes.npz`, `speed-sign-audit-01.json`, `summary-01.json`이다. 초기 부호 검사의 소스도 보존했다. 초기 검사 뒤 합성 native mask 정책을 추가했고, 당시 plan/source 원문은 기존 manifest SHA와 정확히 일치하는 바이트로 복원해 보존했다(`initial-source-recovery-01.json`). 이후 fit의 plan/source 스냅샷은 별도로 고정했다.

후보 source/weight·입력 hash·seed·optimizer proposal/실제 gradient/clip 궤적·1020블록 상태 journal과 종료 사유를 남겼다. 두 감사는 원본 native 값/mask·속도/전류 수식·근육→출력과 지표를 검산했으며 전체 신경망을 다시 실행한 것은 아니다. 기존 원자료·모델·성공/실패는 보존했다.

실제 시각/몸 감각과 차체 기울기·회전·가속도·하중/접지·타이어 온도/슬립을 되먹임하는 조향·가감속, 변속·셋업/차종/상대풍 적응, 빠른 유효 완주/라인/랩타임과 한계 주행은 계속 목표다. 부품 계산/학습·생리 기능·차량 통합·실제 자율 주행을 서로 대체하지 않는다. 목표는 active다.
'''
 Path('docs/visual-turn-speed-and-decoder-sign-20261104.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
