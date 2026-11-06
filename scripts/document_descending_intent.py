"""Write the matched-trial report from sealed summary metrics (Markdown only)."""
from pathlib import Path
import json

def main():
 root=Path('runs/descending-visual-intent-20261104');r=json.loads((root/'summary-01.json').read_text(encoding='utf-8'));models=r['models'];text='''# 기존 시각 하행 명령 보정 — 2026-11-04

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

이번 단계는 기존 시각→비행 하행 입력의 학습과 원본 연속 기록 평가다. 실제 학생 운전·생리 기능·새 코너 자율 주행·온도 적응·한계 주행이 검증됐다는 뜻은 아니다. 학생 제어/시뮬레이터 변경/API/UI 조작은0이며 후보는 승격하지 않았다.

## 문제와 변경

직전 회전 결합 진단에서 같은 몸·신경 상태의 영상 교환이 조향을 바꾸지만 충분하고 대칭적인 방향 판단은 없었다. 기존320→1 시각 하행 어댑터는 합성 좌/우 영상 모두 한쪽 끝에 가까웠다. 실제 TRAIN 두 기록32,597사건도 따로 검사했다. |투사값|>2 비율은41.17%/42.38%이고, 전체에서는 양쪽 부호가 모두 나왔다. 따라서 전체 실제 주행이 한쪽으로 고정됐다고 해석하지 않는다. 이 수치는 회로 전체 손실률이나 운전 정보 손실률이 아니다.

기존 `VisualDescendingPort.projection.weight`320개만 학습했다. 새 운전 모델이나 직접 조향 우회 경로를 추가하지 않았다. 측정 비행 그래프/신경 동역학/포트, 몸 안정화/감각 경계, 정적/운동 시각 투사, 근육→조향·가속·제동 판독과 모든 정규화·부호·tonic/최대 전류는 고정했다. 어댑터는 공학적 시각 의도 입력이며 측정 HS/VS→DN 시냅스 또는 고차 뇌로 부르지 않는다. 출력은 기존 비행 회로와 날개 근육 판독을 통과한다.

두 조건은 같은 fine 부모 fc4a2997…·초기 가중치·seed781123·순서·147회 업데이트·학습률 .001·원래 투사 대비 성분별 ±.025 경계를 사용했다. 실제93구간은 모두 TRAIN0/1의 교사 명령+출력 보존으로 학습한다. 합성54구간(3질감×6위치/속도×3가설 온도)은 real_only에서 출력 보존만, joint에서 좌/우 방향 손실+보존을 적용했다. 따라서 real_only도 합성 구간54회의 anchor 업데이트를 포함한다.

합성 prefix/분기 전체의 gradient를 유지한다. 실제 학습은 감사된 부모의 수치 prefix에서 시작하는32사건 구간이며, 그 수치 경계는 전체 이력 gradient 학습이 아니다. 평가에서는 세 실제 기록을 처음부터 끝까지 연속 실행한다. detach와 수치 상태 초기화를 구별했다. 개발 기록은 학습에 쓰지 않았다. 구성 로더가 세 기록을 메모리에 읽는 것과 실제 fit에 사용하는 구분도 유지한다.

이전 factorial의 합성 몸 fixture는 알려진 가설 하중/휠/기준 압력을 포함한다. 실제 센서 결측을 복구한 자료로 부르지 않는다. 이번 실제 기록 평가는 원본 native bytes와 유효성 계약을 사용한다. 회전 진단의 unknown-mask fixture와 서로 바꾸어 부르지 않는다.

## 연속 실제 기록 결과

아래 숫자는 부모 대비 RMSE 감소율(%)이며 음수는 악화다. 같은 BMW/Monza의 소비된 교사 기록 평가이며 실제 학생 조향 성과가 아니다. 전체/진입 전 조향과 전체 가속·제동을 함께 남긴다.

| 학습 | 기록 | 전체 조향 | 진입 전 조향 | 전체 가속 | 전체 제동 | 보존 gate |
|---|---|---:|---:|---:|---:|---|
'''
 names=['냉간 TRAIN','고온 TRAIN','소비된 개발']
 for arm in ('real_only','joint'):
  for row in models[arm]['actual']:
   v=row['rmse_reduction_percent'];text+=f"| {arm} | {names[row['episode']]} | {v['all'][0]:.4f} | {v['pre'][0]:.4f} | {v['all'][1]:.4f} | {v['all'][2]:.4f} | {row['preservation_gate']} |\n"
 text+='''
보존 gate는 전체 세 축과 진입 전 조향의 RMSE가 부모 대비1.02배 이내라는 제한이다. gate 통과가 유효 완주/최적 제동/타이어 적응을 보장하지 않는다. 두 후보의 연속 평가 합계는97,732사건이다.

## 같은 몸 상태에서의 영상 방향 검사

새 texture781127은 두 fit 전 계획을 봉인했고 학습 코드가 픽셀을 읽지 않았다. 영상은2코너×3횡위치×2속도이며3가설 온도로18group을 비교한다. 온도를 반복한 group을 독립 도로18개로 세지 않는다. yaw0의 선행 시각 구분이며 이전 회전 결합의 도움으로 대체하지 않는다. 코너는공통 prefix 뒤 나타나며 모든 평가는 진입 전이다. 기존 조향 부호 가설의 margin>.0001을 양성으로 세고 목표는.002다. 물리적 최적 조향 label은 아니다.

| 모델 | TRAIN 양성/54 | 새 질감 양성/18 | 새 질감 motion 제거/18 | 새 질감 목표/18 |
|---|---:|---:|---:|---:|
'''
 for arm in ('parent','real_only','joint'):
  row=models[arm];v=row['reserved'];text+=f"| {arm} | {row['train_positive']} | {v['direction_pairs']['fine']} | {v['direction_pairs']['motion_off_after_prefix']} | {v['target_margin_pairs']} |\n"
 text+='\n후보별 새 질감 하행 투사 작동점:\n\n'
 for arm in ('real_only','joint'):
  row=models[arm];text+=f"- {arm}: raw 범위 {row['reserved_intent_raw_range']}, 중심 대비 기울기 비율 {row['reserved_intent_slope_fraction_range']}.\n"
 text+='''
작동점 변화와 조향 오차 변화는 별개다. 새 질감 하나의 방향 수치만으로 실제 새 코너 일반화/생리 기능이 검증됐다고 하지 않는다. 소비한781127은 이후 새 예약 자료로 다시 부르지 않는다.

## 검증·원본과 한계

- TRAIN 첫 합성 pair/80℃에서 현재 투사 gradient norm .000643836을 확인했다. 중앙 차분 step .01은5.35% 상대차, .02는21.79%였다. 현재 점/방향의 제한된 확인이며 작은 수치 간격/모든 비선형 궤적의 정확성을 주장하지 않는다.
- 별도 NumPy Adam/경계 투영 재생으로147회 가중치 이력을 감사했다. 학습 순서·실제93구간/합성54개 커버리지·부모 prefix 시각·변경 허용 범위도 검증했다.
- 두 후보 각각 원래 그래프/몸/모든 근육 판독/다른 시각 가중치와 하행 어댑터의 투사 외 상태가 동일함을 확인했다.
- 저장한 근육→세 출력 수식과 raw 예측→실제 RMSE/합성 방향 통계는 독립 재계산했다. 감사가 전체 신경망을 두 번째 실행한 것은 아니다. 연속 신경 재생은 별도 평가 진입점의 원본 결과로 연결된다.
- 허용 변경/결측·비정상 가중치·형상·다른 몸/운동 시각/판독/정규화·부모 변경 거부10검사와 기존 경계11검사, 합계21개 통과.

`runs/descending-visual-intent-20261104/summary-01.json`이 두 fit/15평가/두 감사 원본을 연결한다. checkpoint는 각 `real_only-fit-01/descending-intent.pt`, `joint-fit-01/descending-intent.pt`에 보존했다. 입력/소스/부모 hash·seed·학습 궤적·종료 사유와 실패/제약을 남겼다. `metadata-clarification-01.json`은 상속된 옛 gain 진단 필드와 이번 실제 gating을 구별한다. 옛 gain gradient 검증이 이번 투사 검증을 대신하지 않는다.

최종 조향/가감속 개선 판정은 위 원본 재생 수치와 방향 검사 양쪽을 따른다. 실제 타이어 온도·자세/회전·가속도·하중/접지 되먹임, 변속·차종/셋업·상대풍 적응과 빠른 유효 완주/레이싱 라인/랩타임·한계 주행은 계속 목표다. 생리 기능·차량 통합·학생 운전 qualification은 별도로 남는다. 전체 목표는 active다.
'''
 Path('docs/descending-visual-intent-calibration-20261104.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
