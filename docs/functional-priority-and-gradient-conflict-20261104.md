# 기능 우선 학습과 gradient 충돌 비교 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·가감속/변속·하중/타이어/온도·셋업/차종/검증된 상대풍·빠른 유효 완주/라인/랩타임을 유지한다. 실측 연결·author 모델·기능 근사·공학적 입력/차량 변환은 구분한다. 지도/world pose/절대 heading/교사 명령은 actor 입력이 아니다.

## 결과

원본은 `runs/functional-priority-learning-20261104/summary-01/`다. 이전 goal turn은 내부 상태·인과 개입과 gain 후보의 학습/전체 재생으로 실제 재현 개선과 예약 시각 악화를 구분했으므로 progress다. 이번에는 같은 파라미터 지점의 목표 충돌을 측정하고, 전체 합성 이력 gradient를 유지한 두 후보를 같은 예산으로 비교했다.

**두 후보 모두 기존 기록 보존은 통과했지만, 기능 우선 후보의 새 영상 반응은 부모3/18에서3/18로 그대로였다. 대조군은6/18이었다. 손실을 바꾸는 것만으로 시각 판단이 좋아졌다는 근거는 없으며 둘 모두 미승격이다.** 연구용 단계이며 실제 한계 주행 달성으로 판정하지 않는다.

## 고정된 같은 상태에서의 gradient

원래 fine 부모fc4a2997…와 직전 gain 후보29a5cbd0…의 두 파라미터 지점에서, 실제 TRAIN episode0/1의 유효 label이 있는12개32사건 window와 합성 TRAIN10개 pair(몸80℃)를 골랐다. 합성은 첫 질감의6조합과 다른 두 질감의중앙 횡위치/두 속도다. 전체 TRAIN 평균을 대표한다고 주장하지 않는다. parameter 업데이트 없이 각 손실의42차원 gradient를 기록했다.

| 지점 | 실제 명령 적합 vs 기능 방향 평균 gradient cosine | 음의 개별 조합 /120 |
|---|---:|---:|
| parent | -0.947996 | 64 |
| previous_gain | -0.930132 | 62 |

같은 지점의 평균 방향이 강하게 반대였지만 모든 개별 표본이 충돌한 것은 아니다. 고정 지점 진단을 실제 Adam 전체 궤적의 인과 증명으로 대신하지 않는다. `diagnostic-audit-01`이 저장 vector로 cosine/개수와 TRAIN 역할을 다시 확인했다.

기존 합성 학습은 prefix와 frame11~24를 수치적으로 진행한 뒤 detach하고 frame25~40만 역전파했다. 이번에는 공통 prefix의 state/traces/previous drive 등8개 값을 gradient를 유지한 clone으로 나누어41프레임 이력을 역전파했다. 손실에 사용한 후반 출력은 기존 방식과 최대차0이었다. 전체 신경 상태의 모든 시점을 비교한 수치는 아니다. episode reset과 연구용 실제 TRAIN cut 이외의 생산 중간 reset은 추가하지 않았다.

처음 epsilon .002/.0005의 float32 중앙 차분은 미분값과 충분히 맞지 않았다. 실패한 수치를 보존한다. 같은 autograd forward 모드·넓은 epsilon .05의 첫 TRAIN pair 재검사에서는 두 지점 상대오차3.430%/.736%였다. epsilon .01/.002는 더 큰 상대오차가 남았다. 작은 간격의 수치적 불안정이라는 근거이며 원인을 모두 분리하거나 전체 gradient 정확도를 증명한 것은 아니다. validator 첫 파일은 괄호 문법 오류로 실행 전 실패했고, 실패 source/log를 보존한 후 수정 실행했다. 이 실패에서 모델 업데이트는 없었다.

## 같은 예산 두 후보

둘 다 **fine 부모와 gain1에서 새로 시작**했다.29a5cbd0…는 위 진단 지점으로만 사용했으며 학습을 이어받지 않았다. seed781118·같은147개 순서(실제93+합성54)·Adam LR.05·clip1·gain42개 범위.25~1.75를 사용했다. 기존 graph/dynamics·temporal projection/traces·pooled/fine 공간 시각·body/native/DN/static·6근육 readout/정규화는 고정했다. gain은 공학적 보정이며 측정된 생리 gain이 아니다.

- supervised: 실제 TRAIN의 기존 교사 명령 적합 + 부모 anchor.
- functional_priority: 실제 TRAIN에서는 부모 anchor + 보존 한계 초과 페널티. 허용 영역 안에서는 교사 오차를 더 줄이는 supervision을 적용하지 않는다.

둘 다 합성 TRAIN은 기존 방향 hinge²×1000(목표 margin.002) +3축 anchor.5이며 prefix/branch 전체 gradient를 유지했다. 실제 보존은 label이 유효한 각 window의3축 MSE와 진입 전 조향 MSE가 부모 MSE×1.02²를 넘을 때만 상대 초과분 제곱을 적용한다. 분모 floor1e-5·weight1이고 **hard constraint가 아닌 위반 페널티**다. 실제 window의 부모 numerical prefix는 이전32,597사건 parity0의 저장 원본을 SHA와93개 clock/coverage 확인 후 재사용했다. 이번 새 연속 재생 사건으로 세지 않는다.

개발 episode2는 공통 loader가 메모리에 읽지만 task/loss에는 넣지 않는다. 새 예약 픽셀/특징은 optimizer가 읽지 않았다. 실제 기록 전체의 보존은 보정 후 별도로 확인했다.

## 동일 조건 평가

새 예약은 학습 전 texture781117·curvature.0195·bend12.5m·실제 사용되는 yaw_rad=.01로 봉인했다. 비작동 yaw 필드는 제거했고 설명도 맞췄다. generator 원래 핸들 exit0과 GPU drain/release를 확인했다.781117도 이번 평가에 소비됐으므로 다음 독립 조건으로 재사용하지 않는다.

| 후보 | TRAIN 방향 기준 /54 | TRAIN 평균 방향 loss | 예약 방향 /18 | 예약 목표 margin.002 /18 |
|---|---:|---:|---:|---:|
| parent | 12 | 0.004265696 | 3 | 0 |
| supervised | 27 | 0.004124048 | 6 | 0 |
| functional_priority | 12 | 0.004207176 | 3 | 0 |

방향 기준은 후반 평균 canonical margin>.0001이다.54는3질감×6개 위치/속도×3가설 온도,18은6조합×3온도이며 독립 실제 코너 수가 아니다. 모두 motion-off-after-prefix 방향0이었다. 부호는 X-forward/Y-left 합성 기하와 과거 측정 canonical steering-curvature 음의 관계에 기댄 **기능 가설**이며, 실제 미래 조향 정답/타이밍 검증을 대신하지 않는다. 기능 우선은 TRAIN 방향 loss를 조금 줄였으나 대조군보다 낫지 않았고 새 조건의 방향 수도 늘리지 못했다. 전체 이력 변경과 목표 변경의 효과를 과거 다른 seed 실험에서 분리해 주장하지 않는다. 이번 두 후보끼리는 계획/초기값/순서가 같다.

| 후보 | 기록 | 사건 | 전체 조향 RMSE 감소율 | 진입 전 감소율 | 가감속 포함 보존 |
|---|---|---:|---:|---:|---|
| supervised | cold TRAIN | 16,425 | 0.4341% | 2.7111% | True |
| supervised | newhot TRAIN | 16,172 | 0.1786% | 5.2573% | True |
| supervised | 이미 소비한 nominal 개발 | 16,269 | 0.0787% | 0.3084% | True |
| functional_priority | cold TRAIN | 16,425 | 0.1384% | 1.3102% | True |
| functional_priority | newhot TRAIN | 16,172 | 0.1761% | 2.6257% | True |
| functional_priority | 이미 소비한 nominal 개발 | 16,269 | 0.0461% | 0.4693% | True |

합계97,732사건은 기존 세 기록을 두 후보로 처음부터 연속 재생한 값이다. 원본 native prefix bytes·camera/feature 시각·마스크·recurrent state를 유지했다. 같은 코스 교사 기록의 재현이며 새 실제 코너나 학생 조작 개선이 아니다. 실제 가감속 각 RMSE도 summary에 있다.

기능 우선의 보존 페널티는93개 실제 update 중6개에서 활성화됐다. 적용 gradient norm은 실제 step 평균.000405·최대.019575, 합성 평균.000268·최대.000550이었다. 서로 다른 파라미터 지점의 기술 통계이므로 이를 고정 지점 충돌 증명으로 쓰지 않는다. 보존 페널티와 anchor가 여전히 기능 학습을 제약할 수 있으며, 충돌을 발견했다고 이미 해결했다고 보지 않는다.

## 몸 회전 결합 경로의 시험 범위

두 후보의3TRAIN+예약8개 평가 원본, 총144group에서 body21의 yaw rate(index4)는 유효하지만 모두0이었다. 기존 PerViewVisualBodyContextBoundary는 yaw×opponent trace를 bias 없는 projection에 넣으므로 이 조건에서는 출력이 정확히0이다. 실제 고정 projection을 읽은 검사에서도0을 확인했고, 임의 trace와 yaw.1의 출력은 비영이었다. 후자는 계산 가능성 확인이지 생리/차량 기능 검증이 아니다. `yaw-context-coverage-01.json`을 따른다.

따라서 이번 통제 방향 시험만으로 시각-몸 회전 결합이 검증됐다고 할 수 없다. 실제 기록은 비영 몸 회전을 포함할 수 있지만 본 비교의 같은 몸 인과 검증과는 다르다. 다음은 같은 시간의 몸 회전과 카메라 회전을 맞춘 비영 회전 조건에서 이 결합 경로를 직접 개입해 검증한다. gain/손실 변경만 반복하지 않고, 필요한 입력/시각-운동 기능과42gain의 반응 범위도 구분한다. 실제 횡위치·도로 상대 방향·속도를 통제한 영상 자료 및 조향 정답/시각 계약 검증도 남아 있다.

## 원본과 검증

- `supervised-fit-01/temporal-gain.pt`: SHA 5fce4823399efa2c05e9d9c707f2bc8f0c2d6f28241431a231d50a7228e2f5c0.
- `functional_priority-fit-01/temporal-gain.pt`: SHA 8a2319eab4513edaa756e8309ce52d1b5235240305e4e697d0a57aa5fe0e30d5.

새 loss/상태 clone/gain13검사와 기존 회로25검사, 총38개를 통과했다. 진단 감사와 두 후보 감사가 완료됐다. 두 후보는 full fixed model/body/motion/readout SHA·TRAIN/prefix/task·reserved 분리·근육 수식/실제 RMSE와 보존을 확인했다. 저장 applied gradient로 독립 NumPy Adam/clip을 재계산한 최대차는6.84e-7/8.36e-7, 선택 objective 합 재계산 최대차2.31e-10이다. 감사가 전체 neural trajectory/gradient를 다시 실행한 것은 아니다. 종료와 source/session 영수증을 따른다.

이번 학생 actuator/API/UI0, actor_qualified=false·두 후보 미승격·전체 목표active다. 기존 원본/실패/체크포인트를 보존했다. 타이어 온도·자세·가속도·하중/접지·가감속/변속·셋업/차종/검증된 상대풍·빠른 유효 완주/레이싱 라인/랩타임·한계 주행과 생리 기능은 계속 미완료다.
