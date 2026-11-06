# 시각 정보 판독과 조건 다양화 비교 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준과 가감속·변속·셋업/차종/상대풍·타이어 적응을 유지한다. 실측 연결·author 생리 모델·공학적 변환·학습한 차량 대응을 구별한다. 지도/world pose/절대 heading/교사 조작은 actor 입력이 아니다.

## 왜 영상이 약하게 기여했는가

기존 몸 감각/이력에서 조향을 재현하는 지름길이 있으며, 기존 한 코스 기록만으로 코너를 외웠는지 또는 처음 보는 코너를 영상으로 판단하는지 구분할 수 없다. 몸의 회전·가속도는 이미 시작한 조향의 결과와 상관되므로 현재 조향 오차가 작아도 선행 판단을 증명하지 않는다. 지도나 코너 번호를 actor에 넣은 것은 아니다. 기록의 body-only 비교와 선행 실패는 앞선 보고를 따른다.

이번에는 `runs/motion-descending-transfer-20261104/probe-01`, `probe-02`에서 고정 시각 특징의 비교 판독기를 따로 적합했다. 이는 연구용 ridge 판독기이며 운전자에 연결하지 않았다. 같은 몸/신경 prefix의 좌우 영상 쌍 순서는 원래 pooled/fine 특징 모두 별도 소비 조건에서6/6까지 읽혔다. 하지만 pooled의 절대 좌우 분류는 TRAIN12/12에서 질감 변경6/12, 위치 하나를 제외한 적합은2/4~4/4로 떨어졌다. fine도 위치 제외2/4씩이었다. 쌍 사이 차이의 판독 가능성은 한 장면의 방향 판단이나 실제 운전 성과가 아니다.

`FineMotionHistory`는 새 fine 특징에 과거 입력을 유지한 정확한 ZOH fast/slow 시간 필터를 적용한 연구용 가설이다. 실제 시간·촬영 일치·명시 episode reset·invalid 입력 중단을 검사한다. innovation/fast-minus-slow도 질감·위치 편향을 안정적으로 해소하지 못했다. 따라서 이 모듈을 actor에 연결하거나 새 비행 하행 회로로 부르지 않는다. 이번 actor에는 DN 우회 경로도 추가하지 않았다.

## 같은 예산의 학습 비교

루트는 `runs/motion-descending-transfer-20261104/`다. `balanced-fit-plan-01.json`에서 두 arm 모두 원래 fine 후보fc4a2997…로 시작하고 실제 TRAIN93개 window와 합성54개 pair, 총147회씩 한 번 학습한다. pooled visual 영상열과 기존 fine motion projection만 각.001 이내에서 보정한다. body/DN/static/근육 readout/chemical graph/정규화/visual bias/첫12열은 고정한다. seed781112·LR1e-4·clip1·실제 손실/anchor·합성 방향 margin.002 및 손실 가중은 두 arm에 같다.

- diverse: 이미 소비한781104/781105/781107 세 조건을 이번 TRAIN으로 명시한다. 과거 평가의 당시 역할은 변경하지 않는다.
- repeated: 동일54개 slot을781104 입력으로 반복한다. 실제 window와 순서/예산은 같다.

부모 numerical prefix는 이전 공동 보정의 `runs/visual-route-protection-20261104/fit-01/parent-prefix-states.pt`를 SHA와 부모 receipt,93개 window coverage 검증 후 참조한다. 이전 전체32,597사건 parity0을 재사용하며 이번에 다시 계산했다고 주장하지 않는다. 이는 명시적 연구용 training cut이고 live 중간 상태 reset이 아니다. 보정 후 평가는 업데이트된 모델로 전체 기록을 연속 재생한다.

새 예약 조건은 학습 전에 `reserved-plan-01.json`으로 봉인한 seed781109·curvature.016·bend13m·yaw−.02rad다. 합성12case와3온도의18쌍은6개의 위치/속도 조합을 반복한 것이며18개의 독립 실제 코너가 아니다. common prefix 최대차8.941e-8, GPU drain/release 완료를 결과로 보존했다. generator 원래 종료 핸들은 관측 응답이 잘려 exit code를 회수하지 못했으므로 exit0으로 단정하지 않는다.

## 결과

두 arm은 동일한 부모·설정·147개 작업 순서를 독립 비교했고 모두 종료했다. 원본 checkpoint SHA는 diverse d757a06c4f53f074662ead7321f9ad824e6f084486ebf54a04d95e97a47cf8fa, repeated 25d068f65c0a68798a7ebfab68f8bbcd8151d87a7dd11970f3917eaaeb723e66다. 둘 모두 새 연구 후보이며 원래 live 후보를 교체하지 않는다.

| arm | TRAIN0 방향 순서 /18 | 새 예약 방향 순서 /18 |
|---|---:|---:|
| 부모 | 3 | 9 |
| diverse | 9 | 9 |
| repeated | 9 | 9 |

방향 순서 기준은 canonical steering margin>.0001이며 모든 motion-off-after-prefix 방향 순서는0/18이다. 두 후보의 TRAIN0/새 예약 목표 margin .002 도달 수는 모두0/18이다. 방향 순서의 작은 양수 여부만으로 생리/주행 기능을 판정하지 않는다. 이번 새 seed781109도 평가에 소비됐으므로 후속 독립 조건으로 재사용하지 않는다.

| arm | 기록 | 사건 | 전체 조향 RMSE 감소율 | 진입 전 감소율 | 가감속 포함 보존 |
|---|---|---:|---:|---:|---|
| diverse | cold TRAIN | 16,425 | 0.1982% | 0.2113% | True |
| diverse | newhot TRAIN | 16,172 | 0.0987% | 1.2239% | True |
| diverse | 소비된 nominal 개발 | 16,269 | 0.1282% | 0.3612% | True |
| repeated | cold TRAIN | 16,425 | 0.1981% | 0.2339% | True |
| repeated | newhot TRAIN | 16,172 | 0.0940% | 1.2341% | True |
| repeated | 소비된 nominal 개발 | 16,269 | 0.1222% | 0.3678% | True |

감소율은 원래 fine 부모 기준이며 음수는 악화다. 전체 조향/진입 전 조향과 가감속의 수치는 `summary-01/result.json`에 있다. 과거 소비된 동일 코스 기록의 재생이며 실제 학생 조작 개선 또는 미지 코너 검증이 아니다. 두 arm의 합계97,732사건을 업데이트한 신경 상태로 처음부터 연속 재생했다.

독립 감사 `audit-diverse-01`/`audit-repeated-01`은 고정 graph/body/DN/static/근육 상태, 허용 visual weight 변화,93개 TRAIN window/prefix 시각 바인딩과54개 합성 slot, 새 seed 분리, 합성 출력의 근육 수식과 실제 기록의 RMSE를 원본 배열에서 확인했다. 감사 자체가 전체 neural trajectory를 다시 재생한 것은 아니다. 관련22검사는 통과했다. 첫 감사는 seed 대신 texture_seed를 읽어야 하는 필드명 오류로 exit1이었고 실패 코드/로그를 보존한 뒤 수정 감사가 통과했다. 두 fit 원래 tool handle exit0은 `process-receipt-final-01.json`,10개 평가 child exit0은 `evaluation-process-receipts-01.json`에서 확인한다. generator는 결과/GPU release는 확인했지만 원래 exit code는 관측하지 못했다.

이 단계는 영상 차이를 읽을 수 있는 입력과 안정적인 조향으로 전달하는 기능 사이의 문제를 좁힌 것이다. 새 영상 일반화 개선은 입증하지 못했으며 두 후보 모두 미승격이다. 다음은 실제로 위치·방향·속도를 통제한 다중 라인/새 코너 자료와 운동 시각→비행 유래 근육 이전 전달 기능을 구분해 다룬다. 실제 학생 주행, 타이어 온도/자세/가속도 적응·가감속/변속/셋업/차종/상대풍·유효 완주/라인/랩타임·한계 주행과 생리 검증은 남아 있다. actor_qualified=false, 학생 actuator/lifecycle/UI0, 전체 목표active다.
