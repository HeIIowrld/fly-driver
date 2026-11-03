# 비행 운동 회로 안의 학습 가능한 물리 명령 경계

2026-10-30, Asia/Seoul. t1705의 CPU 부품 자격 검사다. **운전 학습이나 학생 배포를 완료한 결과는 아니다.**

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 이번 변경은 이 목표의 운동 명령 경계를 학습할 수 있게 만든 것이다. 일반 운전망이나 작은 부품의 성공으로 목표를 대신하지 않는다.

## 바꾼 것과 보호한 것

`src/flybrain_drive/flight_maneuver_learning.py`에 별도 `TrainableManeuverCurrents`와 `TrainablePhysicalManeuverCircuit`를 추가했다. 기존 `flight_maneuver.py`, 실제 제어 trial, 실측 화학 그래프, 보호된 haltere/body 인코더, 운동뉴런 readout은 수정하지 않았다. 새 회로도 원래 `PhysicalManeuverCircuit.drive/advance`를 그대로 상속한다.

학습 경로는 `물리 목표 → DNa/DNg 명령 전류 → 원래 실측 화학 회로 → 날개 운동뉴런 → 동결 readout → neural intent → 명시적 actuator codec`이다. 조향이나 페달에 더하는 residual/PID 경로는 없다. 실제 몸 yaw는 원래 보호된 몸 경계에 한 번 들어가고, 명령 앞에서 다시 빼지 않는다.

새 파라미터는 네 개다. 왼쪽/오른쪽은 기존 인코더의 부호 방향이며 새로 측정한 생체 방향 튜닝을 뜻하지 않는다.

| 계수 | 0 초기값 | 범위와 적용 위치 |
|---|---:|---|
| 왼쪽 DNa gain | .2 | `.2 exp(log(4) tanh(raw))`, .05–.8 |
| 오른쪽 DNa gain | .2 | 같은 식, 반대 부호 명령 집단 |
| DNg speed-error gain | 2 | `2 exp(log(4) tanh(raw))`, .5–8 |
| DNg speed feedforward | 0 | `.5 tanh(raw) tanh(desired_m/s / 10)`를 DNg logit에만 추가 |

feedforward는 목표 속도0에서0이므로 조용한 몸·명령0의 DN 전류를 새로 바꾸지 않는다. 원래 yaw/speed 단위, racing ceiling 계약, dtype/device/유효성 검사를 재사용한다. speed-error 제거와 feedforward 제거는 별도 개입이다. 기존 load8:12/온도12:16/바퀴17:21은 여전히 이 운동 경로의 causal 입력이 아니다. 네 계수만으로 타이어/셋업 적응이나 가감속·조향 결합의 해결을 주장하지 않는다.

## 적용된 native 값과 학습 표적을 구분한다

내장 AI는 vJoy 경로를 우회해 .02보다 작은 native 페달 값을 적용할 수 있다. 따라서 실제 native 값이 .016이라고 해서 학생의 canonical .016을 정답으로 만들지 않는다. 다음 값과 마스크를 별도로 반환한다.

- 원래 `native_target`: 실제 관측을 보존한다.
- `reachable_projected_native_target`: 현재 codec으로 표현 가능한 별도 목표다.
- `target_for_supervision`, `projection_applied`: 명시적으로 선택한 학습 정책과 실제 변경량을 구분한다.
- `native_unreachable_pedals`, `native_pedal_coactivation`, `steering_outside_calibration`, `supervision_valid`: 어떤 이유로 축별 loss를 사용할 수 없는지 남긴다.

기본 정책 `reject_unreachable`은 양의 native<.02 페달 표적을 loss에서 마스킹한다. `nearest_reachable`을 명시한 경우에만 .01 이하→0, .01 초과·.02 미만→.02로 표적을 투영한다. 문턱 위에는 .02를 더하지 않는다. 원래 native 값과 투영 사실을 숨기지 않는다.

표적은 가능한 latent 요청의 **집합**으로 표현한다. native0에는 canonical[0,.01], native≈.02에는 canonical(.01,.02]를 허용한다. 후자의 열린 경계는 선언한 .0001 학습 margin으로 표현한다. 이것은 새로운 실제 cutoff 측정이 아니다. native .02 근방의 관측 tolerance도 .0001로 명시한다. 문턱 위 native는 identity 표적이다. 임의의 한 neural intent가 유일한 정답이라고 정하지 않는다.

native gas와 brake가 동시에 양수이면 하나의 signed forward intent로 표현할 수 없다. 두 longitudinal supervision 셀을 마스킹하고, 원래 모든 관측·상태·몸 감각 유효성은 그대로 둔다. gas−brake로 숨겨서 하나의 정답으로 만들지 않는다. steering의 native gain1.406249850988388은 t1671의 작은 요청 범위 ±.1에만 적용한다. 그 밖의 steering 표적은 마스킹하며 큰 입력으로 선형 외삽하지 않는다. 큰 영역은 별도 실제 보정이 필요하다.

`native_control_valid`는 원래 관측 유효성이고, 실제 loss의 축별 유효성은 capability 검사를 추가한 `supervision_valid`다. 후속 trainer는 두 count를 모두 보고해야 한다. 지원되는 표적이 없던 축을 학습했다고 부르지 않는다.

## 손실과 자료의 자격

`applied_axis_imitation_loss`는 관측된 교사 적용 축과 일치하는 canonical 집합까지의 거리를 계산한다. 이는 **교사가 실제로 한 제어의 모방**이며 최적 제동이나 최적 레이싱 라인의 증명이 아니다. 목표 속도0을 full-brake 정답으로 만드는 변환은 없다.

`canonical_actuator_tensors`는 원래 canonical, hard encoded 요청, gap, 투영 거리와 미분 가능한 feasibility loss를 반환한다. encoded hard projection에 가짜 identity gradient를 주는 straight-through 방식은 쓰지 않는다. feasibility만 줄이면 제어를0으로 만드는 해도 있으므로, 이 값만으로 운전 학습을 판정할 수 없다.

`validate_motor_training_episode`와 `configs/flight-motor-boundary-training-contract-20261030.json`은 다음 경계를 둔다. loader는 먼저 source manifest SHA를 검증하고 그 안의 원래 episode record를 넘겨야 한다.

- 원래 역할이 TRAIN인 전체 이력만 학습한다. development/evaluation을 TRAIN으로 바꾸거나 과거 제동 DEV(t1465)를 재분류하지 않는다.
- 물리 명령의 출처를 교사 yaw reference+외부 과제 또는 시각 actor 예측+외부 과제로 명시한다. 학생 자신의 작은 요청을 최적 교사 정답으로 부르지 않는다.
- 원래 전체 관측 수·clock·몸 mask fingerprint와 양의 간격≤80ms를 유지한다. body/native control은 같은 native packet과 원래 receipt에서 나온다.
- 몸 관측은 decision 이전이고 age≤50ms이며 원래 유효 yaw/speed를 요구한다. 불가능한 몸 감각을 oracle로 메우거나 행을 삭제해 통과시키지 않는다.
- 축별 teacher loss mask와 actor의 몸 validity를 구분한다. native 관측·표적 projection·학생 요청·실제 적용의 출처와 단위를 따로 남긴다.

실제 고속 단위 학습에는 양쪽 회전과 양/영/음의 speed error가 독립적으로 변하는 자료, 실제 움직이는 가속·coast·제동·변속,60→100/가변 pace가 필요하다. 하중·온도·타이어·셋업 적응에는 해당 조건이 변한 별도 자료와 causal 경로가 필요하다. t1689의1m/s 부품 자료는 몸속도 약0–.403m/s뿐이고 학생이 스스로 만든 제어다. 이것으로 고속 plant나 최적 페달·접지 한계를 식별할 수 없다.

## 실제 CPU 자격 검사

최신 원본은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1705-motor-boundary-learning-qualification-v3`다.

- 원래 t1689의68개 몸/명령/마스크 입력에서 초기 DN drive가 기존 회로와 **bitwise 동일**했다.
- 67개 원래 간격/3.6801014초를 이전 입력 유지로 재생했을 때 새 회로와 기존 `PhysicalManeuverCircuit`의 수치 상태가 bitwise 동일했다. 중간 reset·retiming은 없다. 실제 t1689 GPU intent의 재생 오차를 다시 판정한 결과와는 구분한다.
- 현재 허용된 tilt0/1·yaw4·speed16만 유효한 합성60km/h 몸과 목표100/20km/h, yaw±.3rad/s 네 조건을 주입했다. 이것은 새 실제 고속 운전 자료가 아니다.
- 실제 실측 화학 회로를 거친 네 새 계수의 gradient가 모두 유한·비영이었다. 합성 intent Jacobian rank4, singular values 약 .449225/.008404/.005274/.002489였다. 이는 이 신경 모형의 국소 미분 가능성이지 차량 plant 식별이 아니다.
- 고정된 운동뉴런 activity 제곱의 진단 목적함수로 Adam1회만 수행했다. 네 계수만 바뀌었고 원래 sparse graph의 layout/indices/values, 보호된 몸 인코더와 readout은 모두 bitwise 동일했다. 그 뒤 원래 신경 응답이 바뀌고 조용한 명령0의 drive는 기존과 같음을 확인했다.
- 원래 체크포인트 readout을 그대로 두었고 이 새 qualifier에서 private neutral offset을 재보정하지 않았다. 50ms 과도 응답의 intent 부호를 목표 속도 추종 성공으로 해석하지 않는다.

최종 변경 뒤 관련25개 검사와 Ruff가 통과했다. 첫 t1705는 sparse CPU tensor에 `torch.equal`이 지원되지 않아 마지막 검증에서 실패했다. 원본과 diagnostic step1을 보존했고, sparse storage를 직접 비교하도록 고친 v2는 통과했다. v2의 합성 all-valid mask를 현재 허용 채널만으로 더 좁힌 별도 v3도 통과했다. 세 실행은 각각 새0초기값에서 시작한 별도 진단 step1이며, 한 운전 후보의 누적3회 학습이 아니다. **운전 학습 업데이트는 모두0**이다.

원래 NPZ/PT/카메라는 복제하지 않았다. 저장한 `diagnostic-boundary-only.pt`는 네 새 계수와 고정 작은 command buffer만 포함하고 원래 graph/checkpoint는 SHA 참조로 재사용한다. artifact schema가 `boundary_diagnostic_not_driving_candidate`여서 실제 운전 후보로 부르지 않는다. GPU·차량 API·MMF·UI·외부 프로세스 조작은0이다.

60→100/가변속·조향과 가감속·변속·하중/타이어/셋업 적응·유효 완주·빠른 레이싱 라인/랩타임 전체 목표는 그대로 남는다. 다음은 사전에 역할·조건을 정한 실제 교사/native 자료의 자격과 단위를 검사한 뒤, 이 명령 경계 안에서 모방/과제 학습을 수행하고 독립 실제 응답을 검증하는 것이다.
