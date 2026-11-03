# 구조·입출력·학습 계약

**2026-10-27 구분:** 이 문서의 본문은 현재 MuJoCo 구현과 legacy/retina_v1 경로의 계약이다. 새 목표인 **겹눈·기관별 몸 피드백·실제 초파리 회로→Assetto Corsa 완주**의 설계 계약은 [설계안](assetto-fly-design.md), 구현 순서는 [연구 계획](research-plan.md)을 따른다. 아래 MECar 좌우 토크 API가 AC에 이미 연결됐다는 뜻은 아니다.

## 계획된 backend 공통 경계 — 아직 미구현

후속 구현으로 SI 단위/좌표·두 시계·actor/TrainingOnly의 감각 계약, 합성 방향/수용각 겹눈, 기관 부호화, 관측 packing과 MuJoCo V→순환 추론 경계는 구현·검증했다. [실행 증거](sensory-integration-20261027.md). 이 절의 AC bridge·완전한 v2 수집/학습·DriveIntent 게임 변환은 아직 미구현이다.

`ActorObservation v2`(눈별 감각·기관 감각·validity/age·직전 적용 명령), `FlyState`(시각/기관/회로의 지속 상태), `DriveIntent`(전진·회전·감속), `TrainingOnly`(교사·정답 경로·critic), `Diagnostic`을 별도 객체/메시지로 나눈다. AC bridge는 실제 steering/throttle/brake로 변환하며 가용하지 않은 독립 모터 토크를 가정하지 않는다. 관측에 world pose·AI line·진행률·미래 경로를 넣지 않는다.

원영상의 실제 시야·eye map·수용각과 기관별 단위/축/대역폭/지연을 manifest로 고정한다. 수신 시점보다 미래인 telemetry를 actor에 사용하지 않는다. 양안 시점이 다른 경우 그 차이를 기록하고, 단일 좁은 화면의 복제를 넓은 양안 입력으로 취급하지 않는다. 기존 observation shape·checkpoint는 v2로 자동 변환됐다고 가정하지 않는다.

## 현재 MuJoCo 구현 계약

후속 실험의 `RoutedController`는 별도 `evaluate_routed_controller.py`에서 실행하는 부분 회로 모델이다. 시각 경계 대리 신호와 기관 feature를 `ModalityPortAdapter`로 분리하고 `SparseCircuit.step_with_drive`로 입력 역할 뉴런에만 주입한다. 수치 상태는 에피소드 안에서 유지하며, 명령은 DNa02 출력의 차동 상태만 읽는다. 기존 PPO/`policy_from_config`에 이 실험 모델이 자동 통합된 것은 아니다. [조향·제거·연결 절단 결과와 생물학적 제한](routed-circuit-driving-20261027.md).

차량 축은 X 전방·Y 좌측·Z 위쪽이고 world 좌표도 Z 위쪽이다. 양의 yaw·곡률·전륜 조향은 좌회전이다. 별도 조향장치가 방향을 만들고 좌우 후륜 추진력이 보조 요모멘트를 만든다. 오른쪽 추진력을 더하면 좌회전 성분이 생긴다. `VehicleConfig`의 질량·기하·스프링·감쇠·토크 수치는 초기 예시이며 실차 보정값이 아니다.

## 물리와 노면

`SyntheticRoad`는 XYZ 중심선과 3D 누적거리·투영·좌우 경계·법선을 만든다. 일정 경사는 비율로, bank는 도 단위로 설정한다. 같은 XY에 `flattened=true`를 주면 Z/bank/요철을 제거해 T2의 쌍을 만들 수 있다. 원래 고도를 서스펜션 스트로크에 직접 넣지 않는다.

MuJoCo 모델은 freejoint 차체 6자유도, 네 suspension slide, 두 steering hinge, 네 wheel spin으로 **16개 속도 자유도**를 갖는다. 중력은 world Z 아래로 작용한다. 합성 표면을 접촉 heightfield로 사용하고 카메라가 같은 geometry를 본다. 시각 경계 표지는 같은 높이에서 렌더링하지만 접촉 장애물은 아니다. 시작 자세는 노면 frame에 맞추고 0.8초 동안 접촉/스프링을 안정화한다.

현재 표면은 단일 값의 높이장이다. 다층 도로·교량·가드레일, 검증된 공기 타이어·마찰 변화·공력·모터 맵은 없다. 향후 실측 서킷에서는 visual mesh와 contact geometry를 별도로 검증한다. PyChrono·Assetto Corsa·MJX는 설계 후보이며 이 API의 동작 backend가 아니다.

## 관측과 정보 조건

Gymnasium 계약은 `reset -> (observation, info)`, `step -> (observation, reward, terminated, truncated, info)`다. 에피소드 종료 뒤에는 reset이 필요하다. 보상과 진단은 정답 상태를 사용할 수 있으나 정책은 observation만 받는다. 학습 코드에서 `info.truth`를 정책에 넘기지 않는다.

| 필드 | shape | 뜻 |
|---|---|---|
| body | 38 float32 | 아래 센서/진단 채널을 BODY_SCALE로 나눈 값 |
| body_valid | 38 float32 | 채널별 유효성. encoder가 값에 mask를 곱함 |
| images | 2×H×W×3 uint8 | 좌/우 실제 렌더링 RGB, 기본 32×32 |
| camera_valid | 2 float32 | O=0, V=1. renderer 실패는 예외 |
| camera_age | 2 float32 | 가장 최근 exposure의 경과 시간, 초 |

body 순서는 gyro 3축, specific-force accelerometer 3축, suspension 변위 4개·변화율 4개, wheel angular speed 4개, 실제 steering angle 2개, **명령 토크** 2개, 이전 정규화 의도 2개, 전륜 encoder 평균 속도 1개, 스프링 기반 하중 추정 4개, 잠정 중력 방향 추정 3개, 진단용 경로 특징 6개다. 명령 토크를 실제 전류 센서 측정으로 표현하지 않는다. 스프링 하중은 정답 접촉력과 다르며 비현가 질량 등을 반영한 하중 추정기가 필요하다.

O 모드의 마지막 6개 값은 진행률·횡오차·방위오차·종경사·bank·곡률 정답이다. V 모드에서는 이 값과 유효성을 모두 0으로 설정한다. O 카메라는 모두 무효다. V의 두 카메라는 차체 전방 좌우 25° 방향, 수직 FOV 80°로 설정한다. 요청 설정은 30 Hz지만 100 Hz 제어 tick에서 마지막 실제 노출 뒤 1/30초를 검사하는 현재 clock의 실제 간격은 0.04초, **25 Hz**다. 물리는 1 kHz다. 영상은 노출 사이에 유지하고 실제 frame age를 갱신한다. 두 눈의 촬영 시점은 동일 물리 상태다. 일반 perspective 카메라 두 개이며 겹눈 광학/방향 샘플링은 아직 구현하지 않았다.

현재 몸 센서는 이상적 계측이다. mask는 정보 분리를 구현하지만 잡음·bias·지연·고장 주입은 아직 없다. 중력 추정은 gyro 적분과 저속에서의 가속도 보정만 사용한다. 지속 가감속·코너링에 충분한 추정기가 아니므로 그 성능을 주장하지 않는다. FeCO 위치 tuning·방향 채널·진동 대역, 국소/상행 분리는 연구 계획에 있으며 현재는 연속 원시 채널이다. static tilt를 haltere 신호라고 부르지 않는다.

## 정책과 구동

공통 CNN은 두 RGB를 6채널로 받아 시각 특징을 만들고 mask 처리한 몸 센서·유효성·frame age와 합친다. 운동 시각 필터와 retinotopy, RGB 의미 가지는 후속 항목이다. GRU와 graph 정책은 같은 초기 encoder를 사용한다.

GRU는 64개 상태를 갖는다. graph 모델은 명시적 input 포트로만 감각 특징을 주입하고 output 포트 상태로만 의도를 읽는다. 유형 쌍 이득·유형별 bias와 시간 상수를 공유한다. `alpha = 1-exp(-dt/tau)`이며 희소 메시지는 edge gather/index_add로 계산한다. 구조적 시냅스 수를 incoming 합으로 정규화한 것은 수치 초기화이며 생리학적 전도도가 아니다. 전달물질·수용체 부호 모델은 없다. 임의 유형 쌍 이득은 학습할 수 있다.

`state`는 틱 사이에 유지하고 reset mask가 있는 배치 행만 0으로 초기화한다. detach는 gradient 경계만 바꾼다. 주 출력은 정규화된 두 의도다: `v_ref=(a0+1)*8/2 [m/s]`, `kappa_ref=a1*0.3 [1/m]`. 차량 제어기는 encoder speed와 gyro yaw를 받아 제한된 Ackermann 조향, 후륜 좌우 torque, 네 wheel brake로 변환한다. 부호·한도·제동은 검증 전 prototype이며 실제 TC/TV·회생·브레이크 분담 계약을 대체하지 않는다. 정책에는 직접 roll/pitch나 활성 서스펜션 명령이 없다.

## 초기 PPO

단일 물리 환경의 순환 PPO는 전체 rollout 시퀀스를 현재 가중치로 다시 계산한다. 정책/optimizer는 선언한 CPU 또는 CUDA 장치에서 실행한다. CUDA 요청이 불가능하면 실패하고 CPU로 바꾸지 않는다. rollout 시작은 명시적 collector reset이며 그 안의 숨은 상태는 유지한다. 종료 관측의 critic 값을 reset 전에 얻는다. 물리 실패/완주 `terminated`에서는 bootstrap을 끄고, 시간 제한 `truncated`에서는 bootstrap을 유지하되 GAE가 reset 에피소드를 넘지 않게 한다. collector horizon 끝도 value로 bootstrap한다.

`num_envs>1`이면 `parallel_learning.py`가 spawn 방식의 여러 MuJoCo worker를 사용한다. shared observation을 N개씩 정책에 넣고 T×N 시퀀스를 현재 가중치로 다시 계산한다. 숨은 상태는 N행이며 reset/GAE 경계도 행별로 처리한다. 모든 worker가 한 episode처럼 reset되는 것을 방지하고 terminal observation의 value를 먼저 계산한 후 종료한 worker만 재시작한다. GPU는 정책·CNN·optimizer와 EGL 렌더링에 쓰고 물리는 CPU 프로세스로 병렬화한다.

Gaussian latent를 샘플링한 뒤 tanh로 action을 만들며 log probability는 tanh Jacobian을 포함한다. PPO clip, value loss, gradient clip을 사용한다. squashed entropy 근사는 사용하지 않는다. 같은 observation을 쓰는 critic이다. 주기적으로 checkpoint를 원자적으로 저장하고 progress를 갱신한다. checkpoint는 모델·optimizer·Torch CPU/CUDA RNG를 포함하지만 정확한 중단 재개 기능은 없다. 단일 reset rollout 방식은 긴 코스의 기억 평가를 위한 최종 학습기가 아니다. 원격 초기 학습은 명시한 X 범위에서 여러 경사·bank 구간의 시작점을 뽑는다.

초기 보상은 3D 진행량과 횡오차 비용, 실패 비용이다. 계획의 slip·충격·하중·서스펜션 여유 보상은 아직 없다. BC/DAgger와 고정 budget의 recurrent PPO 확장 순서는 연구 계획을 따른다. smoke PPO는 optimizer 동작 확인용이며 BC/DAgger를 건너뛴 과학적 결과로 사용하지 않는다.

## 영상 기록과 좌우 구동 후속

기존 `evaluate-suite --record` 구현은 정책의 원본 두 입력·외부 관찰 화면·명령/몸 센서/진단 trace를 HTML 재생과 raw NPZ로 저장한다. 사용자 지침에 따라 새 설명용 HTML 생성·갱신·자동 열기는 수행하지 않는다. 새 AC 기록·설명 경로는 Markdown+JSON/NPZ/PNG/영상으로 준비해야 한다. 관찰 화면·qpos/qvel은 진단 정답이며 actor 입력이 아니다. 이 문단의 legacy 정책은 속도/곡률2출력이고, 최신 retina_v1/residual 계약은 아래와 [시각·TV 구현](vision-tv-implementation.md)에서 구분한다.

이 문단은 legacy 경로다. 후속 `retina_v1`은 방향 grid·운동 필터와 유효성을 제공하고,
off/fixed/residual 구동은 3-action을 사용한다. off/fixed의 미사용 action은 확률비에서
제외한다. `residual`은 학습 가능한 yaw moment를 제한 배분기로 보낸다. 교사 sequence
BC와 checkpoint schema 복원을 포함한 최신 계약은 [시각·TV 구현](vision-tv-implementation.md)이다.

## 외부 데이터와 증거

CSV import는 ID·type·role과 directed aggregate synapse count를 검증해 npz로 저장한다. 역할 검토, ROI 추출, 좌우·시각 위치 대응, boundary loss·도달성 분석은 연구자가 별도로 확정해야 한다. `synthetic_demo`는 56개 합성 노드이며 실제 초파리 데이터가 없다. C7 edge swap은 뉴런별 차수·포트·가중치 배열을 보존하지만 도달성·경로 길이를 맞추지 않는다.

실행은 새 run 디렉터리를 요구한다. manifest에 config·src/config 파일별 SHA-256·설치 버전·Git revision을 남기고, 종료/실패 사유를 result에 기록한다. rollout은 센서·적용 의도·제어·정답 pose trace와 전체 step 시간 분위수를 남긴다. PPO는 interaction 수·loss·parameter 변화·checkpoint를 남긴다. 실행 중 Git dirty 여부까지 기록한 소스 해시를 기준으로 재현한다. 대규모 실험에는 원본 영상·학습 데이터 split·계산 장치·checkpoint replay·독립 시드 집계가 추가로 필요하다.
