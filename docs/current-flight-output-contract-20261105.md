# 현재 flight actor의 출력·전달·정지 계약

**후속 구현 — 2026-11-05:** `pilot_mailbox_flight_control`은 기존 canonical 3축 계약과 원본 결정 시계를 그대로 사용한다. CPU 생산자와 독립 CPU actuator를 단일 슬롯 공유 메모리로 직접 연결한다. CPU actuator는 lease, 명시 정지, native guard를 확인한 뒤에만 새 결정을 SDK에 보낸다. 계획 관측 완료와 source 만료를 분리하며, 원본 정지 증명/반환 뒤 소유 카메라의 terminal echo와 같은 세션의 idle heartbeat를 확인한 후에만 기존 CSP pause를 발행한다. [후속 실행·실패 근거](lease-balanced-learning-20261105.md). 아래 초기 연결 계약과 당시 RPC 경로 기록은 과거 결과를 재현하기 위해 보존한다.

2026-11-05, Asia/Seoul. 구현은 `current_flight_control_*`, `current_flight_actuation`, `current_flight_actuator_process`, `current_flight_dispatcher`에 추가한다. 봉인된 기존 loader/owner/ticker와 과거 manifest는 수정하지 않는다. 실행 결과는 별도 폐루프 보고서와 `runs/student-closed-loop-20261105/`를 따른다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 이 문서는 공학적 제어 연결의 계약이며 생리 검증·자율 완주를 대신하지 않는다.

## 출력의 의미

현재 6-MN head는 modeled MN 활동을 canonical steering/throttle/brake 3값으로 판독한다. 실측 근력이나 2값 비행 intent가 아니다. 구형 `flight_actuator.canonical_controls`를 적용하지 않는다.

| 채널 | 모델 단위/범위 | 현재 SDK 요청 |
|---|---|---|
| steering | 기존 교사 canonical 정규화 값 −1..1, 좌회전 음수/우회전 양수 | 값 × 0.6 |
| throttle | 독립 정규화 페달 0..1 | 그대로 |
| brake | 독립 정규화 페달 0..1 | 그대로 |

근거: `scripts/prepare_synchronized_sac_student.py`의 라벨은 `[control['steering']/.6,control['throttle'],control['brake']]`다. `scripts/train_qualified_sac_unified.py`가 이 라벨을 로드하고 현재 checkpoint 계보가 해당 학습을 잇는다. 따라서 0.6은 같은 축 프로파일에서 원래 SDK 요청을 복원하는 배율이다. 기존 데이터의 현재 조작 감독을 미래 경로 정답이나 실제 적용 피드백으로 바꾸어 부르지 않는다.

같은 프로파일·DLL·장치·producer session의 32표본 교정은 SDK raw/정규화 요청은 449.999949이고 canonical에 해당하는 SDK raw 이득은 ×0.6≈270이다. 같은 원본 native bytes의 별도 이득은 1.406249901/native raw per SDK request이며, canonical1에 대한 native raw는 약0.84375다. SDK raw와 native raw는 서로 다른 척도이며 물리 각도 단위나 전 범위 선형성을 새로 주장하지 않는다. `validate_calibration(...,steering_request_scale=.6)`와 현재 hash/장치 식별을 적용한다. 다른 차/프로파일에 상수를 그대로 일반화하지 않는다. 범위를 넘거나 비정상인 값은 clipping으로 감추지 않고 거부한다. 두 페달 동시 출력도 그대로 기록하며 임의 상호 배제하지 않는다.

- 타력: steering=0, throttle=0, brake=0. 정지 명령이 아니다.
- 준비/오류/종료: steering=0, throttle=0, brake=1. 학생이 학습한 제동으로 세지 않는다.
- raw prediction, 배율, 최종 요청, SDK 시간 경계, 실제 native 피드백을 별도 보존한다. driver acceptance는 게임 적용 증거가 아니다.

## 원본 시각에 묶인 결정

`FlightDecisionEnvelope`는 run·episode·generation·checkpoint SHA256, 증가 decision ID, publication/capture ID, 몸 packet/최초 receipt, 네 view render lower/upper와 원본 body first-copy-start, camera receipt, publication·추론 시작/완료, canonical3을 담는다. Python frozen dataclass와 tuple을 사용한다.

기존 단일 ticker가 회로를 진행한다. owner는 RPC 대기 중 원본 몸 감각 이력을 순서대로 처리한다. 이력 준비는 회로를 진행하지 않고 원본 receipt도 다시 찍지 않는다. 첫 실시간 학생 준비에서 1,271행의 뒤늦은 처리가 43.5ms를 소비했던 실패를 보존했다. 새 mailbox의 조회는 추론을 재호출하지 않는다. publication context는 owner가 실제 선택해 발급한 context ID로 연결한다. 카메라 재사용 중 몸 packet이 갱신되는 정상 결정을 허용하지만 같은 decision의 재적용은 거부한다. publication의 visual/context/source witness를 짧은 condition lock 안에서 함께 선택한다. 추론 중에도 mailbox 조회와 다음 publication을 허용하고, 선택 당시의 witness로 결과를 봉인한다. 추론 전체를 lock 안에서 실행하던 첫 버전은 조회 지연으로 실패해 교체했다.

조회와 실제 SDK 적용 직전에 원본 시각을 재검사한다. body≤50ms, oldest view≤150ms, source-to-render 상한≤25ms, RPC≤50ms를 유지한다. SDK 호출 종료가 source deadline을 넘은 경우에도 원래 요청을 기록하고 제동한다. 재수신 시 시각을 새로 찍지 않는다. stale/future/세대·순서 불일치/EOF/unknown은 오류 종료하며 새 run 전 자동 재시도하지 않는다.

## 준비와 독립 정지

GPU 호출자와 별도인 CPU actuator process만 vJoy를 소유한다. 해당 프로세스는 lean bootstrap으로 GPU pilot main과 PyTorch를 불러오지 않고, 실제 실행 소스 hash·PID·Python 경로와 torch 미로딩을 ready 응답에서 확인한다. 준비 중에는 중립/full brake이며 학생 요청은 0이다. 최초 유효 publication 후 0.5초 이상·유효 camera group 3개 이상, 최근 실제 5개 출력의 축별 범위 최대 0.05 이하를 확인한다. 또한 가장 최근 publication/capture의 실제 출력이고, 준비 판단 시 원본 body age≤25ms·view age≤75ms인 경우에만 arm한다. 이는 준비 여유이며 적용 한도 body50ms/view150ms를 바꾸지 않는다. 준비 영수증의 파일 쓰기는 제어가 끝난 뒤 수행한다. 수치 재귀 상태를 중간 초기화하지 않는다. readiness는 지속 진동·시각 기능·운전 능력의 증명이 아니다.

각 명령의 유효기한은 `min(body_first_receipt+0.05, oldest_render_lower+0.15)`다. 새 결정이 끊기거나 호출자가 멎으면 독립 owner가 다음 poll에서 제동한다. 매 poll은 실제 host clock을 사용하며 OS 스케줄 지연의 절대 상한을 보장한다고 주장하지 않는다. 실제 지연은 원본 로그에서 확인한다. 준비 owner도 총 10초로 한정한다.

기본 첫 학생 trial은 예정 0.8초·강제 상한 1초/100결정·10km/h·적분 travel 및 native 진단 변위 1m·tyres out 0·새 damage 0·forward gear이며 외부 진실값은 guard에만 있다. 이는 학습 지원 하한 13m/s보다 낮은 연결 시험이며 운전 성능 평가가 아니다. scene이 끊길 때 한도를 늘리거나 오래된 출력을 유지하지 않는다.

정지 후 최대 3초 동안 brake를 유지하며 서로 다른 native packet 3개에서 speed≤0.1km/h·gas≤0.02·brake≥0.98·native steer raw≤0.0015625001(기존 SDK raw0.5에 대응하는 실측 교정 허용치)를 확인한다. 이후 driver를 해제하고 SDK FREE 상태를 확인한다. 증명 실패도 보존한다. caller가 timeout이어도 제동 owner를 무단 terminate하지 않는다. 제동 요청을 먼저 확인한 후 actor 추론을 즉시 중단하고, 독립 owner가 실제 정지·해제를 입증하도록 기다린다. actor/카메라 정리·원본 archive보다 제동과 해제를 먼저 수행하고 CSP API pause 결과도 별도 기록한다.

## 검증과 한계

계약/타임스탬프/중복/지연/EOF/SDK 늦은 반환/정지 실패와 별도 프로세스의 caller stall을 검사한다. 부모/후보는 각 엄격 loader와 checkpoint hash로 복원하고 원본 128결정 parity·각각의 moving shadow·학생 trial을 구분한다. 실제 조향/가속/감속 반응과 source age, 차체 회전·가속도·하중·네 바퀴 실측 온도를 원본에 남긴다.

초기 온도별 차이는 실제 주행 중 온도 적응 증명이 아니다. 짧은 연결 시험 뒤에도 겹눈 운동 시각과 비행 유래 균형·운동의 인과적 기여, 코너 선행 판단, 가감속·타이어/셋업·차종·상대풍/변속 적응, 반복 유효랩과 빠른 레이싱 라인/랩타임 검증이 남는다.

## 카메라 상태 불일치와 GPU 종료

CSP의 이전 immutable terminal이 stopped인데 주기 heartbeat가 Cannot commit camera metadata/failed인 사례를 보존했다. 실제 제어는 항상 정상 PausedMMFStartupGate를 통과해야 한다. completed 전용 진단을 stopped에 확대 적용하지 않는다. 기존 stopped-metadata 진단 조건을 만족하는 별도 정지 카메라 시험만 허용하고, 제어 장치를 획득하지 않은 새 60프레임 요청이 정상 completed와 일치하는 heartbeat로 끝난 뒤 일반 시작 검사를 다시 받는다. 이 진단은 새로운 epoch의 온도 설정 전에 수행하며, 초기 온도 setter는 이후 한 번만 실행한다.

GPU interpreter 종료 실패도 실제 실패로 남긴다. 새 worker entry는 target의 finally가 끝나고 소유 제어·추론·센서 자식이 종료한 뒤 현재 interpreter에 명시적 종료 경계를 적용하며 원래 launcher handle의 exit를 보존한다. 검증된 Windows System32 conhost는 비제어 console helper로 구별한다. 이전 실행의 PID를 재사용된 숫자만 보고 종료하지 않는다. 이번 작업에서 생긴 두 preflight GPU 종료 실패는 원래 parent·생성시각·명령·제어 미시작을 확인한 별도 종료 영수증을 남겼다.

실시간 오류 원자료를 통해 GPU 특징 계산 전에 원본 view age≤25ms인 새 카메라를 선택하도록 추가 여유를 둔다. 실제 적용 제한 view150ms는 그대로다. source-qualified 다음 영상의 계산 중에는 제한된 handover 예고를 사용하되 대기25ms·원래view150ms를 넘기지 않는다. 지연 중 반복적인 sleep(0) busy spin을 제거해 몸 감각 수신·발행 스레드가 진행하도록 한다. 소유 회로는 여전히 하나이며 실제 clock 간격을 기록하고 중간 상태를 초기화하지 않는다.

최종 실행 근거: [학생 연결·비교·실패 보고](student-closed-loop-20261105.md). 최종173검사 중172통과/1skip, 부모/후보 각 초기40/80°C의 SDK254회와462출력 재생차0을 확인했다. 네 조건 모두 실제 출발 기준 미달이며 정지·해제/종료 검증과 운전 능력 판정은 구분한다.

2026-11-05 후속 실행: [출발 분리 진단·저속 학습·부모/후보 비교](student-launch-learning-20261105.md). 신규 head는 기존6 modeled muscle만 사용하는 지도 학습 페달 등록이며 canonical3 이후 보정은 없다. 추가로 pending 새 영상 전환에 두25ms 추론 슬롯을 예약해 불필요한 옛 영상 추론을 줄였다. 실제 대기25ms와 적용 body50ms/view150ms 등 원래 유효기한은 유지한다.

후속187검사 중186통과/1skip이다. pending 전환 대기는 이전 수치 상태의 held body age≤25ms일 때만 허용하며 실제 몸 감각 진행과 원래50ms 검사를 유지한다. 준비는 빠른 읽기 전용 observer로 실제 출력을 소비하고 observer를 멈춘 뒤 arm 직전에도 최신 publication/원본 시각·안정 조건을 재검사한다. 실제 source lease 만료가 남아 모든 지연을 해소했다고 주장하지 않는다. armed 진단 변위는 전체 제동 후 총 이동 거리의 hard bound가 아니다.
