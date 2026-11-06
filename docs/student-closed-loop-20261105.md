# 현재 flight 학생의 출력 계약·전달·정지와 실제 AC 연결 시험

**후속 상태 — 2026-11-05:** [출발 원인 분리·저속 학습·실제 재비교](student-launch-learning-20261105.md)까지 진행했다. 아래는 그 이전 실행/인계 기록이다.

2026-11-05, Asia/Seoul. 사용자 요청에 따라 구현·테스트와 실제 부모/후보 비교를 수행했다. 원본은 `runs/student-closed-loop-20261105/`다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 판정

현재 통합 flight actor의 실제 픽셀→운동 시각→몸 감각/재귀 회로→canonical3→SDK→native 피드백→독립 제동/해제를 연결했다. 부모와 dual_guard 후보를 실제 초기 타이어 온도 40/80°C 조건에서 비교한다. 예정 0.8초의 연결 완료와 자율 주행 성공은 구분한다. 실제 출발 기준은 최고속도 0.5km/h 이상과 변위 0.02m 이상이며, 이를 못 넘으면 `completed_student_actuation_without_motion`으로 남긴다. 유효랩·자율 완주·한계 주행은 입증하지 않았다. 후보 승격·추가 학습·가중치 갱신은 없다.

AC 데스크톱 입력은 0회다. 세션 restart/start/pause/resume은 검증된 CSP lifecycle/file API, 감각과 제어는 background MMF/vJoy만 사용했다. 설치 Lua를 변경하지 않았다. 출력의 throttle/brake 동시 활성도 임의로 제거하지 않았다. 교사가 운전하는 moving shadow와 실제 학생 제어 기록은 별개다.

## 구현과 검증

[출력·전달·정지 계약](current-flight-output-contract-20261105.md)에 단위와 실행 경계를 정리했다. 기존 봉인된 loader/owner/ticker 대신 `src/flybrain_drive/current_flight_*.py`에 버전 경로를 추가했다.

- canonical steering −1..1에 학습 라벨의 역변환 ×0.6을 한 번 적용하며 throttle/brake는 독립 0..1이다. 비정상·범위 밖 값은 거부한다. 구형 intent2 codec은 사용하지 않는다.
- 단일 회로 owner와 immutable 최신 결정 mailbox를 연결했다. 원본 body 최초 receipt·네 view render 경계·capture/publication/decision·episode/generation·checkpoint hash를 보존한다. 조회는 추가 추론을 하지 않는다. 회로 상태는 에피소드 동안 유지한다.
- 실제 적용 제한 body50ms/view150ms/source-render25ms/RPC50ms를 유지했다. 입력 선택 view25ms와 준비 body25ms/view75ms는 더 엄격한 여유다. 대기 중 원래 시각을 새로 찍거나 제한을 늘리지 않는다.
- lean CPU actuator가 독립적으로 vJoy를 소유하고 명령 유효기한을 감시한다. 요청 시도는 SDK 전 기록한다. unknown/지연/실패는 재시도 없이 제동한다. caller stall도 별도 실제 subprocess 시험으로 검증했다.
- stop ACK 뒤 추론을 중단하고 독립 owner가 서로 다른 native packet 3개에서 속도≤0.1km/h·gas≤0.02·brake≥0.98·교정된 중립을 확인한다. 그 뒤 SDK FREE, CSP pause, 소유 프로세스 원래 handle의 exit를 별도로 기록한다.

**최종 전체 검사:** `tests-03/result.json`: 173개 중 172개 통과·1개 Windows 해당 없음 skip, 실패/오류0, 원래 child handle exit0. 계약 경계·정지·unknown·타임스탬프/세대/중복·동시 publication·상태 유지·lean spawn·실제 caller stall과 기존 관련 검사를 포함한다. `owner-parity-02`는 부모/후보 각각 원본128결정의 출력과 근육 최대차0이다. 실제 moving shadow는 부모 `parent80-moving-03/run` 488출력/113영상, 후보 `candidate80-moving-06/run` 477출력/111영상이며 독립 전체 재생 최대차0이다.

SDK raw와 native steer raw는 다른 척도다. 기존 교정32표본의 원본 bytes/hash를 재해석하여 SDK 이득449.999949/request와 native 이득1.406249901/request를 분리했다. native stop 중립 허용치는0.001562500066이며 SDK raw0.5에 대응한다. 물리 각도 단위·전 범위 선형성을 주장하지 않는다. 초기 `parent80-student-audit-02`의 SDK 이득을 native에 적용한 비교는 잘못된 당시 감사로 보존하며 최종 감사에 쓰지 않는다. 원본 데이터와 실제 중립값0은 변경하지 않았다.

실제 학생 영상의 별도 독립 감사도 완료했다. 후보 초기 실패 run의17묶음(`candidate80-student-pixels-01`)과 최종 부모80 시험의23묶음(`parent80-student-pixels-04`)을 원본 RGB bytes·네 view 원본 시계로 처음부터 재구성했다. 최종23묶음의 scene/static/fine motion/coarse 최대차는 각각8.9407e-8/0/4.4703e-8/2.9802e-8이며 연속 시각 상태를 유지했다. GPU 감사의 원래 launcher handle도exit0이다.

## 고정 checkpoint와 재현성

| 모델 | 원본 checkpoint | SHA256 |
|---|---|---|
| 부모 | `runs/fixture-adjoint-learning-20261104/input_adjoint-fit-01/ports.pt` | `723286d6f79401e266d24a7c384680f95cf3edbabf7807139bfb39177dea6812` |
| 후보 | `runs/dual-reference-learning-20261105/dual_guard-fit-01/ports.pt` | `8de4aad03027e36dc8dcd2655745f6a3769af8ee254f9095e2cdd6ef9d0c6099` |

seed781605, 단일 회로 owner, 동일 최종 구현과 동일 제한을 사용했다. 각 run에 source/hash/config·원본 픽셀 archive·몸 감각 원본 bytes/clock·선택된 시각 특징·회로 출력·SDK 시도/영수증·실제 native 피드백 bytes·종료 사유를 보존했다. 최종 네 조건은 각1회로 통계적 우열·온도 적응 검증이 아니다. 실제 온도는 설정값과 구분해 아래 기록한다.

## 최종 동일 구현 비교

| 조건 | 첫 학생 요청 뒤 실제 네 바퀴 온도°C | SDK accepted | 첫~마지막 요청 초 | 최대 km/h | 최대 변위 mm | 평균 throttle/brake |
|---|---|---:|---:|---:|---:|---|
| [부모40](../runs/student-closed-loop-20261105/parent40-student-audit-01/result.json) | 39.874/39.874/39.882/39.882 | 64 | 0.799877 | 0.111955 | 1.321297 | 0.054446/0.129301 |
| [후보40](../runs/student-closed-loop-20261105/candidate40-student-audit-02/result.json) | 39.874/39.874/39.882/39.882 | 62 | 0.795633 | 0.085040 | 1.388136 | 0.059491/0.100732 |
| [부모80](../runs/student-closed-loop-20261105/parent80-student-audit-04/result.json) | 79.468/79.468/79.497/79.497 | 64 | 0.805636 | 0.112897 | 2.304324 | 0.081082/0.094755 |
| [후보80](../runs/student-closed-loop-20261105/candidate80-student-audit-02/result.json) | 79.461/79.461/79.492/79.492 | 64 | 0.811456 | 0.112016 | 2.182449 | 0.076218/0.086554 |

네 조건 합계 학생 SDK accepted **254회**, 전체 회로 출력 **462개**를 원본 시각과 재생해 최대차0을 확인했다. SDK 호출 완료 시 최악 body age 24.862ms/view age 129.874ms로 원래50/150ms 안이다. 모든 조건에서 두 페달>0.02의 동시 활성 비율은100%다. 네 조건 모두 `completed_student_actuation_without_motion`, stop reason `explicit_stop`이며 native 중립 정지·SDK FREE·CSP pause·CPU owner/actuator/GPU의 원래 handle exit0을 확인했다. 단 한 번의 조건 비교이고 출발하지 못했으므로 후보의 실제 주행 우열이나 온도 적응을 판정하지 않는다.

최종 비교 전의 실패한 학생 시험까지 합치면 SDK accepted311회다. [전체 원본 요약](../runs/student-closed-loop-20261105/comparison-summary-01.json)은 inner actuator 영수증 기준이며 teacher 주행·준비 제어·긴급 제동은 학생 명령으로 세지 않는다.

## 실패와 수정의 원자료

각 이름은 `runs/student-closed-loop-20261105/` 아래다. 실패한 실행을 성공 기록으로 교체하지 않았다.

| 원본 | 관측/처리 |
|---|---|
| `preflight-01` | tyres out1이라 학생 시작 불가. 다음 CSP reset 뒤 재검사 |
| `parent-moving-01` | 추론 전체 lock으로 latest RPC 최대35.6ms, stale view. 22출력 재생차0이나 전체 실패; 당시 ticker snapshot 보존 |
| `parent-moving-02`, `candidate-moving-04` | 실제 온도가 허용대 밖이라 제어 전 거부. 카메라 정지 진단을 일회 온도 setter 앞으로 옮김 |
| `candidate80-moving-01`, `candidate80-moving-02` | stopped terminal과 failed heartbeat 불일치, completed 전용 진단 부적합. 제어0; GPU interpreter teardown도 실패. 해당 생성시각/parent/명령·미시작을 확인한 종료 영수증 보존 |
| `camera-reconciliation-01` | 별도 정지 카메라60프레임이 정상 completed/heartbeat를 복원. 제어 장치 획득0. 이후 운전은 정상 startup gate 적용 |
| `candidate-moving-03` | Lua pin 인자 오타로 사전 거부. 공통 PIN 기본값 사용으로 수정; 설치 Lua는 변경 없음 |
| `candidate80-moving-05` | 느린 monitor가 이미 만료한 최신 결정을 조회해81출력 후 종료. 빠른 read-only observer로 분리; 전체 실패와81출력 재생차0 모두 기록 |
| `parent80-student-01` | 첫 context의1271행 backlog 처리43.5ms, 준비 실패·학생0. RPC idle 중 원본 몸 이력 수집, lean child bootstrap 도입 |
| `parent80-student-02`, `parent80-student-03`, `candidate80-student-01` | 각각4/41/12회 실제 학생 요청 후 source lease 만료. 모두 제동·해제·pause 확인. 부모03은 당시 actuator raw feedback bytes가 없어 소급 감사 한계가 있음; 새 버전은 원본 bytes 저장 |
| `candidate40-student-01` | 1단 준비 중 file telemetry timeout으로 학생 시작 전 중단·학생0; final-pause 완료·gear child exit1. 새 디렉터리 candidate40-student-02에서 조건 전체 재준비 |
| 위 stale source 실패 뒤 | feature 계산 전 원본view≤25ms 선택, 제한된 pending handover, defer busy-spin 제거. 원래view150ms를 넘는 pending은 계속 거부; 회로 중간 reset 없음 |

이전 외부 sequence 요약 중 실패 시 학생 수 기본값0인 항목이 있다. 실제 요청 수는 **inner `run/actuator/actuator-terminal.json`의 영수증과 events**가 권위이며 요약기는 이를 읽는다. unknown 시도와 SDK accepted도 구분한다. 과거 미해결 orphan PID는 이번 소유 프로세스로 오인해 정리하지 않았다.

## 남은 연구와 다음 실행

이번 결과는 연결·실시간 실패 처리의 검증이다. 실제 출발과 충분한 조향/가속/감속 반응은 통과하지 못했다. 정지 출발 시험은 학습 지원 속도13–74m/s 밖이다. 두 페달 동시 출력과 작은 throttle을 확인했지만 이것만으로 원인을 확정하거나 모든 속도에서 운전 불가라고 결론 내리지 않는다. 온도별 출력 차이는 실제 타이어 grip 적응 증명이 아니다.

다음은 고정 actor의 정지/저속 지원 범위와 gear/clutch/페달·실제 차량 반응의 단위/시각을 진단하고, 기존 실패 원자료를 이용해 출발 조건의 결손을 분리하는 것이다. 임의 throttle boost·brake 삭제·교사 혼합으로 현재 학생의 주행 성공을 만들지 않는다. 추가 증류/튜닝에 앞서 누락된 시각·비행 제어 부품과 감각 되먹임의 반응/인과 기여를 우선한다. 이후 역할을 고정한 코너/몸 조건 비교와 경로 제거·지연 개입, 실제 학생의 자세 안정화·가감속·하중 이동·접지/타이어·셋업/차종/상대풍·변속 적응, 반복 유효랩·빠른 레이싱 라인/랩타임 검증으로 이어간다. actor에 지도/world pose/절대 heading/교사 action을 주입하지 않는다. 실측 뉴런·생리 근거·기능 근사·차량 대응 가설·학습 어댑터를 계속 구분한다. 목표는 active다.

## 최종 외부 상태와 실행 종료

`final-preflight-01/result.json`의 읽기 전용 재관측에서 CSP paused=true·camera_active=false·native AI=false·vJoy FREE·brake1·속도0.000460km/h를 확인했다. 이번 최종 네 시험의 CPU actor/native producer/actuator와 GPU launcher는 모두 원래 handle에서exit0이다. 마지막 픽셀 감사도exit0이며 이 작업의 제어·감사 실행은 종료했다. 과거 orphan의 미해결 상태는 별개이며 임의 종료하지 않았다. CSP 누적 heartbeat_write_failures10도 원본 상태에 보존했다.

최종 소스43파일·고정 checkpoint·기존 HEAD/branch는 `final-source-receipt-01.json`에 기록했다. 기존 dirty workspace와 원본 결과를 유지했고 commit/push/배포를 수행하지 않았다. `comparison-summary-01.json`, 각 감사/테스트의 result·원래 handle 영수증, 최종 문서 hash는 `final-review-receipt-01.json`에서 추적한다. 최신 보고 이후의 다음 작업은 위 출발/저속 범위·부품 인과 검증이며 실행 중인 학습이나 학생 주행으로 인계하지 않는다.
