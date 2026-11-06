# 현재 비행 actor의 CPU owner 연결 검증 — 2026-11-04

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

같은 몸 상태에서도 영상에 따라 조향이 달라지는 반응은 확인됐지만, 코너를 보고 미리 조향하는 기능이 충분하다는 결론은 아니다. 최신 fixture input_adjoint 후보의 동일 몸/native/clock/회로 prefix 영상 교환은 기대 방향25/30, 목표 margin .002 충족13/30이다. 원본 RGB의 동일 FlyVis 상태 복제 한 TRAIN 사례는 margin .001954729로 목표 미달이다. 이 수치는 이번 새 실험이 아니라 [직전 학습의 원본 결과](fixture-adjoint-learning-20261104.md)다. donor의 도로 횡위치 등도 달라 물리적으로 가능한 코너만의 반사실로 해석하지 않는다.

“코너를 다 외웠다”는 것은 현재 증거로 단정할 수 없다. 회전·속도 등 몸 감각으로 현재 조향을 추측하고 과거 영상/몸 감각의 회로 상태로 출력을 이어가는 shortcut과 같은 코스 반복 학습에 따른 암기 가능성이 있다. 몸만 쓰는 과거 비교군도 현재 조향 오차를 크게 줄였지만 진입 전 미래 조향에서는 상수보다 나빴다. 따라서 시각 기여가 작다는 결과를 독립적인 선행 경로 판단이 구현됐다는 뜻으로 읽지 않는다. actor에 지도·world pose·진행률·교사 명령을 주지 않는 정보 경계는 유지한다. 새로운 코너/트랙 일반화는 미검증이다.

이번에는 가중치를 추가로 반복 학습하지 않고 학습 모델과 실행 모델의 불일치를 수정했다. 기존 owner는 현재 SpatialMotionDNOverlay와 SymmetricNormalizedFlightSpeedCommand를 실행하지 못한다. 새 current_flight_runtime은 exact checkpoint SHA와 학습 소스 hash를 검증하고 부모 chain을 복원해 PowerBoundedSpeedDecisionRuntime을 만든다. 새 current_flight_owner는 CPU 자식에서 직접 복원한다. GPU 부모의 모델 객체를 넘기지 않는다. 이는 공학적 실행 경계이며 새 생리 회로나 실측 근육 구현이 아니다.

| 검증 | 결과 |
|---|---|
| 원본 native byte/camera join 기록 재생 | 128결정, 조향·페달 및 근육 출력 최대차0 |
| 기록 재생 최장 요청 | 14.919ms, 원래50ms 제한 유지 |
| 최신 시각·속도 경계 | 체크포인트와 타입/hash 복원 확인 |
| live native 읽기 전용 시작/상태/종료 | 42원본 사본, sampler/ticker 오류없음 |
| 프로토콜/strict loader | pytest18개 통과 |
| 소유 프로세스 | owner31956/30908, native producer45996 모두exit0·강제종료없음 |
| 학생 제어·시뮬레이터 변경API·desktop 자동화 | 모두0회 |

요청마다 순번과 실제 단계 시각을 기록하며 stage 메시지에 의해50ms deadline이 연장되지 않는다. 순서 불일치·timeout·EOF는 poison 처리하고 응답이 불명확하면 재시도하지 않는다. 에피소드 안의 숫자 상태 초기화는 금지한다. live 경로는 기존 NativeFlightBodyTicker와 원본 sampler를 사용해 몸50ms/영상150ms/source-render25ms 제한을 유지한다. 기록 재생은 별도 recorded_replay 모드로 원래 decision/receipt를 사용하며 이를 live 신선도 검증으로 세지 않는다.

원본은 runs/current-flight-owner-20261104/recorded-validation-02와 native-probe-01의 manifest/result 및 owner/client terminal/stage trace다. recorded-validation-01은 응답 연결 단절 및 deadline 처리를 보강하기 전의 결과로 보존한다. 최종 소스와 일치하는 것은02와 native-probe-01이다. 이전 checkpoint와 sealed source는 보존했다. 이번 weight update0·후보 승격없음이다.

AC는 Monza/Z4 GT3, generation1117858ULL:0.0016212, epoch7, paused=true/live=false에서 읽었다. live native startup은 성공했지만 카메라 publication0·추론 출력0이다. 정지한 센서의 과거 receipt를 갱신하거나 가짜 움직임을 만들지 않았다. 과거 moving-shadow의 unknown-response timeout 원인은 이번 짧은 시작 검사로 규명되지 않았다. 새 단계 기록을 움직이는 실제 영상/native 조합에 연결해야 원인을 좁힐 수 있다.

다음은 현재 owner를 사용하는 GPU 운동 시각 연결 및 움직이는 교사 관측의 read-only 정합 검사다. 원래 신선도와 no-retry 경계를 통과한 뒤에만 학생의 제한된 폐루프 조향을 비교한다. 같은 몸·동일 회로 상태에서 영상 변경의 선행 조향 방향/크기, 전체/진입 전 조향 보호, 새 코너와 타이어·셋업 조건을 함께 평가한다. 조향 외 가감속·차체 기울기·회전·관성·하중·접지/타이어 상태 되먹임, 상대풍·변속·차종 적응 및 빠른 유효 완주·레이싱 라인/랩타임은 계속 미완료다. 이번 연결 검증을 실제 조향 개선이나 목표 완료로 세지 않는다.
