# 실제 제동·하중 되먹임 계측 경로

2026-10-29, Asia/Seoul. [대전제](PROJECT_CHARTER.md)를 유지한다.
**30km/h 부근에서 실제 브레이크와 속도·바퀴 하중 반응을 원시 센서로 확인했다.**
이번 것은 공학적 교사 개입 시험이다. 초파리 회로의 제동 학습·운전 최적화 성공이 아니다.
일반 주행 목표 30km/h와 사용자 속도 상향 선호를 유지한다.

## 구현

설치된 1차 CSP SDK의 `physics.setAIStopCounter(carIndex, time)`는 AI에 제동 대기를 요청한다.
시그니처/주석·SDK 해시는 `assets/external/csp-ai-brake-api-evidence-20261029/source.json`에 보존했다.
API 인수의 시간과 실제 브레이크 작동 시간은 동일하다고 가정하지 않았다.

새 `expert-brake-pulse30` 프로파일은 기존 유한 30초 AI 주행의 소유권·카메라·속도/거리 제한을 유지한다.
3초 이상 경과하고 25~35km/h일 때 `setAIStopCounter(0, .4)`를 **한 번만** 요청한다.
불확실한 setter 결과는 재시도하지 않으며 종료 시 counter를 0으로 해제하고 AI를 끈다.
이후 native 정지·full brake 확인 전까지 기존 종료 절차를 유지한다.
시간·프로파일·API가 달라지면 요청을 거부한다. 학생 명령이나 키보드/마우스 조작은 없다.

서버는 요청 뒤 1.5초 범위에 brake>0.05와 1km/h 이상 속도 감소를 둘 다 요구한다.
마지막 정지 복구의 full brake를 주행 제동 반응에 섞지 않는다.
새 프로파일은 자연스러운 AI 정답과 구분하며 기존 `common_expert_settings` 라벨러가 명시적으로 거부한다.
별도 정답 버전 없이 과거 학습 자료에 섞이지 않게 했다. flight shadow와 동시 사용도 거부한다.

배포 전 관련 검사 **219개**와 Ruff가 통과했다. 실제 Lua 상태 기계에서 단발 요청·불확실한 결과·해제,
시간 인수 변경/없는 API의 거부, 제동 없는 속도 감소와 종료 제동의 오판 방지를 검사했다.
검사 중 heartbeat의 없는 physics 처리와 실패 경로의 시험 시각을 수정한 뒤 통과했다.
실제 native AI 표시와 CSP AI-off의 기존 불일치는 해결된 것으로 바꾸지 않았다.

## 실제 실행과 독립 센서 결과

t1464에서 이미 정차·일시정지·AI/camera/probe-off인 상태에 새 Lua를 원자적으로 배치했다.
게임 엔진을 재시작하지 않았고 UI 조작은 0이다. 새 runtime generation은
`4525688ULL:0.0011165`이며 새 capability를 실제 heartbeat에서 확인했다.

t1465는 개발용 진단 역할을 사전에 고정한 30초 기록이다. 738관측/카메라 대기 누락 2회였다.
t1466의 모든 원시 actor 재생 차이는 0이지만 최대 관측 간격 122.284ms/80ms 초과 1회가 있어
연속 40ms 학습 자격은 false로 유지했다. 도움선 OFF는 확인했다. 정답 생성이나 fitting은 없다.

t1467은 원시 native bytes를 각각 해시 검증하고, camera의 원래 causal clock 교환을 재구성했다.
제동 요청의 host receipt 시각 불확실성은 약 3.316ms다. 그 범위 뒤의 서로 다른 native packet만 사용하고,
1.5초 이내 반응을 별도로 대조했다. 물리적 명령/노출 시각의 동기화 인증은 아니다.

| 항목 | native 관측 |
|---|---:|
| 요청 직전 속도 | 30.8319km/h |
| 1.5초 이내 최소 속도 | 0.06290km/h |
| 최대 브레이크 | 1.0 |
| brake>0.05인 native 표본 | 38 |
| 해당 표본 첫~끝 receipt 간격 | 1.17033초 |
| 요청 전 0.5초의 평균 앞바퀴 하중 비율 | 0.51164 |
| 제동 표본의 평균 앞바퀴 하중 비율 | 0.62797 |

따라서 [이전 속도 지시 감소](flight-new-region-speed-transition-20261029.md)의 타력 감속과 달리
실제 브레이크·차량 반응을 기록했다. `time=.4` 요청을 실제 0.4초짜리 제동이라고 부르지 않는다.
표본 간격은 정확한 물리 작동 지속시간도 아니다. 앞바퀴 하중 비율은 원시 wheelLoad에서 계산한
이 한 시험의 관측치이며, 범용 하중 제어/타이어 적응/생리적 기능 검증이 아니다.

## 학습에 연결하기 전 남은 일

이번 경로는 강한 제동으로 거의 정지했다. 일정 30km/h 과제의 최적 행동을 가르치는 정답은 아니다.
원래 30km/h 목표와 강제 정지를 그대로 묶어 imitation label로 쓰면 목적이 모순된다.
이 기록은 우선 제동의 감각·물리 반응 계측용으로 보존한다.
새 학습 과제에는 actor에게 허용된 명시적 감속/정지 목표와 실제 요청·응답의 시간 계약이 필요하다.
좌표·미래 경로를 actor에 넘겨 이를 대신하지 않는다.

다음은 실제 제동 반응을 이용한 가상 몸 제어 식별과 목표에 맞는 감속 과제다.
개발 역할/fit 금지를 유지하고, 새 TRAIN/독립 평가 경로는 개발 구간과 분리해 수집 전에 정한다.
관측 공백과 source의 연속 학습 자격 false도 유지한다. 기존 신경 모델/가중치는 수정하지 않았다.
시각 일반화, 비행 명령/운동 회로와 빠른 몸 피드백 통합, 기울기·하중·타이어/셋업 적응,
학생 단독 완주와 빠른 레이싱 라인/랩타임은 모두 계속 검증해야 한다.

마지막 새 heartbeat는 paused/AI-off/camera-off/probe-off, 0.00923156km/h·error 없음이었다.
모든 계측/감사 프로세스는 종료 코드 0이다.

## 근거 위치

- 설정: `configs/flight-development-brake-pulse-20261029.json`.
- 코드: `integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua`,
  `scripts/control_ac_session.py`, `scripts/record_ac_engine_ai_probe.py`,
  `scripts/audit_ac_brake_intervention.py`.
- 로컬 원본: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/`의 t1464~t1467 폴더.
- NAS compact 보존: `runs/flight-actual-brake-feedback-20261029-t1468`.
  대용량 원시 영상은 로컬에 유지하며 보존 영수증의 해시/크기/경로로 참조한다.
