# 엔진 AI API 정차 시험 — 2026-10-28

실제 AC에서 화면 조작 없이 기본 AI를 활성화하고 0.5초 뒤 해제하는 시험을 완료했다. 이는 향후 **실행된 전문가 운전 자료**를 수집하기 위한 API 호환성 검증이다. 파리 모델의 운전 성공이나 기본 AI의 주행 성능을 입증한 결과는 아니다.

## 실제 결과

| 항목 | 증거 |
|---|---|
| 설치·재로딩 | `runs/ac-stationary-ai-probe-deployment-20261028-t585` |
| 소유 요청 1개와 API 영수증 | `runs/ac-stationary-ai-probe-20261028-t586` |
| 독립 후속 상태 조회 | `runs/ac-after-stationary-ai-probe-status-20261028-t587` |
| 검사 | 실제 Lua 제어 흐름, Python 클라이언트, 경로·시계·카메라 기하 관련 55개 통과, Ruff 통과 |
| 엔진 | 기존 BMW Z4 GT3 / Monza 오프라인 엔진, HWND 5836050 |
| 활성화 확인 | `car.isAIControlled == true`를 lease 도중 관측 |
| 종료 | AI false, paused true, expert probe inactive, lifecycle 오류 없음 |
| 위치 변화 | CSP 위치를 기준으로 최대 0 m |
| 독립 native 종료 계측 | 속도 약 0.001619 km/h, gas 0, brake 1 |
| 전면 UI·키·vJoy 조작 | 모두 0; 시험 전후 게임 창이 전면 창이 아님 |

새 설치 Lua SHA-256은 `854db49cba6bc6ac0728e049eb90eda15182dabea4ca4d883e040f6a67a86fb2`이다. 이전 `94c5781f…` 설치 파일은 t585에 원본 바이트로 백업했다. 최초 t584 시도는 설정 파일을 Windows 기본 cp949로 읽는 오류로 **설치 전에** 종료됐다. UTF-8을 명시한 t585만 실제 설치를 수행했다.

## 제어 계약과 검사 범위

`scripts/control_ac_session.py --command expert-probe --execute --output <새 실행 폴더>`는 일시 정지·AI 해제·정차·브레이크 유지·카메라 비활성·vJoy FREE·오프라인 차량/트랙/설치 해시 일치를 확인한다. 요청 허용값은 속도 제한 15 km/h와 lease 0.5초로 고정했다. 이 명령에는 resume 단계가 없다.

서버는 accepted 영수증을 먼저 저장하고 다음 setter를 각각 한 번 호출했다.

1. `physics.setAITopSpeed(0, 15)`
2. `physics.setCarAutopilot(true, false)`
3. lease 종료 시 `physics.setCarAutopilot(false, false)`
4. AI 해제를 관측한 뒤 속도 제한을 SDK 기본 무제한 값 `1e9`로 돌림.

setter는 SDK에서 void 함수이므로 `returned: true`는 예외 없이 반환됐다는 뜻이다. 활성화 성공은 별도 AI 상태 getter로 검증했다. 속도 제한의 실제 주행 중 준수 여부는 검증하지 않았다. 이전 임의 속도 제한값을 읽어서 복원한 것이 아니라 SDK 기본값으로 설정한 것이다.

게임 내 update 감시가 lease를 종료하므로 Python 응답 대기가 사라져도 정리 단계가 진행된다. 앱 자체의 강제 종료·게임 프로세스 정지까지 독립적으로 보장하는 감시는 아니다. 정상 onRelease에서는 AI 해제를 한 번 요청하고 accepted 기록은 재실행하지 않는다. AI setter 예외·종료 상태 미관측은 unknown으로 보존하며 같은 요청을 재시도하지 않는다. 파일 쓰기 실패는 영수증 저장만 재시도한다.

움직임·위치 소실·일시 정지 해제·AI 미관측은 성공 판정을 막는다. 활성 lease 중 새 lifecycle 요청과 카메라 수집 시작도 막는다. 이러한 실패 경로와 teardown 미관측·앱 reload의 비재실행은 실제 Lua를 실행하는 테스트로 검사했다.

## 실제 전문가 자료 수집으로 이어갈 조건

다음 단계는 **시작 위치 재시작 후** 별도 명령으로 짧은 저속 AI 주행을 검증하는 것이다. 기존 학생 모델의 제어와 AI 제어는 동시에 켜지 않는다. 주행 lease에는 거리·시간·속도·센서 신선도·AI 상태·engine/session 식별자 감시와 종료 후 정차 확인이 필요하다. 정차 시험의 성공을 근거로 바로 전체 랩을 요청하지 않는다.

AI의 실제 적용 조향·페달을 카메라/몸 감각과 시간 정렬해야 한다. 설치 SDK에서 `physics.getCarInputControls()`는 **AI 제어 중에도 사용자 입력을 반환**한다고 설명하므로 이를 전문가 출력 라벨로 사용하면 안 된다. 실제 `car.steer`는 스티어링휠 각도(deg)이고 gas/brake는 0..1이다. 실제 움직이는 상태에서 native/SDK 대조와 조향 역변환·지연을 검증한 뒤에만 실행된 전문가 라벨로 등록한다.

API 자동운전 활성화는 SDK 설명상 랩을 무효화한다. 따라서 AI 자료에는 `teacher_kind=engine_ai`와 lap invalidation을 보존하고, 학생 모델의 독립 유효 랩 성공과 구분해야 한다. 경로·차량 위치·AI 내부 정보는 교사/감시용이며 학생 actor 입력에는 추가하지 않는다. 겹눈 네 시점과 등록된 몸 감각의 원본·나이·유효성 마스크를 보존한다.

전체 코스의 자료 수집, 학생 모델의 연속 완주, 반복 성능, 타이어 온도·하중·셋업에 따른 주행 최적화는 아직 미완료다.
