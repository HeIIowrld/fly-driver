# 화면 조작 없는 AC 세션 제어

2026-10-28 KST. 사용자가 화면 자동 조작의 불편함을 지적해 이후 AC 창 활성화·클릭·키 입력을
중단했다. 학습과 에피소드 초기화도 API 경로를 우선하고, 실패 시 화면 조작으로 자동 우회하지
않는 지침을 저장소 AGENTS.md에 기록했다.

## 확인한 API와 실제 결과

설치된 CSP0.2.11의 `extension/internal/lua-sdk/ac_apps/lib.lua`에서 다음 API를 확인했다.
[CSP 개발자 SDK 저장소](https://github.com/ac-custom-shaders-patch/acc-lua-sdk)는 완전한 API
정의가 CSP 패키지 내부에 포함됨을 명시한다. 현재 게임에서 아래 호출을 실제 검증했다.

| 동작 | 호출 | 증거 |
|---|---|---|
| 현재 상태 읽기 | `ac.getSim()`, `ac.getCar(0)` | t218 |
| 오프라인 세션 재시작 | `ac.tryToRestartSession()` | t219, epoch0→1 및 준비 메뉴 확인 |
| 주행 화면 진입 | `ac.tryToStart(true)` | t220, 메뉴 해제 및 live 확인 |
| 일시정지 | `ac.tryToPause(true)` | t221, paused 확인 |
| 재개 | `ac.tryToPause(false)` | t222, paused 해제 확인 |

네 변경 호출은 각1회 요청했고 UI 입력과 조향·페달 메시지는0개였다. t219/t220 초기 결과의
`foreground_before`는 실제로 명령 뒤 측정값이었다. 두 값은 false였으며 이후 코드에서 진짜
before와 after 측정을 분리했다. 과거 결과를 수정하지 않는다. 실제 창 활성화 API를 호출한
것은 아니다. 재시작 직후 차량 초기화 과정에서 순간적인 속도 값이 나올 수 있으므로 다음
동작 전에 별도의 native 정차·제동 검사를 통과해야 한다.

재시작 API는 이 버전에서 Lua `function`이 아닌 FFI `cdata`로 제공된다. 함수 존재 확인은
`type(...) == 'function'` 대신 nil 여부와 실제 호출 결과를 사용한다.

## 구현과 실행

`integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua`에 카메라 요청과 구분된 세션 수명주기
처리부를 추가했다. 따라서 현재 앱 전체를 읽기 전용이라고 부르지 않는다. 카메라 수집 자체는
읽기 전용이고 세션 변경은 별도의 명시적 요청으로만 수행한다. 학생 신경망 출력에서 세션
명령을 만들지 않는다. 월드 위치·맵이 actor 입력에 추가되지도 않는다.

서비스 파일은 `%LOCALAPPDATA%/flybrain/csp-session/`에 있다. 기본 CLI는 읽기 전용이다.

```powershell
$env:PYTHONPATH='src'
python -m scripts.control_ac_session --command status --output runs/UNIQUE-status
python -m scripts.control_ac_session --command restart --execute --output runs/UNIQUE-restart
python -m scripts.control_ac_session --command start --execute --output runs/UNIQUE-start
python -m scripts.control_ac_session --command pause --execute --output runs/UNIQUE-pause
python -m scripts.control_ac_session --command resume --execute --output runs/UNIQUE-resume
```

각 요청은 고유 ID, 앱 generation, 세션 epoch, 짧은 만료 시각을 포함한다. 디스크에 수락
기록을 먼저 쓴 뒤 API를 한 번 호출하며, 결과 파일이 있는 ID는 앱 재로드 후에도 재실행하지
않는다. 응답이나 목표 상태를 확인하지 못하면 결과를 알 수 없다고 종료하고 같은 동작을
자동 재시도하지 않는다. 클라이언트는 vJoy가 해제됐고 카메라 수집이 끝났는지 확인한다.
Lua와 native 계측 양쪽에서 정차·제동·오프라인·차량/트랙을 확인한 뒤 변경한다.

직접 위치/물리 상태를 바꾸는 physics API, 조향·페달 CSP 대체 경로, 임의 코드 실행 요청은
이 서비스에 추가하지 않았다. 기존 조향·페달은 검증한 vJoy 경로를 계속 사용한다.

지원 API가 존재한다는 것과 게임이 렌더 없이 headless로 학습되거나 실시간보다 빠르게
진행된다는 것은 다르다. 현재 검증 범위는 실행 중인 원본 AC와 CSP의 백그라운드 세션 제어다.
관련10개 테스트와 Ruff 검사를 통과했다. 서킷 완주·타이어/셋업 적응은 여전히 전체 목표다.

## 이번 교사 수집의 별도 상태

t209는 학생 시험 끝 위치에서41개 요청 뒤 측면 여유1m 제한으로 중단했다. t212는 원래
출발점에서361개 요청 뒤40ms 결정 마감 초과로 중단했다. 마지막 완전한 피드백은359번으로,
약19.4km/h와 gap약-0.01m였다. 두 run 모두 카메라 종료와 자료 해시 보존을 확인했으나
완료된 연속 교사 자료로 봉인하지 않았다. 세션 API 도입 후 다음 수집은 화면 조작 없이
시작점 복귀→정차 확인→조향 재보정→기어 연결 순서로 준비한다.

기하 교사의 영상·몸 감각·행동 재생을 봉인하는 코드 분기는 추가했지만 실제 완료 run에 대한
통합 검증은 아직 남아 있다. 과거 SAC 봉인 자료의 성공을 이 새 분기의 검증으로 대신하지 않는다.
