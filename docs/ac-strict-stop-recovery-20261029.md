# 독립 정지 확인 복구와 새 교사 수집

2026-10-29, Asia/Seoul. [이전 v3 실패](flight-timing-coverage-and-v3-collection-20261029.md)를 보존한다.
자동차를 초파리의 가상 몸으로 삼아 시각·비행 균형·조향·가감속·하중/타이어·레이싱 라인/랩타임을
학습한다는 [대전제](PROJECT_CHARTER.md)를 유지한다. 이번 작업은 그 자료 수집 경로의 복구이며 학생 운전 성과가 아니다.

## 결과와 현재 경로

기존 native<0.1km/h·full brake·zero gas 기준을 유지했다.
teacher cleanup의 CSP pause 조건을<0.01km/h로 강화한 코드가 실제 엔진에 배치됐고,
새 엔진의3초 움직임 시험에서 native 정지 후0.009756527km/h·영상74관측 모두 유효·누락0을 확인했다.
t1410 recorder와 t1411 원시 재생 감사는 종료0이다. raw pixel/native byte 재생 차0과80ms 초과 간격0을 확인했다.
이3초 결과는 여러30초 lease 또는 모든 코스의 정지 성공을 보장하지 않는다.

새 [v4 교사 수집 계획](../configs/flight-aid-free-corpus-v4-20261029.json)은24개30초 lease의 역할을
수집 전에 고정했다. 이전 v3를 재개하거나 실패 개발 구간의 역할을 바꾸지 않았다.
640×360 창·비전면 실행이며 UI 활성화/클릭/키보드 입력은0이다.
진행 원본은 로컬 `runtime-runs/20261029-t1412-teacher-corpus-v4`다.
수집 완료·완주·학습을 주장하려면 이 실행의 실제 terminal 결과와 코스 범위를 별도로 확인해야 한다.
이번 마지막 실제 확인에서는 앞5개 lease가 종료0/총3,695관측/카메라 누락0이며,
해당 train/개발5구간의 원시·표적 감사도 완료했다. index5 평가 수집 중이었다. 다음 재개 시 현재 상태를 다시 확인한다.

## 실패한 짧은 복구도 보존

t1401은 이미 끝난 교사 실패와 동일한 native packet/prefix를 확인한 뒤 코드만 hot reload했다.
이 배치 당시 일반 native admission 실패를 성공으로 바꾸지 않았다.
t1402는 기존 full brake 상태에서 최대0.35초만 재개하는 별도 corrective lease를 한 번 보냈다.
AI·페달·조향 setter는0이며, 기존 교사 terminal ID·같은 창/epoch·독립 native 실패 기록이 있어야 admission된다.
속도0.2km/h·위치 이동0.02m·gas<0.02·brake>0.98·CSP AI-off 제한과 서버 자체 deadline을 둔다.
deadline은 게임 update에서 검사하는 소프트웨어 기한이며 운영체제 수준의 실시간 상한을 인증하지 않는다.

실제 재개22.807ms 뒤 speed0.516028km/h로 envelope를 벗어나 서버가 pause를 요청했다.
최대 위치 변화는0.000732974m였고, 이 결과는 `api_returned_false`·복구 실패로 남았다.
추가 반복 또는 제한 확대는 하지 않았다. native packet은222775→222778로 바뀌었지만
정지 속도 기준을 만족하지 못했다. 이후 heartbeat에서는 paused/full brake를 확인했다.
원인(물리 재개 시 상태 변화 등)은 확정하지 않았다. 단순 수직 흔들림으로 해석하지 않았다.

이 실험에서 `tryToPause(true)` 반환 직후 같은 update의 상태는 아직 paused=false일 수 있음을 확인했다.
서버는 이제 같은 pause 호출의 관측 결과를 기다린다. setter를 반복하지 않고,
추가1초 안에 paused가 관측되지 않으면 error로 남긴다. 부적격/과속/가속·브레이크 변화/이동 초과/
클라이언트 부재·앱 해제/비동기 pause/중복 요청을 포함해 Lua와 host 검사193개가 통과했다.

## 소유한 엔진만 새로 시작

t1403은 t1318 기록으로 직접 시작한 엔진28428임을 확인하고, 현재 paused·카메라/AI/lease 없음·
vJoy FREE·zero gas/full brake·불변 실패 packet을 확인한 뒤 그 프로세스만 종료했다.
원래 프로세스 handle의 실제 종료 신호를 기록했다. 게임 상태 초기화이며 정상 native guard를 완화한 resume가 아니다.
다른 프로세스·llama-server는 건드리지 않았다.

검증한 기존640×360/windowed 설정 그대로 한 번 cold start했다.
새 엔진36644/창4525688의 heartbeat를3초 이상 관측했고 foreground=false였다.
현재 실행 generation은 `4525688ULL:0.0017325`다. launch 요청도 SW_SHOWNOACTIVATE이며 실제 상태를 확인했다.
이것은 무화면 엔진 또는 완전한 headless 실행을 뜻하지 않는다.

t1404의 start API는 완료했지만 cockpit 진입 후 native brake가0이어서 t1405 pause가 거부됐다.
이어진 t1407도 paused 조건을 만족하지 못해 lifecycle 요청0으로 실패했다.
두 실패를 보존한다. 초기 두 명령을 한 shell에 묶어 앞 명령 실패 뒤에도 진단이 실행된 점은 잘못이었다.
이후부터 의존 단계는 실제 종료0을 확인한 뒤 각각 실행했다.

t1408은 독립 native AI-off·실제 정차를 확인한 뒤 vJoy로 centered/full brake만 설정하고 해제했다.
장치 요청7회이며 가속·조향 과제는 없었다. t1409 pause가 실제 완료된 다음에만 t1410 주행 시험을 시작했다.
새 엔진 초기 native AI-off는 확인했지만, teacher disable 뒤의 native AI 표시 불일치가 일반적으로 해결됐다는 의미는 아니다.
파리 actor의 소유권 인증은 별도로 유지한다.

## 증거와 남은 목표

로컬 원본은 `runtime-runs/20261029-t1401-*`부터`t1411-*`에 보존했다.
실행별 실제 소스·배치 전후 Lua·요청/수신·native bytes·실패·프로세스 종료 증거를 함께 유지한다.
오프라인 실행 source snapshot은`t1406`이며 snapshot과 실행 중 프로세스 메모리 검사는 구분한다.
초기 t1401의 import 경로 오류와 t1404의 CLI 문법 오류는 API 호출 전 실패였고 새 요청을 전송하지 않았다.

수집 자료를 역할대로 원시 감사·표적 봉인하고, 실제 제동 표적·코스 커버리지를 확인한 뒤
고정 저자 시각 네트워크와 빠른 몸 회로를 유지하며 감각 융합/경로 판단을 다시 검증한다.
가감속·하중 이동·타이어/셋업 적응·유효 완주·빠른 레이싱 라인·랩타임 목표는 미완료다.
