# API 엔진 재시작과 진단 게시자 적용

기록일:2026-10-29,Asia/Seoul. [프로젝트 대전제](PROJECT_CHARTER.md)를 유지한다.
이 기록은 진단 게시자 수정을 실제 게임에 적용한 증거다. 운전 모델·생리 기능·안정화·완주 판정은 별도다.

## 확인한 결과

| 기록 | 결과 |
|---|---|
| [t1211 서비스 배포](../runs/ac-engine-restart-service-deploy-20261029-t1211/result.json) | 기존 Lua를 백업하고 재시작 API를 추가한 소스를 배포. 새 세대 heartbeat로 실제 재로딩 확인. 전면 활성화 없음 |
| [t1212 읽기 전용 점검](../runs/ac-engine-publisher-restart-preflight-20261029-t1212/result.json) | 기존 엔진PID46748 활성·정차·제동·pause·AIoff·cameraoff·vJoy 해제·설치 해시 확인. 출력/재시작 요청0 |
| [t1213 실제 적용](../runs/ac-engine-publisher-restart-20261029-t1213/result.json) | 게시자 백업·파일 교체 후 CSP 재시작 요청1회. 원래 프로세스 핸들에서PID46748 종료0, 새PID50656·새 게시자 세션·실제 패킷의3ms 설정 확인. UI 입력0·출력0·새 창 비전면 |
| [t1214 API start](../runs/ac-new-publisher-start-20261029-t1214/result.json) | 준비 메뉴에서 주행 세션 진입. 창 조작 없음 |
| [t1215 pause 거부](../runs/ac-new-publisher-pause-20261029-t1215/result.json) | start 뒤 native 브레이크0으로 초기화돼 기존 정차/제동 가드가 거부. 요청을 반복하거나 가드를 완화하지 않음 |
| [t1216 브레이크 복구](../runs/ac-new-publisher-brake-hold-20261029-t1216/result.json) | 이미 정차·가속0인 동일 엔진에서 중앙조향/가속0/제동1만7요청. 연속5 native 제동 표본·장치 해제·후속 응답 확인 |
| [t1217 pause](../runs/ac-new-publisher-pause-20261029-t1217/result.json) | 정상 일시정지 확인. 전면 활성화 없음 |
| [t1218 새 게시자50표본](../runs/ac-new-publisher-diagnostic-proof-20261029-t1218/result.json) | 읽기 전용 신선도 검사 통과, 게시 오류 증가0, 최대 원래 수신 나이0.0954ms. 실제 이동 중 지연의 보장은 아님 |
| [t1219 실제 비교 전 점검](../runs/flight-publisher-retry-preflight-20261029-t1219/result.json) | 새 게시자·원래 시각/비행 인증과 source 해시를 묶은 새 계획 통과 |

현재 게시자 적용 소스 SHA256은
`a6d9b99d13aafc0e9eec7e7ef36356936c55aaf72aa2420d5d720f54159e528a`다.
업로드 파일만 확인한 것이 아니라, 새로운 엔진 프로세스와 producer_session_id 및
`publisher_replace_retry_limit_seconds=.003`을 실제 wire bytes에서 대조했다.

## 재시작 계약

설치된 SDK `extension/internal/lua-sdk/ac_apps/lib.lua`의 `ac.restartAssettoCorsa(raceIni)`를
`restart-engine`이라는 제한된 CSP 명령으로 노출했다. 기존 `restart`는 세션 재시작으로 남는다.
새 명령은 고정 차량/트랙의 오프라인·정차·가속0·제동·pause·AIoff·카메라 종료 조건과
외부 소유자의 vJoy 해제 확인, 원래 PID, 만료 시각·세대를 요구한다. 임의 race.ini는 받지 않는다.

서버는 실행 의도를 먼저 기록하고 동일 요청ID를 다시 실행하지 않는다.
void API가 반환됐다는 사실을 성공으로 판정하지 않는다. 클라이언트가 같은 원래 프로세스 핸들의
종료와 새 PID/세대/게시자를 모두 확인해야 적용 완료다. 결과 미확정이면 동일 핸들/PID부터 조사한다.

- 서비스: `integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua`
- 배포·재시작 클라이언트: `scripts/restart_ac_engine_publisher.py`
- 원래 프로세스 수명 확인: `src/flybrain_drive/win_process_receipt.py`
- 새 엔진의 정차 브레이크 초기화: `scripts/hold_ac_stationary_brake.py`
- 원본 백업: `runs/ac-engine-restart-service-deploy-20261029-t1211/installed-before.lua`,
  `runs/ac-engine-publisher-restart-20261029-t1213/publisher-installed-before.py`

수명주기·실제 자식 프로세스 핸들·진단 파일·기존 카메라 관련189검사와 Ruff 통과.
재시작 자체를 검증한 실제 증거는 t1213이며 합성 테스트를 게임 재시작 증거로 바꾸어 부르지 않는다.

## 다음 실제 비교

계획은 `configs/flight-physical-yaw-publisher-retry-v2-20261029.json`이다.
t1213 적용 증거, t1218 진단 읽기, t1196/t1197의 원래 시각/비행 재생 인증을 고정했다.
새 계획의 이전 성공 시험은0으로 시작한다.12회/3블록·양방향 교란·257관측·50/80/150ms 제한과
원래 관측·시간·회로 상태는 유지한다. 실제 진행/결론은 [현재 문맥](CONTEXT.md)과 실행 원본을 따른다.

## 새 게시자의 실제 실행 결과

[t1220 실제 블록](../runs/flight-publisher-retry-block0-20261029-t1220/result.json)의 full/trial-00은
257관측을 완료했다. [t1221 독립 감사](../runs/flight-publisher-retry-full-audit-20261029-t1221/result.json)는
전체 원본 재생 오차 최대1.83e-7, native717기록·주행선OFF270프레임을 대조했다.
최고6.160km/h, 위치 변화9.404m, 펄스280.627ms, 최대 관측 간격53.036ms였다.
추론 시간 중앙값27.006ms/p95 32.849ms/최대43.573ms다. 페달은 시험용6km/h 제어기다.

같은 블록의 yaw0/trial-01은 정차7관측 뒤8번째 원본 간격81.820ms로80ms 제한을 넘어 실패했다.
처리는42.803→51.824→62.284→65.630→68.435→70.693ms로 늘었다.
진단 읽기·신선도 중단은 없었고 실제 이동 조향 출력은 적용하지 않았다.
제동·중립·pause·worker 종료를 확인했으며, 실패 관측을 삭제하거나 간격을 다시 매기지 않았다.

[t1222 원본 prefix 계측](../runs/flight-publisher-retry-zero-prefix-profile-20261029-t1222/result.json)은
8관측 모두를 재생해 앞선7추론 출력을 대조하고 마지막 같은 가드 실패를 재현했다.
계측 재생에서 주요 시각 GPU 구간은29.988→55.231ms로 늘었고 비행 구간은약6.4~8.2ms였다.
이는 원래 간격 증가에 따라 적분 단계가 더 많아지는 이력이다. 오프라인 실행으로도 같은 원본을
빠뜨리지 않고 처리했으며 실제 지연 합격을 주장하지 않는다. GPU/호스트 경합 원인은 아직 미확정이다.

[t1224 진단 wire 감사](../runs/ac-new-publisher-retry-observed-20261029-t1224/result.json)에서
full515패킷·yaw0 16패킷의 해시와 순번을 확인했다. full에서 게시자 교체 재시도 수가0→2로 늘었지만
게시 실패 수는0이었다. 실제 재시도 작동의 증거이며 모든 부하에서의 무중단 보장은 아니다.

[t1223 마지막 상태](../runs/flight-publisher-retry-final-status-20261029-t1223/result.json)는
paused/AIoff/cameraoff/주행선OFF/창 비전면이었다. 이 단계의 소유 실행·감사 핸들은 모두 종료했다.

다음은 원래 수치 적분을 유지하며 시각·비행 실행 비용과 자원 경합을 더 줄이는 것이다.
새 계산 경로를 넣으면 원본 전체 뉴런·결합 재생으로 먼저 검증하고 양 모드를 공정하게 다시 비교한다.
예정12회/3블록은 미완료이며, full1회 성공으로 실제 안정화·시각 경로 추종·학습한 가감속·완주를 판정하지 않는다.
