# 진단 신선도 재실패와 게시자 파일 교체 충돌

기록일:2026-10-29,Asia/Seoul. [프로젝트 대전제](PROJECT_CHARTER.md)는 유지한다.
이 문서는 수정 미배포 시점의 기록이다. 이후 실제 적용은
[API 엔진 재시작·새 게시자·실제 비교 결과](ac-api-engine-restart-and-publisher-deployment-20261029.md)를 따른다.
현재 문제는 가상 몸/비행 회로의 실제 비교를 끝까지 수행하기 위한 진단 연결의 중단이다.
부품 수리나 full1회 성공으로 자율 주행·안정화·랩타임 목표를 완료 처리하지 않는다.

## 실제 결과

진단 읽기의 최대10ms 새 순번 대기를 t1207 원본 코드 해시로 고정한 별도 계획
`configs/flight-physical-yaw-graphed-fresh-diagnostic-v2-20261029.json`을 만들었다.
과거 full 성공 t1205를 이 계획의 성공 시험으로 가져오지 않았다.

- [t1208 읽기 전용 점검](../runs/flight-graphed-fresh-preflight-20261029-t1208/result.json)은 통과했다.
- [t1209 실제 블록](../runs/flight-graphed-fresh-block0-20261029-t1209/result.json)은 첫 full 조건의
  138관측 뒤 진단 기록 신선도 오류로 중단됐다. 새 순번 재시도4회, 읽기 충돌 재시도0회였다.
  신경망 조향은 적용했지만 교란 전 중단되어 유효한 full/yaw0 쌍은 없다.
  제동·중립·pause·worker 종료를 확인했고 실패 원본과 영상을 보존했다.
- 새 [실패 진단 기록](../runs/flight-graphed-fresh-block0-20261029-t1209/trial-00/failure-diagnostic.json)은
  마지막 원본 순번2315038과 최초 수신 시각·wire bytes·해시를 보존한다.
  예외 시각과의 차이는80.364ms였다. 오래된 기록을 새 시각으로 반환하지 않았다.

## 확인한 원인 후보

실패 직전 원본 패킷의 `publisher_errors`는279였고 이후 [t1210 읽기 전용 확인](../runs/flight-diagnostic-publisher-collision-evidence-20261029-t1210/result.json)에서는280이었다.
원래 게시자는 파일 교체 실패 시 다음 AC update까지 기다리므로 파일 읽기 충돌1회가
프레임 간격과 겹치면50ms 신선도 제한을 넘길 수 있다. 이 실행의 정확한 오류 호출 스택은
기록되지 않았으므로 해당 순간의 단일 원인으로 확정하지 않는다.

Windows의 파일 공유 옵션과 교체 동작은 [CreateFile 공식 문서](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea)와
[ReplaceFileW 공식 문서](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew)를 확인했다.
읽기 핸들에 FILE_SHARE_DELETE를 허용하는 시도도 실제 Windows에서 시험했으나,
현재 게시자가 쓰는 `os.replace`는 열린 대상 파일 교체에 WinError5를 반환했다.
그 시도는 검증 실패 후 철회했다. 동작하지 않는 우회 코드를 배포하지 않았다.

## 준비한 수정 — 아직 게임에 미적용

`integrations/ac/flybrain_telemetry/flybrain_telemetry.py`의 FilePublisher가 이미 직렬화한
동일 패킷의 파일 교체만 최대3ms 재시도하도록 수정했다. Windows 오류5/32/33만 해당한다.
센서 값을 다시 읽거나 순번·측정 시각을 바꾸지 않는다. 실패가 지속되면 기존처럼 오류를 반환한다.
누적 교체 재시도 수와3ms 제한을 다음 게시 패킷에 명시한다.

관련30검사와 Ruff를 통과했다. 실제 Windows 읽기 핸들이 첫 교체를 막고, 그 핸들을
닫은 뒤 재시도한 동일 bytes가 게시되는 회귀 검사도 포함한다.
이는 로컬 게시자 시험이며 실행 중인 게임의 수정 적용 증거는 아니다.

t1210에서 확인한 해시:

| 위치 | SHA256 |
|---|---|
| 준비된 프로젝트 소스 | `a6d9b99d13aafc0e9eec7e7ef36356936c55aaf72aa2420d5d720f54159e528a` |
| 현재 게임 설치 파일 | `7fcd217f85c15b86bf632c825090f31463e6294f8dfdcf41aa62bb5418188330` |

설치 파일과 실행 중인 게시자는 이전 버전이다. 따라서 t1208 계획을 그대로 반복해 새 수정의
효과를 주장하지 않는다. 이 단계에서 앱 파일 배포나 엔진 프로세스 재시작은 하지 않았다.

## 바로 다음 작업

1. 제동·정차·pause·AIoff·cameraoff·vJoy 해제와 설치 파일 해시를 다시 확인한 뒤,
   백업과 영수증을 남겨 게시자 수정 파일을 배포한다.
2. Python 앱을 실제로 다시 읽게 할 엔진 프로세스 재시작 경로를 준비·검증한다.
   기존 CSP의 `restart`는 세션 재시작이며 Python 코드 재적용 증거가 아니다.
   설치된 SDK의 `extension/internal/lua-sdk/ac_apps/lib.lua:14348`에는
   `ac.restartAssettoCorsa(raceIni)`가 명시돼 있다. 현재 FlyBrain API에는 노출하지 않았으므로
   제한된 명령·사전 조건·한 번만 요청·기존/새 PID 및 세대 확인·실패 보존을 구현해야 한다.
   창 클릭·키 입력으로 대신하지 않는다. API 결과 미확정은 동일 소유 핸들/PID부터 확인한다.
3. 새 게시자 세션과 `publisher_replace_retry_limit_seconds=.003`을 실제 패킷에서 확인하고
   새 읽기 전용 증거 및 transport 출처를 고정한다. 파일 배치만으로 적용 완료 처리하지 않는다.
4. 원래12회/3블록 full/yaw0 양방향 비교와 전체 독립 원본 재생을 이어간다.
   50ms 진단·80ms 명령·150ms 영상 제한, 원래 관측·시간·회로 상태는 유지한다.

최신 t1210의 게임은 paused/AIoff/cameraoff였다. 이 단계의 모든 소유 실행 핸들은 종료했다.
다른 작업은 종료하지 않았다. 전체 목표·운전 후보·실제 안정화 비교는 미완료다.
