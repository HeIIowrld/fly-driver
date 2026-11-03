# 게임 내부 센서 저장 지연 분리와 코너 자료 수집

갱신: 2026-10-29, Asia/Seoul. 앞선 기록은
[120FPS와 지연 계측](flight-framecap120-and-publisher-timing-20261029.md)에 보존한다.
원본 실행은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/`다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다.
차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 실측한 지연

내부 Python 센서가 SDK를 읽은 뒤 같은 게임 콜백에서 파일을 쓰고 있었다.
t1598은 계측 소스 설치와 단일 재시작까지 진행했지만 마지막 증거 저장에서
mappingproxy JSON 직렬화 오류로 종료1이다. 기존48364의 종료0과 새23056이 관측됐다.
재시작을 재전송하지 않고 t1599에서 신선한 새 producer/engine/nativeAI-off를 읽기 전용으로 확인했다.
증거 저장은 원본 wire bytes+SHA를 기록하는 기존 record_packet으로 수정했다.
원래 실패 결과를 성공으로 덮어쓰지 않았다.

t1604 관찰은2971개 원본 packet wire와 SHA를 보존했다. t1610에서 전부 재검증했다.
동기 open-write-close 최대85.3696ms/publish85.8162ms가 확인됐다.
단방향 도착 시각 차의 최솟값으로 추정한 해당 호출 구간은 host274761.6063~274761.6916이며,
같은 시간대의 영상491에서94.6035ms 공백이 나타났다. 시계 대응은 추정이며 노출 동기화나
단독 원인 증명이 아니다. 콜백 안의 파일 호출이 긴 시간을 차지한다는 근거로 사용한다.

## 실제 수정

`LatestFilePublisher`는 SDK 읽기와 측정 시각/순서 부여를 원래 콜백에 두고,
측정값을 독립 복사해 하나의 대기 슬롯에 넣는다. 별도 작업자가 JSON화/파일 쓰기/교체를 한다.
디스크 호출 동안 콜백 잠금을 잡지 않는다. 대기 슬롯을 덮으면 생략 횟수를 기록하고,
worker 오류도 후속 packet에 남긴다. 쓰기가 늦어져도 측정 시각을 다시 매기지 않는다.
역압 때문에 과거 측정값을 무제한 쌓지 않는다. 원래 파일 교체 재시도3ms 예산을 유지한다.
단일 OS 호출 자체의 소요 시간을3ms로 제한할 수 있다는 뜻은 아니다.

내장 threading이 없는 환경에서는 기존 동기 FilePublisher로 명시적으로 fallback한다.
이번 설치는 실제 packet의 `publisher_delivery=bounded_latest_worker_v1`로 worker 사용을 확인했다.
관측 누락/순서 생략은 감추지 않으며 이 파일을 연속50Hz 물리 관측으로 취급하지 않는다.
현재 actor 수집은 이 파일 대신 native physics와 camera MMF를 사용한다.

디스크 작업자를 막은 상태에서도 콜백 반환, 대기 슬롯1개, 깊은 복사, 순서1→3 생략과
원래 timestamp 보존, 종료 뒤 worker 종료를 검사했다. 관련 file/async/SDK/recovery42검사 통과.
별도 TCP7검사는 재실행 통과했다. 앞선 통합 실행은 TCP reset fixture에서 WinError10054로
1실패/47통과였다. 이를 없던 실패로 바꾸지 않는다. Python3.3 문법 검사도 통과했다.

t1611에서 publisher를 백업하고 새 소스 SHA
`b7c6f3ee24dc251fdfea4895dea593ab4cb15f56b2b4d4ac43df1d41bffcaa7a`를 설치했다.
기존23056의 종료0, 새43716/generation `602292ULL:0.0003732`를 확인했다.
FPS120 파일은 동일해서 교체0, UI조작0, foreground-after=false, 운전 소유권 부여0이다.
별도 recovery-only 정차/독립 native graphics pause 검증을 사용했고 학생 native AI-off 가드는 유지했다.

## 새 조건의 검증

| 시험 | 관측 | 대기누락 | 최대속도 | 원시 actor 최대 간격 |
| --- | ---: | ---: | ---: | ---: |
| 동기 계측 t1605 3초 |75|0|56.6800km/h|41.7693ms|
| 동기 계측 t1607 30초 |740|0|61.0957km/h|43.2701ms|
| 별도 worker t1616 3초 |75|0|56.5239km/h|41.2413ms|
| 별도 worker t1619 30초 |740|0|61.0944km/h|43.6435ms|

t1615 정차8묶음 검증 완료. t1620의 원시 actor 배열 차0/80ms 초과0/baseline advance 확인이다.
t1621/1622는 몸16→21 전체740관측에서 기존16을 정확히 보존했다. 무효 채널은 무효로 남겼다.
t1623의 영상 원본 최대 간격은50.6693ms,150ms 초과0이다.
t1618은75초 동안2289 packet과16 read 오류를 보존했다. 전체 실행 이후의 latched 최대는
worker open-write-close72.2448ms/replace110.488ms, 콜백 publish0.336ms다.
단일30초 성공으로 장기 수집이나 모든 엔진 공백의 해결을 주장하지 않는다.

t1624는740관측을 실제 저자 시각망 네 독립 상태로28.2813초 GPU 재생했다.
원본 시간30.0600초,6651 내부tick,peak5,515,289,088B, 저자 가중치 변경0이다.
실제 L1/L2/Mi1/Tm3/Tm1/Tm9 출력도 보존했다. 운동 회로 통합/학생 운전 검증은 아니다.

## 코너 자료와 남은 목표

계획 `configs/flight-spatial-publisher-async-corpus-20261029.json`은 같은 사전 지리 역할
TRAIN .27~.63/개발 .67~.84, TRAIN3/개발1, 최대18 lease를 유지한다.
영상/물리/배포 소스/설정 SHA를 고정했고 이전 t1593 실패는 그대로 보존했다.
완료 이력마다 원시 감사와 기존 시간 기준을 통과해야 세며 실패 tail을 잘라 fit하지 않는다.
이 새 조건에서 t1625 수집을 시작했다. 종료 여부와 완료 수는 아래 최신 실행 상태를 따른다.

이번 optimizer update0/학생 배포0이다. 앞으로 새 코너 자료의 시각→조향·가감속 학습,
비행 유래 운동 회로의 인과적 통합, 실제 제동/하중/몸 기울기/타이어·셋업 적응,
60→100/다속도, 학생 유효 완주, 빠른 레이싱 라인·랩타임을 검증해야 한다.

## 최신 실행 상태

21:16 KST 실제 프로세스 확인: t1625 collector wrapper29472/base48712, tool session34658가 실행 중이다.
첫transit은737관측/대기누락0/최대60.8878km/h로 완료했고 독립 원시 감사 디렉터리도 생성됐다.
현재 두 번째 transit wrapper9420/base9848가 실제 실행 중이며 완료 TRAIN0/개발0이다.
진행 파일만으로 종료를 판단하지 말고 같은 tool handle/PID/생성시각과 결과를 다시 확인한다.
동일 수집을 중복 실행하거나 활성 소스/계획/배포본을 변경하지 않는다.

최신 추가 확인 2026-10-29T21:18:53 KST: episode-03-transit / child wrapper49540 실행 중이다.
완료 이력의 독립 감사: [{"audit": "episode-00-transit-raw-audit", "status": "completed", "decisions": 737, "maximum_interval_seconds": 0.04296210000757128, "intervals_above80ms": 0}, {"audit": "episode-01-transit-raw-audit", "status": "completed", "decisions": 739, "maximum_interval_seconds": 0.043315500020980835, "intervals_above80ms": 0}, {"audit": "episode-02-transit-raw-audit", "status": "completed", "decisions": 739, "maximum_interval_seconds": 0.043248599977232516, "intervals_above80ms": 0}]
완료 TRAIN/개발: {"train": 0, "development": 0}; collector base48712/tool34658를 계속 관찰한다.
