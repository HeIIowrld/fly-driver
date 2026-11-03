# 실시간 공유메모리 연결과 새 코너 시각 자료

2026-10-29, Asia/Seoul. **초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는
가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며
조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.**

실시간 metadata/ACK 공유메모리 경로를 연결했다. 새 코너의 완전한 train 이력2개,
1,479관측을 확보하고 GPU의 고정된 초파리 시각망으로 전부 재생했다.
**수집 전체는 세 번째 train의 원본 영상 공백으로 실패했다. 새 조향 학습·학생 배치는 아직 없다.**

## 실시간 v3 전송과 보존

`--mmf-metadata`는 batched shared GPU copy·비동기 저장과 함께 명시적으로 선택한다.
기존 경로는 유지했다. 새 source 요청은 `mmf-stream-<GUID>`, 최대1200프레임,
`host_persistence_protocol=immutable_ram_v3`이며 정차/full brake/CSP AI-off에서만 시작한다.
새 flight shadow/학생 제어에는 아직 사용하지 않는다.

네 영상의 복사가 모두 끝난 뒤 소유한 ACK map을 한 번 공개한다.
실제 준비시각을 receipt로 삼고 다음7개 파일의 불변 바이트를 하나의 FIFO 작업으로 저장한다.

- BMP4개, 원래 수신한 frame JSON, 실제 ACK JSON, 실제 frame/ACK map 두 개를 이어 붙인 `.mmf`.
- map은 각각32848바이트이며 packet 재구성이 아니라 실제 읽은 메모리 snapshot을 보존한다.
- 최대 대기열8묶음/묶음1MiB. 대기열 초과·저장 오류는 실행 실패이며 silent drop은 없다.
- 종료 시 모든 수락 작업을 drain하고 파일 hash를 readback한다. 불완전한 최신 metadata/ACK/slot도 실패 원문으로 남긴다.

stream은 `flybrain.csp.immutable_rgb_stream.v3`, pipeline은
`copy_all_then_mmf_ack_then_async_persist_v3`, persistence journal은v2다.
독립 감사는 실제 slot의 header/sequence/SHA와 원본 JSON·receipt·GPU copy 순서를 함께 확인한다.
구형 stream 감사가 새v3를 조용히 받아들이지 못하도록 했다.
관련247검사와 Ruff를 통과했다. 기존150ms/110ms 신선도·교사 속도/거리/정지 제한은 유지한다.

t1520에서 Lua를 정차 상태로 한 번 원자 배치했다. 현재 generation은
`4525688ULL:0.0012764`, Lua SHA256은
`a9f583e54da981c5e700b186d51af09fdb9249aa6112cf89a87608cd05940386`다.
엔진 재시작·desktop 활성화·클릭·키 입력은0이다.

## 유한 주행 시험

| 실행 | 결과 | 관측 / 대기 누락 | 최고속도 | 독립 원시 감사 |
|---|---|---|---:|---|
| t1521,3초 | 완료 |74 / 0|58.9138km/h|t1522,전체 배열 차0/229묶음 원본 slot 검증|
| t1523,30초 | 완료 |739 / 0|60.9280km/h|t1524,전체 배열 차0/1008묶음 원본 slot 검증|

30초 시험의 최대 관측 간격43.1867ms/80ms초과0이다.
t1525에서1008묶음 모두 publication이 저장 완료보다 먼저였고 대기열 최대2였다.
receipt→publication 중앙값0.1263ms/최대0.8776ms, receipt→저장 완료 최대14.3008ms였다.
전체 환경의 안정성·최적 운전·학생 성과의 증거로 일반화하지 않는다. 역할은 calibration/fit 금지다.

## 새 지리 구간 수집과 실패

새 plan `flight-spatial-corner-mmf-corpus-20261029.json`은 예전 실패 batch를 재시작한 것이 아니다.
이전 역할은 유지하고 같은 지리 분리 기준(train0.27~0.63/development0.67~0.84/transit 나머지)을
새 소스와 이전 terminal 위치에 사전 고정했다. source snapshot은t1526이다.

t1526은 transit3개와 train2개 완료 뒤 episode05가 stale로 실패해 collector/자식이 종료1이다.
선언한 train3/development1을 완수하지 못했고 새 development는0개다.
완료 train2개를 모두 대상으로 t1527 partial diagnostic 원시 감사를 했다. 원래 batch 실패를 성공으로 바꾸지 않았다.

| 원래 이력 | 진행률 전체 범위 | 관측 | 최대 관측 간격 | CSP 보고 조향 범위 |
|---|---|---:|---:|---|
| episode03 train |0.276142~0.362774|739|43.9564ms|−203.279~+33.967|
| episode04 train |0.362765~0.449609|740|74.4165ms|−82.124~+67.919|

모든 이력·원래 시각을 유지했고 지리 구간 밖/바퀴 이탈은0이다. 픽셀·몸·관측 재생 차0,
모든 실제 MMF slot 원문 검증, 두 이력 모두80ms초과0을 확인했다.
조향 수치는 CSP 보고값이며 이 감사 자체가 조향 단위/학생 action codec을 인증하지 않는다.
두 이력의 시작3초 뒤 driving 중 brake>0.01은 모두0이다. 실제 코너 제동 학습 자료라고 부르지 않는다.

실패 episode05는504관측/대기누락1/최고61.163km/h/진행률0.503467에서 끝났다.
t1528 원시 무결성은 차0이나94.3672ms 관측 간격1회가 남는다. 학습에 넣지 않았다.
t1529는 원본 frame604→605의179.7476ms 공백/게임 render frame17개 진행을 확인했다.
다음 metadata 수신 지연 상한은3.597ms, 겹친 저장 작업은9.784ms였다.
MMF 뒤에도 원본 렌더링/update 공백이 남는 증거이며 전송만으로 해결됐다고 주장하지 않는다.
신선도 확대·시간 변경·실패 구간 삭제·같은 batch 자동 재시도는 하지 않았다.

## 실제 초파리 시각망 재생

t1530은 원본1479관측을 기존 가상 몸16개 값/마스크로 모두 재생했다.
기존에 등록하지 않은 가속/각속도 mask는 그대로 invalid이며 oracle로 채우지 않았다.
몸21·새 고속 actor goal 변환을 한 결과는 아니다.

t1531은 두 train 이력의 실제 RGB를 저자 Flyvis 시각망에 넣어 GPU에서 재생했다.
4개 카메라는 각각 독립 신경 상태를 가지며, 이력당1회 grey 초기화 후 원래 관측 시각·이전 영상 hold를 유지했다.
기존 검증된 FractionalVisualTail을 명시적으로 선택하는 flag를 추가했고 원래 fractional dt와 저자 식을 유지했다.

- episode03:739관측/6642내부step, 원래30.0981초를26.4281초에 재생.
- episode04:740관측/6655내부step, 원래30.1744초를26.2122초에 재생.
- GPU tensor peak5,515,289,088바이트, 저자 가중치 변경0, optimizer update0, 학생 조작0.
- 원래 원시 시각/retina/T4·T5/중앙 세포 반응을 별도 `responses.npz`로 보존했다.

이는 신경 입력 캐시 생성이며 조향 학습 성공이 아니다. t1473의 시각 기여 개발 실패는 아직 바뀌지 않았다.
다음은 두 새 train 이력과 원래 역할을 유지한 기존 자료에서 앞길 방향/곡률의 보조 표적을 만들고,
TRAIN만으로 학습한 시각 경계를 고정 운동 회로와 연결해 검증하는 것이다.
새 development가 확보되지 않았으므로 기존 알려진 development를 유지하며 blind 평가로 부르지 않는다.
새 고속 goal/action codec은 기존0..30 체크포인트와 별도이며 임의 재해석하지 않는다.

## 실행 상태와 보존

모든 소유 job은 terminal이다. collector handle16119 종료1, 원시 audit7141/99296 종료0,
GPU replay87571 종료0이다. 마지막 게임은paused/AI-off/camera-off/probe-off/full brake/약0.009475km/h였다.
native/CSP AI flag 불일치는 별도 미해결이며 학생 제어권을 우회하지 않는다.

로컬 원본은 `runtime-runs/20261029-t1520-*`부터`t1531-*`에 유지한다.
NAS 요약은 `runs/ac-live-mmf-spatial-data-20261029-t1532`이며 대용량 raw ZIP/신경 반응은
로컬 경로·크기·SHA256으로 참조한다. 전체 대용량 원시 자료를 NAS로 복사했다고 주장하지 않는다.
