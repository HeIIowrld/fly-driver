# 구간별 코너 수집 준비와 원본 영상 공백

2026-10-29, Asia/Seoul. **초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는
가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며
조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.**

## 실제 결과

구간별 역할을 미리 정하는 수집기를 구현했고 관련 73검사/Ruff를 통과했다.
그러나 t1508의 첫 transit에서 원본 영상 공백으로 중단됐다. **새 train/development 코너
자료는 아직 0개다.** 직전 t1502의 30초 성공을 여러 구간의 안정성으로 일반화하지 않는다.
파리 모델의 t1473 시각 조향 개발 실패와 학생 미배치는 그대로다.

## 자료 분포와 새 수집 규칙

원래 t1393의 역할별 경로 감사에서 train/development가 관측한 1% 구간은 합계 37개였다.
새 위치 구간은 수집 전에 `configs/flight-spatial-corner-corpus-20261029.json`에 고정했다.

| 위치 진행률 | 역할 |
|---|---|
| 0.27~0.63 | train 후보, 완전 이력 3개 목표 |
| 0.67~0.84 | development 후보, 완전 이력 1개 목표 |
| 나머지·경계 접근·역할 충족 뒤 | transit, fit 금지 |

기존 development 0.20~0.25/0.94~1.00 및 기존 평가 위치 0.08~0.125를 새 train에 넣지 않는다.
평가 배열·평가 receipt는 열지 않았다. 새 development는 이미 진단한 트랙의 알려진 검증용이며 blind 평가가 아니다.
진행률과 위치는 수집 배정/감사 전용이다. actor에 전달하지 않는다.

각 lease는 시작 전에 역할을 기록하고, 이전 terminal receipt와 새 실제 시작 위치가 맞는지 확인한다.
완전 이력의 모든 telemetry/cleanup 위치가 선언 구간 안에 있고 바퀴 이탈이 없어야 후보가 된다.
구간을 넘거나 역행하면 전체 이력을 거부하며 좋은 부분만 자르지 않는다. 후보도 원시/정답 검증 전 fit 금지다.

다음 lease의 계획 전진량 0.13은 배정 휴리스틱이다. 720m 변위 제한이 곡선의 호 길이를
제한한다고 주장하지 않는다. 실제 전 이력의 지리 범위 확인이 최종 배정 검증이다.
총 최대 18개 유한 lease, 목표는60km/h, 이동은3초 또는30초이며 기존80km/h/정지/영상 신선도 제한을 유지한다.
현재 위치에서 이어가고 재시작·teleport·UI 조작은 하지 않았다.

## 첫 이동 구간 실패와 독립 감사

t1508 collector session32140 및 recorder/control 자식은 모두 종료1이다. 자동 재시도하지 않았다.
시작 진행률0.903948→종료0.924132, 최고60.8644km/h, 최대 변위113.129m였다.
238관측/카메라 대기 누락12회이고 stop reason은 `owned camera ended or became stale`다.
원래 transit/fit 금지 역할을 유지한다.

- t1509 원시 재생 차0, 370묶음 저장 완료/해시 검증. 이는 수집 성공이 아니다.
- 관측 최대간격565.9724ms, 80ms초과1회라 연속 학습 자료로 승인하지 않는다.
- t1510: GPU 복사~ACK 최대5.6477ms, render→ACK 상한 최대43.1093ms였다.
- t1511: 370묶음 모두 publication이 저장 완료보다 앞섰다. 최대 대기열1,
  receipt→publication 최대2.8410ms. 파일 저장/readback 최대595.8284ms가 남았다.
- t1512: frame309→310 원본 시각3405.4684088→3406.103761, 간격635.3522ms.
  render frame7331054→7331061로7개 진행했다. 다음 metadata를 읽기까지의 원본 시각 기준 지연은
  약7.479ms다. 이전 프레임을 메모리에 전달한 뒤 다음 metadata 자체를 기다렸다.

호스트 파일 저장과 원본 렌더링의 지연이 겹쳤다는 증거다. GPU 부하·스토리지·스케줄러 등
단일 원인을 확정하지 않는다. 게임이 완전히 멈췄다거나 Python 저장 직렬화가 계속 원인이라고 단정하지 않는다.
신선도 제한 확대, 공백 제거, 타임스탬프 변경, 재귀 상태 초기화는 하지 않는다.

## 다음 구현 근거

설치된 CSP 기본 SDK의 `ac_apps/lib.lua`에서 `ac.readMemoryMappedFile`,
`ac.writeMemoryMappedFile`, `ac.disposeMemoryMappedFile`의 공식 로컬 선언을 확인했다.
쓰기용 map은 기존 map이면 내용을 유지하며, 읽기용 map의 내용 수정은 금지되어 있다.
다음은 별도 이름/소유권의 metadata·ACK map으로 매 프레임 파일 hot path를 제거하는 opt-in 경로다.
request/generation/캡처 번호·4개 영상 소유권·실제 receipt·종료 후 원본 저장을 보존해야 한다.
현재는 API 근거 확인 단계이며 이 map 경로는 아직 구현/배포하지 않았다.

설치된 원본 launcher `themes/default/js/ui.js:2981`에서는
`FPS_CAP_MS = (1000 / fps).toFixed(6)` 변환도 확인했다. 현 설정은640×360/windowed,
VSYNC0/FPS_CAP_MS0이다. 프레임 제한은 아직 바꾸지 않았고 재시작도 하지 않았다.
120FPS 제한은 가능한 별도 비교 조건이며 이번 중단의 해결책으로 입증하지 않았다.

## 보존과 재개

모든 소유 job은 terminal이다. 마지막 heartbeat는 paused/AI-off/camera-off/probe-off,
gas0/brake1/약0.009925km/h다. native/CSP AI flag 불일치는 해결하지 않았고 학생 제어권을 우회하지 않는다.

- 실행 소스: `C:/Users/jeff4/AppData/Local/flybrain/execution-snapshots/t1507-spatial-teacher`.
- 실패 원본: `runtime-runs/20261029-t1508-spatial-corner-corpus/episode-00-transit`.
- 독립 감사: 같은 local runtime-runs의 t1509~t1512.
- NAS 요약: `runs/flight-spatial-source-interruption-20261029-t1513`.
  대용량 원시 ZIP/이벤트는 로컬 유지, 경로·크기·해시를 참조한다.
  새 구현 전 같은 plan/실패 batch를 다시 시작하지 않는다.
