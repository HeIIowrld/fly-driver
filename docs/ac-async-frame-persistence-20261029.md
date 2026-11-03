# 메모리 영상 수신과 파일 저장 분리

2026-10-29, Asia/Seoul. [프로젝트 대전제](PROJECT_CHARTER.md)를 유지한다.
초파리 겹눈·운동 시각과 비행 균형·운동 회로를 자동차라는 가상 몸에 연결하고,
차체 기울기·회전·하중·타이어 감각으로 조향·가감속·빠른 라인과 랩타임을 학습한다.

**새 비동기 저장 경로로 60km/h 교사 30초 수집을 완료했다. 파리 시각 조향은 여전히 개발 검증 실패·실제 제어 미적용이다.**

## 구현과 정보 경계

기존 저장 직렬화 실패(t1494/t1495)는 보존했다. 새 경로는
`--async-frame-persistence --batched-shared-copy --shared-transport --image-format bmp`에서만 선택한다.
기본 경로는 그대로다. 게임 창 활성화·클릭·키보드 입력은 하지 않았다.

네 GPU 영상 모두를 소유한 RGB 메모리로 복사하고 ACK한 뒤, 형상·노출·투영을 검증한다.
BMP 바이트와 RGB 해시를 메모리에서 만들고 실제 준비시각을 receipt로 기록한다.
별도 FIFO 스레드가 네 BMP와 stream JSONL을 저장하고 파일 해시를 다시 읽어 확인한다.
최대 대기열은 8묶음이며 다음 GPU 복사/ACK 전에 용량을 확인한다.
대기열 초과·쓰기 실패는 실행 실패다. 프레임을 버리거나 시각을 소급하지 않는다.

새 프로토콜은 `flybrain.csp.immutable_rgb_stream.v2`, pipeline은
`copy_all_then_ack_then_async_persist_v2`다. 기존 receipt 의미를 조용히 바꾸지 않는다.
`actor_published_only_after_persistence=false`를 명시하고, 전체 저장 완료와 독립 재생 감사 전에는
자료를 학습용으로 승인하지 않는다. writer 종료·모든 수락 작업·파일·stream 해시를 감사한다.
신선도 150ms/actor 대기 110ms, 교사 속도·거리·정지 한계는 유지했다.
새 경로의 flight shadow/학생 제어는 허용하지 않았다.

## 검증

관련 66개 검사와 Ruff 통과. 막힌 저장 작업이 수신 제출을 막지 않는지,
대기열이 찼을 때 실패하는지, 기존 파일을 덮어쓰지 않는지, 가변 바이트를 거부하는지,
미종료 journal·시각 역전·파일 손상이 독립 감사에서 거부되는지를 확인했다.

| 시험 | 결과 | 관측/대기 누락 | 최고 속도 | 원시 재생 |
|---|---|---|---:|---|
| t1499, 3초 준비 | 완료 | 74 / 0 | 59.0506km/h | t1500, 모든 배열 차 0 |
| t1502, 30초 교사 | 완료 | 739 / 0 | 61.1368km/h | t1503, 모든 배열 차 0 |

t1499는 218묶음, t1502는 1,000묶음의 네 카메라 파일·해시·저장 종료를 확인했다.
30초 시험의 최대 관측 간격은 48.5582ms, 80ms 초과는 0이다.
이는 한 번의 유한 시험 결과이며 전체 환경에서의 안정성·유효 랩·학생 성능 증거가 아니다.

t1505는 원시 단위 이름을 명확히 한 시간 감사다. 기존 t1504의 `degrees` 필드명은
이 감사 자체로 단위를 검증하지 않았으므로 사용하지 않는다. 두 실행과 실행 소스 모두 보존한다.

- 1,000묶음 모두 실제 publication이 파일 저장 완료보다 빨랐다. 최대 저장 대기열은 1이었다.
- 메모리 준비 중앙값 1.9034ms, 최대 36.6009ms.
- receipt부터 실제 publication까지 중앙값 1.8958ms, 최대 3.4020ms.
- receipt부터 저장·readback 완료까지 중앙값 7.7615ms, 최대 16.7297ms.
- native `steerAngle` 원시 범위 −0.808235~+0.122297. 새 조향 단위/액션 codec 검증으로 해석하지 않는다.
  조향 변화가 작은 이번 구간을 다양한 코너·복구 상태 자료라고 부르지 않는다.

## 남은 조향 과제

시각→파리 운동 회로의 t1473 개발 실패는 바뀌지 않았다. 평균 시각 치환이 실제 영상보다 나았고,
새 학생을 차량에 연결하지 않았다. 현재 작업은 그 문제를 해결할 자료 수집 경로의 개선이다.

다음은 원래 train/dev 역할을 유지하면서 좌우 코너·횡오차·다속도·실제 제동 과제의
경로별 역할을 미리 정하고 확보하는 것이다. 직선 반복을 새로운 경로 다양성으로 세지 않는다.
배경 교사 실행과 파리 모델 조향을 구분하고, 영상/몸 감각 외 지도·미래 정답 라인을 actor에 넣지 않는다.
현재 calibration 자료는 fit 금지이며 action codec·정답 라벨·학생 제어권 불일치도 별도 미해결이다.

## 실행과 보존

모든 새 수집·감사 프로세스는 종료0이다. 종료 후 heartbeat는 paused/AI-off/camera-off/probe-off,
gas0/brake1, 약0.00968km/h였다. 새 실행 전 fresh 상태를 다시 확인한다.

- 원본: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1499-*`부터 `t1505-*`.
- 새 실행 소스 사본: `C:/Users/jeff4/AppData/Local/flybrain/execution-snapshots/t1499-async-persistence-preflight`.
- 수정 전 worker 사본: `execution-snapshots/t1498-before-async-persistence`, SHA256
  `94a7ff251508a41220f8c2399f9ed6b8323aebcd4e81158591759a9a44b4dea7`.
- 사전 선언: `configs/flight-async-persistence-preflight-20261029.json`와
  `configs/flight-async-persistence-speed60-pilot-20261029.json`.
- NAS 요약: `runs/ac-async-frame-persistence-20261029-t1506`.
  대용량 원시 ZIP은 로컬에 유지하고 경로·크기·SHA256을 기록한다. 전체 원시 파일을 NAS로 복사했다고 주장하지 않는다.
