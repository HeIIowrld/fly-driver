# 네 화면 묶음 복사와 저장 직렬화 지연

2026-10-29, Asia/Seoul. [대전제](PROJECT_CHARTER.md)를 유지한다.
**GPU 복사 경로는 빨라졌지만, 30초 수집은 여전히 실패했다. 다음 프레임 읽기가 이전 프레임의 저장·전처리에 막히는 사례를 확인했다.**
파리 모델의 조향/레이싱 성과가 아니라 실시간 영상 수신 경로의 작업이다.

## 구현과 같은 영상의 픽셀 검증

새 `SharedTextureReader.read_group`은 네 source texture를 각각 AddRef로 유지하고,
네 개의 own staging texture 복사를 모두 enqueue한 뒤 한 번 Flush하고 Map READ로 읽는다.
캐시가 교체돼도 그룹 안의 source/staging이 먼저 해제되지 않게 한다.
실패 시 부분 그룹을 반환하거나 ACK하지 않는다. 수신 원본 texture에는 쓰지 않는다.
기존 sequential 경로는 남기고 `--batched-shared-copy`에서만 새 경로를 선택한다.

group polling deadline은40ms다. driver 호출 자체의 실행 시간을 강제 중단하는 실시간 상한으로
해석하지 않는다. actor150ms/대기110ms 및 teacher 속도/거리/정지 제한은 바꾸지 않았다.
관련58검사와 Ruff를 통과했고 실제 driver 비교도 했다.

t1484의 한 immutable paused 영상에서 warmup 한 쌍 뒤 네 쌍의 순서를 번갈아 비교했다.
PNG reference와 두 방법의 총40개 화면 비교가 모두 픽셀 차0이었다.
warmup을 제외한4회 그룹 시간 중앙값은 sequential5.1303ms, batch3.0382ms였다.
같은 정지 영상의 작은 비교이며 실제 주행의 tail latency 개선/모든 환경 성능 보장은 아니다.

## 첫 묶음 복사 주행과 전체 프레임 계측

t1485 3초 시험은52.11km/h/74관측/누락0, t1486 원시 재생 차0으로 끝났다.
t1488은276개 모든 capture 그룹의 copy journal을 확인했다.

t1487 30초 시험은61.17km/h까지 갔지만29.3878초에 camera stale로 실패했다.
717관측/대기 누락5이며 도구session61155 종료1이다. 원래30초 완료로 승격하지 않는다.
t1489에서870개 그룹 전체의 다음 수치를 확인했다.

| 단계 | 중앙값 | 최대 |
|---|---:|---:|
| 그룹 복사 시작→ACK 직전 | 1.963ms | 5.357ms |
| render group clock→ACK 직전 상한 | 8.433ms | 106.705ms |
| Map polling 구간 | 0.0058ms | 3.020ms |
| CPU packing/Unmap 구간 | 0.221ms | 0.903ms |

새 Map 단계만으로 전체 지연을 설명할 수 없다. 실패 근처 capture807→808 간격은236.230ms였고
game render counter는5개 진행했다. frame807은 actor가 사용하지 않았지만 전체 journal에 남아 있다.
t1490 원시 재생 완료도 원래 task 실패를 덮어쓰지 않는다.

## 종료 후 진단 저장과 지연 구간 분리

첫 버전은 ACK 뒤 매 프레임 copy journal JSON을 썼다. 후속 버전에서는 이를
request frame 수 이하의 메모리에 보관하고 capture loop가 끝난 뒤 한 파일로 저장한다.
진단 파일 I/O를 실시간 경로에 추가하지 않기 위한 변경이다. 실제 sensor 이미지 저장은
현재 버전에서도 actor publication 이전에 수행한다.

새 journal v2는 frame 경로 발견·metadata 처리·ACK 게시 완료·이미지 저장/전처리·event log flush·
actor publication 뒤의 실제 시각을 나눠 기록한다. 시각을 다시 매기거나 과거 run을 수정하지 않는다.
파서도 기존 per-frame journal과 새 v2를 별도로 지원한다.

t1491 준비 시험은59.078km/h/74관측/누락0이었고 t1492 원시 재생 차0,
t1493 timing 검사도 완료했다. 그러나 t1494 30초 시험은5관측/최고0.259km/h에서 조기에 실패했다.
같은 코드 반복 수집을 더 하지 않았다. t1495/t1496은 실패 원본의 timing/무결성 감사다.

t1495의95개 그룹에서 GPU 그룹 복사 최대7.122ms, metadata 처리 최대1.405ms,
ACK 파일 게시 최대1.823ms였지만, **저장·전처리 구간 최대141.067ms**였다.
frame87이 그 구간을 처리하는 동안 다음 frame88은 이미 렌더돼 있었다.
원래 causal clock 범위를 보수적으로 적용해도 frame88은 이전 처리 종료보다138.424ms 먼저 준비됐고,
host는 이전 저장·전처리가 끝난 뒤1.466ms가 지나서야 frame88 경로를 확인했다.
그 결과 frame88의 render→발견 상한은144.045ms였다.

이 한 실패에서는 직렬 worker가 다음 영상 읽기를 지연한 것이 구체적으로 확인된다.
141ms 안에서 개별 BMP 쓰기·hash·전처리·OS scheduling이 얼마나 차지했는지는 아직 분리하지 못했다.
모든 과거 실패가 같은 원인이라거나 파일시스템 한 요소만의 문제라고 단정하지 않는다.

## 다음 구현: 별도 실시간 stream 계약

단순 GPU 복사 최적화로 저장 직렬화는 사라지지 않았다. 다음 버전은 다음 원칙으로 구현한다.

1. 같은 원본 identity/geometry 검증, 네 RGB의 own immutable copy와 hash를 끝낸 뒤
   실제 메모리 영상 준비 시각에 수신을 기록한다. native body 시각도 원래대로 쓴다.
2. 이미지 저장은 bounded queue/별도 worker에서 수행한다. queue 초과·저장 실패는 명시적 실패이며
   수락한 프레임을 조용히 버리거나 오래된 영상을 새 영상처럼 게시하지 않는다.
3. 새 stream은 **publication 전에 파일 저장을 끝내는 기존 계약과 구분**해야 한다.
   `actor_published_only_after_persistence=true`를 거짓으로 유지하거나 옛 run 시각을 바꾸지 않는다.
4. 모든 저장과 hash 검증이 완료되기 전에는 학습 run으로 승인하지 않는다.
   별도 원시 재생/관측 시각/실패 복구 검증 전에는 학생 고속 제어에 연결하지 않는다.
5. source150ms/actor110ms 기준을 넓히는 방식으로 해결하지 않는다.

이 stream 분리는 아직 구현 완료가 아니다. 조향의 시각 일반화 실패도 그대로 남는다.
그 작업과 실제 교사 오차 상태/코너 자료를 함께 진행해, 초파리 시각/비행·몸 회로로
조향·가감속·기울기·하중·타이어/셋업과 빠른 유효 레이싱 라인/랩타임을 검증해야 한다.

## 보존과 실행 상태

소유한 모든 계측/감사 프로세스는 terminal이다. 실패한 주행 원본과 역할calibration/fit 금지를 보존했다.
원본은 로컬 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/`의 t1484~t1496 폴더다.
NAS compact 보존은 `runs/ac-batched-copy-serial-persistence-20261029-t1497`이며 대용량 원시 영상은
로컬에 유지하고 hash/크기/경로로 참조한다.
v1 worker 소스는 변경 내역에서 재구성한 bytes가 t1487 manifest SHA256과 정확히 일치함을 확인해 별도로 보존했다.
이는 설정 실행 소스의 복원이며 프로세스 메모리를 소급 검사했다는 주장은 아니다.
