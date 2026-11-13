# Mugello 독립 코너 판독 평가 · 2026-11-11

BMW Z4 GT3의 오프라인 핫랩에서 주행선 표시를 끄고 Assetto Corsa 내장 AI로 주행해 네 방향 영상 1,000묶음을 기록했다. 녹화 전에 판독 가중치, 경계·경보 규칙, 채점 기준을 해시로 고정했다. 녹화 파일 해시를 확인했고 종료 후 AI 운전을 끄고 일시정지했다. [고정 설정](../runs/vision-corner-recovered-runner-20261111-01/mugello-protocol.json) · [녹화 결과](../runs/vision-mugello-independent-capture-20261111-01/result.json) · [종료 상태](../runs/vision-mugello-independent-capture-20261111-01/after.json)

시작 4–20m 전의 이동 표본은 좌 24프레임(물리 코너 ID 7개), 우 40프레임(ID 14개)이다. 직선 241, 범위 밖 168, 코너 내부 447, 정지 80프레임을 포함해 1,000프레임을 같은 채점기에 넣었다. 스플라인·트랙 위치는 정답과 평가에 사용했다. [입고 프레임](../runs/vision-mugello-independent-evaluation-20261111-01/base/admission-rows.jsonl)

## 판독 방식

기준은 세 시점의 전·좌·우·후 영상에서 읽은 T4/T5 운동 시각 특징을 사용한다. 새 후보는 상대 색 대비로 경계를 검출한다. 2·4·6·8m에서는 투영이 유효한 화면을 쓰고, 8–24m의 원거리 검출에는 전방 화면을 쓴다. 기존 경보가 살아 있으면 경계 방향을 대조하고 영상 거리 추정치를 섞는다. 기존 경보가 사라진 경우 좌우 반전 차이로 만든 방향 점수가 원래 방향을 지지할 때 기존 경보와 거리를 복구한다. [실행기](../scripts/evaluate_recovered_corner.py) · [경계 검출기](../scripts/relative_corner_boundaries.py)

새 복구 방향 점수의 정규화 특징 계수 제곱합 비중은 전 45.2%, 좌 15.3%, 우 13.5%, 후 26.1%였다. 새 거리 성분은 구역 평균 제거를 반영해 전 18.7%, 좌 25.2%, 우 32.6%, 후 23.5%였다. 원본 분기의 계수 크기이며, 실제 점수에는 좌우 반전 분기와 각 프레임의 특징값이 함께 들어간다. 기존 판독의 분기 합산 기여와 프레임별 무력화 검사는 별도 기록에 있다. [기존 가중치와 외부 구현](corner-camera-and-external-code-20261111.md)

## 개발 영상 재생

| 서킷 | 좌 적중 | 우 적중 | 방향·거리 ±5m | 범위 밖 경보 | 전체 기준 |
| --- | ---: | ---: | ---: | ---: | --- |
| Magione | 3→13 | 62→62 | 45→71 | 27→27 | 충족 |
| Laguna Seca | 4→4 | 0→0 | 4→4 | 12→12 | 미충족 |
| Zandvoort | 1→3 | 8→10 | 6→8 | 89→61 | 안정 예고 미충족 |

세 결과는 가중치 선택에 사용한 개발 영상의 재생이다. Magione의 분모는 설치 스플라인으로 다시 채점한 좌 64, 우 79프레임이다. [Magione](../runs/vision-corner-recovered-runner-20261111-01/magione/result.json) · [Laguna Seca](../runs/vision-corner-recovered-runner-20261111-01/ks_laguna_seca/result.json) · [Zandvoort](../runs/vision-corner-recovered-runner-20261111-01/ks_zandvoort/result.json)

## 독립 평가 결과

| 지표 | 기존 고정 판독 | 새 후보 |
| --- | ---: | ---: |
| 좌회전 방향 적중 | 4/24 | 4/24 |
| 우회전 방향 적중 | 1/40 | 1/40 |
| 방향·거리 ±5m 동시 적중 | 3/64 | 2/64 |
| 직선 오경보 | 0/241 | 0/241 |
| 범위 밖 오경보 | 12/168 | 5/168 |
| 보류 / 미출력 | 10 / 0 | 5 / 0 |
| 공통 연속 3쌍의 거리 변화 평균 절대 오차 | 4.94m | 3.97m |
| 접근 중 예고 거리가 거꾸로 증가한 공통 쌍 | 2/3 | 2/3 |
| 안정 예고가 나온 물리 코너 | 1/21 | 1/21 |

새 후보는 사전 기준을 통과하지 못했다. 놓친 접근 59개는 모두 직선 출력이었다. 경보 복구 규칙이 작동한 프레임은 0개였다. 기준과 후보 모두 같은 좌 코너에서 시작 10.13m 전에 최초 안정 예고를 냈다. 우측 코너 ID 14·17에는 4–20m 접근 표본이 각각 한 프레임뿐이어서 두 프레임 연속 적중 조건을 검증할 수 없었다. 연속 채점이 가능했던 19개 물리 코너에서도 안정 예고는 한 곳뿐이었다. [공식 결과](../runs/vision-mugello-independent-evaluation-20261111-01/result.json) · [후보 프레임 채점](../runs/vision-mugello-independent-evaluation-20261111-01/score/rows.jsonl) · [기준 프레임 채점](../runs/vision-mugello-independent-evaluation-20261111-01/base/parent/score/rows.jsonl)

다음 비교는 실제 접근을 직선으로 출력한 판독 단계를 영상의 경계·운동 증거와 함께 확인한다. 독립 녹화 입고 검사에는 4–20m에서 연속 채점이 가능한 좌우 물리 코너 수를 포함한다. 이번 고정 평가와 원본은 보존한다.
