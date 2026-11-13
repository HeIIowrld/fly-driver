# Highlands Short 독립 서킷 재검증 · 2026-11-13

[공유 검증 기록과 원본 SHA-256](evidence/corner-line-20261113.json). 아래 `runs/` 경로는 연구 원본과 실행 기록을 가리킨다.

Silverstone 1967·Vallelunga Extended·Monza 비교에 쓴 코너 판독과 별도로 동결한 경로 판독을 Highlands Short 저장 원본에 적용했다. Highlands는 두 판독의 학습과 설정 선택에서 제외했다. 코너 판독은 실제 4–20m 접근에서 방향과 시작 거리 ±5m가 일치할 때 적중으로 셌다. 보류와 미출력은 접근 분모에 포함했다. 코너 설정 `runs/vision-corner-left-risk-reallocation-20261113-01/frozen.json` · 경로 설정 `runs/vision-racing-line-motion-readout-20261113-01/frozen.json`

BMW Z4 GT3 오프라인 Hotlap의 내장 AI 주행을 CSP 네 방향 카메라로 녹화했다. 기존 카메라 앱은 정차·일시정지 상태의 브레이크 입력을 요구했다. 가상 브레이크로 정차 신호를 확인하고 장치를 반환했다. 백그라운드 카메라 요청의 초기 대기가 지연됐으나 원본 생산자와 소비자가 1,200묶음을 모두 저장한 것을 확인했다. 매 프레임 영상과 MMF 기록을 원본 영수증의 해시와 대조했다. 정차 663장, 코너 내부 304장, 직선 46장, 범위 밖 134장, 실제 접근 53장이었다. 기존 원본을 두 번 묶었으며 두 보관본의 8,410개 구성 파일 해시가 모두 같았다. 각각의 FlyVis 재생에서 최종 코너 출력 1,200장이 바이트 단위로 일치했다. 원본 영수증 `runs/vision-highlands-short-fixed-corner-capture-20261113-06/result.json` · 입고 `runs/vision-highlands-short-fixed-corner-capture-20261113-06/admission-summary.json` · 두 번째 재생 검증 `runs/vision-corner-frozen-transfer-20261113-01/verification.json`

## 코너 방향과 거리

| 저장 영상 | 좌 방향·거리 동시 적중 | 우 방향·거리 동시 적중 | 직선·범위 밖 오경보 | 내부 오경보 |
| --- | ---: | ---: | ---: | ---: |
| Silverstone 1967 | 0→1/11 | 2/13 유지 | 11→5 | 7→7 |
| Vallelunga Extended | 0→2/23 | 5/38 유지 | 6→4 | 10→10 |
| Monza | 0→1/24 | 4/42 유지 | 26→13 | — |
| Highlands Short | 0→2/21 | 1/32 유지 | 12→8 | 5→9 |

Highlands에서 새 좌 적중은 서로 다른 두 코너에 있었다. 845번은 실제 11.09m에 9m, 1014번은 실제 9.71m에 11m를 출력했다. 접근 53장 중 보류는 42장, 미출력은 0장이었다. 반대 방향 오류는 2건을 유지했고 거리 오류는 3→1건이었다. 코너 내부 오경보 순증은 4건으로 개발에서 정한 녹화별 상한 3건을 1건 넘었다. Monza는 기존 라벨로 비교했으며 내부 오경보 분모가 없다. Highlands 프레임 채점 `runs/vision-highlands-short-fixed-corner-evaluation-20261113-01/result.json` · 기존 세 영상 비교 `runs/vision-corner-left-risk-reallocation-20261113-01/comparison.json`

## 내부 경보 감소 재생

첫 독립 결과를 확인한 뒤 기존 개발 네 서킷에서 경보 억제 조건을 동결하고 네 저장 영상을 다시 채점했다. 우회전은 모두 기존 프레임별 출력을 유지했다. 원래 확인 문턱과 추가 문턱을 함께 고른 조건은 Highlands 내부 경보를 5건으로 줄였으나 Silverstone 좌회전 적중이 0/11이었고 Monza 거리 오류가 2건 늘었다. 기존 문턱을 고정하고 추가 좌회전에만 진입 가능성을 요구한 조건은 Highlands 내부 경보 5건, 좌회전 1/21이었고 Silverstone 0/11이었다. 개발 경보를 가장 적게 바꾸는 조건은 세 기존 비교의 좌회전 적중을 보존했고 Highlands 내부 경보는 9건이었다. 이 세 설정에 대한 Highlands 결과는 이미 평가한 원본을 재생한 것이다. 문턱 재선택 `runs/vision-corner-entry-phase-veto-20261113-01/comparison.json` · 추가 경보만 억제 `runs/vision-corner-added-left-phase-veto-20261113-01/comparison.json` · 최소 변경 `runs/vision-corner-minimal-added-veto-20261113-01/comparison.json`

개발에서 내부 경보 조건과 좌 적중 보존을 충족한 12개 후보를 기존 비교와 Highlands에서 모두 재생했다. 추가 좌회전에는 기존 확인 문턱과 함께 진입 확률이 코너 내부 확률 이상일 것을 요구하는 설정을 동결했다. Silverstone과 Vallelunga의 좌 적중 1/11·2/23을 보존했다. Monza 좌 적중은 기존 판독과 같은 0/24였고 범위 밖 경보는 26→9건이었다. Highlands 좌 1/21, 우 1/32, 내부 경보 5→7건으로 기존 한도를 충족했다. 설정 선택에 Highlands를 포함했다. 후보 12개 채점 `runs/vision-corner-phase-veto-frontier-20261113-01/comparison.json` · 새 고정 설정 `runs/vision-corner-phase-veto-frontier-20261113-01/frozen.json`

## 5·10·20m 경로

Highlands에서 537프레임에 경로를 출력했고 663프레임은 보류했다. 실제 주행의 20m 점을 채점할 수 있었던 526장에서 횡오차 평균은 고정 경계·몸 감각 판독 0.64m, 몸 감각 대조 0.53m, 차체 전방 직선 대조 1.32m였다. 같은 모델에 T4/T5를 추가한 대조는 0.74m였다. 차체 원점이 도로 안에 있던 경로 532개를 호길이 0.25m 간격으로 검사했다. 3개 경로가 도로 밖 표본을 포함했고, 31개 경로는 경계까지 1.1m 여유를 확보하지 못했다. 이 검사는 저장된 주행의 궤적과 도로 경계를 사용했다. 주행 추종과 랩타임은 이 비교에서 측정하지 않았다. 경로 채점 `runs/vision-highlands-short-fixed-line-evaluation-20261113-01/evaluation.json` · 독립 원본·코너·경로 검증 `runs/vision-highlands-short-fixed-line-evaluation-20261113-01/verification.json`
