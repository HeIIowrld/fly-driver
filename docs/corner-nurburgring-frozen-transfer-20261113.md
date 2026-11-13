# 고정 코너·레이싱 라인 판독의 Nürburgring Sprint 독립 평가 · 2026-11-13

[공유 검증 기록과 원본 SHA-256](evidence/corner-line-20261113.json). 아래 `runs/` 경로는 연구 원본과 실행 기록을 가리킨다.

첫 고정 코너 설정은 Highlands Short에서 좌 적중을 늘렸고 코너 내부 오경보 순증 상한을 1건 넘었다. 개발 네 서킷에서 좌 적중 보존과 내부 오경보 조건을 충족한 후보 12개를 확인했다. Silverstone·Vallelunga·Monza와 Highlands에서 이 후보들을 재생해, 추가 좌회전에서 진입 확률이 코너 내부 확률 이상인 경우만 경보하는 설정을 동결했다. 녹화별 오류 순증 상한은 직선·범위 밖 4건, 내부 3건, 반대 방향 1건, 거리 실패 1건으로 유지했다. [Highlands와 경보 억제 실험](corner-highlands-frozen-transfer-20261113.md) · 후보 채점 `runs/vision-corner-phase-veto-frontier-20261113-01/comparison.json` · 동결 설정 `runs/vision-corner-phase-veto-frontier-20261113-01/frozen.json`

동결 후 Nürburgring Sprint에서 BMW Z4 GT3 내장 AI의 오프라인 주행을 녹화했다. 이 서킷은 가중치 학습과 설정 선택에서 제외했다. 주행선 표시 없이 네 방향 영상 1,200묶음을 모두 저장했다. 정차 270장, 접근 84장, 코너 내부 437장, 직선 189장, 거리 범위 밖 218장, 위치 등록 실패 2장이었다. 실제 접근은 좌 19장(5개 코너), 우 65장(14개 코너)이었다. 기록용 정차 브레이크를 해제하고 장치를 반환했다. 게임은 녹화 후 일시정지했다. 원본과 반사 영상에서 각각 1,198장의 FlyVis 특징이 나왔다. 원본 기록 `runs/vision-nurburgring-sprint-capture-20261113-01/result.json` · 전체 영상 검증 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/capture-verification.json` · 특징 추출 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/archive-replay/result.json`

## 4–20m 코너 방향과 시작 거리

정답 방향과 거리 ±5m를 함께 맞힌 프레임을 적중으로 셌다. 보류와 미출력은 접근 분모에 들어간다. Monza는 기존 라벨을 재사용했으며 내부 오경보는 채점 범위에 없다.

| 영상 | 좌 동시 적중 | 우 동시 적중 | 직선·범위 밖 오경보 | 코너 내부 오경보 |
| --- | ---: | ---: | ---: | ---: |
| Silverstone 1967 | 0→1/11 | 2/13 유지 | 11→4 | 7→6 |
| Vallelunga Extended | 0→2/23 | 5/38 유지 | 6→3 | 10→8 |
| Monza | 0/24 유지 | 4/42 유지 | 26→9 | — |
| Highlands Short · 설정 선택에 포함 | 0→1/21 | 1/32 유지 | 12→7 | 5→7 |
| Nürburgring Sprint · 독립 | 0/19 유지 | 2/65 유지 | 18→12 | 9→5 |

Nürburgring은 모든 오류 상한을 충족했다. 반대 방향 오류는 4→0건, 거리 실패는 1건을 유지했다. 우 적중은 한 코너에서 나왔고 두 프레임 연속 적중은 없었다. 방향을 맞힌 우회전 3장의 시작 거리 평균 절대오차는 2.71m였다. 접근 84장 중 보류는 70장, 미출력은 0장이었다. 좌 접근 19장은 모두 보류했다. 원래 영상 경계에서 뽑은 후보를 추가 판독하는 과정에서 10장은 적격 좌 후보가 없었고 8장은 확인 점수가 문턱 아래였다. 한 장에서 진입 확률 조건이 걸렸다. 독립 코너 채점 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/result.json` · 좌회전 실패 지점 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/approach-audit.json` · 채점 검산 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/verification.json`

## 5·10·20m 레이싱 라인

영상 경계와 몸 감각을 쓴 고정 경로 판독은 930장에 경로를 출력하고 270장을 보류했다. 곡률을 줄인 기준선과 같은 20m 전방 단면을 비교할 수 있는 923장에서 횡오차는 1.29m였다. 같은 기준으로 몸 감각 대조 2.65m, T4/T5 추가 1.39m, 평가용 지도 중심선 0.68m였다. 출력 경로를 0.25m 간격으로 검사한 923개 중 2개가 도로 밖 표본을 포함했다. 실제로 지나간 AI 궤적을 별도로 예측한 고정 모델은 919장에서 20m 호길이 횡오차 0.65m, 몸 감각 대조 0.55m였다. 지도와 렌더 위치는 감독과 평가에 사용했다. 차량 추종과 유효 랩타임은 측정하지 않았다. 기준선 채점 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/optimized-line-result.json` · 실제 주행 궤적 채점 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/recorded-path-evaluation/evaluation.json` · 인과성·좌우 반사·결측 확인 `runs/vision-nurburgring-sprint-frozen-evaluation-20261113-01/line-verification.json`
