# 영상 경계에서 읽은 코너 시작 거리 · 2026-11-11

Magione, Laguna Seca, Zandvoort, Mugello의 네 방향 영상 4,887묶음을 동일한 상대 색 대비 경계 판독기로 재생했다. 기존에 기록한 경계 후보의 방향과 거리를 프레임별로 재현했다. 스플라인은 평가 정답으로만 썼다. 네 서킷 모두 이미 공개한 평가 결과가 있으므로 여기서는 개발 비교로 사용한다. [재생 코드](../runs/vision-corner-onset-support-20261111-01/audit.py) · [전체 비교](../runs/vision-corner-onset-support-20261111-01/transfer-result.json)

## 방향은 맞았지만 시작 위치는 서킷별로 달랐다

거리 오차는 예고 거리에서 실제 남은 거리를 뺀 값이다. 양수는 시작을 실제보다 멀게 읽은 경우다.

| 서킷 | 경계 방향 적중 좌·우 | 강한 곡률 시작 거리 오차 중앙값 좌·우 | 두 프레임 연속으로 경계 방향이 맞은 쌍 | 강한 곡률에서 ±5m | 약한 같은 방향 굽음까지 거슬러 ±5m |
| --- | --- | --- | ---: | ---: | ---: |
| Magione | 18/64 · 13/79 | +10.51m · +7.31m | 21 | 1 | 1 |
| Laguna Seca | 3/38 · 2/25 | +0.55m · +6.41m | 1 | 3 | 3 |
| Zandvoort | 4/31 · 5/27 | +2.74m · −6.57m | 2 | 4 | 4 |
| Mugello | 2/24 · 8/40 | +2.48m · +2.67m | 1 | 7 | 7 |

강한 굽음에 앞선 연속 단면 중 기존 곡률 기준의 절반을 넘는 같은 방향 구간까지 돌아가도 ±5m 적중은 유지됐다. Magione에서는 방향이 맞은 연속 21쌍의 거리 변화 평균 절대 오차가 좌 0.64m, 우 0.91→0.48m였다. 절대 시작 위치와 연속 프레임에서의 거리 변화가 서로 다르게 움직였다. [Magione](../runs/vision-corner-onset-support-20261111-01/magione/result.json) · [Laguna Seca](../runs/vision-corner-onset-support-20261111-01/ks_laguna_seca/result.json) · [Zandvoort](../runs/vision-corner-onset-support-20261111-01/ks_zandvoort/result.json) · [Mugello](../runs/vision-corner-onset-support-20261111-01/mugello/result.json)

## 경계 관측 범위로 보정한 거리의 전이

강한 곡률 위치와 그 위치를 포함하는 연속 관측 단면의 첫 거리 사이를 보간했다. 한 서킷을 통째로 빼고 나머지 세 서킷의 거리 오차를 서킷별 동일 비중으로 최소화했다. 아래의 평균 절대 오차는 경계 방향이 맞은 프레임에서 계산했다. 전체 출력 비교에서는 연속 경계가 같은 방향을 보이고 몸 상대 이동과 일치할 때, 직선 출력에만 경보를 추가했다.

| 제외 서킷 | 강한 곡률 → 제외 보정 거리 오차 | 전체 판독 좌·우·결합 적중 | 직선·범위 밖 오경보 |
| --- | ---: | --- | ---: |
| Magione | 9.48→7.81m | 13·62·71 유지 | 27 유지 |
| Laguna Seca | 4.96→4.60m | 4·0·4 유지 | 12→14 |
| Zandvoort | 5.20→6.96m | 3·10·8 → 3·11·8 | 61→64 |
| Mugello | 3.13→3.73m | 4·1·2 → 4·2·3 | 5→7 |

Mugello의 우 방향·결합 적중이 각각 한 건 늘었고 안정 예고는 기존과 같은 한 곳이었다. 좌 적중은 네 서킷 모두 늘지 않았다. 보정은 두 서킷에서 거리 오차를 키웠고, 전체 판독에 적용하면 세 서킷에서 오경보가 늘었다. [서킷 제외 재생](../runs/vision-corner-onset-support-20261111-01/transfer-result.json) · [변경 프레임](../runs/vision-corner-onset-support-20261111-01/counterfactual-changes.jsonl)

## 경계 판독이 누락되는 단계

기존 조건은 상대 이동으로 정렬한 과거 경계의 굽음이 같은 위치에서 계속 뒷받침될 때만 후보를 낸다. 이를 같은 방향의 관측 두 개만 요구하는 조건과, 융합 경계만 보는 조건으로 바꿨다. 아래 숫자는 영상 경계 단계의 후보 수다. 최종 경보에는 영상 유무·거리 관문이 추가된다.

| 서킷 | 좌·우 적중 기존 → 두 관측 → 융합 경계 | 직선·범위 밖 후보 기존 → 두 관측 → 융합 경계 |
| --- | --- | --- |
| Magione | 18·13 → 38·43 → 33·52 | 77→291→383 |
| Laguna Seca | 3·2 → 3·4 → 9·8 | 14→24→148 |
| Zandvoort | 4·5 → 7·7 → 9·13 | 34→67→134 |
| Mugello | 2·8 → 3·11 → 16·21 | 28→59→239 |

상대 이동 일치까지 요구했을 때 Mugello의 두 관측 조건은 맞는 접근 1→3개, 직선·범위 밖 후보 2→6개였다. 단면 확인을 완화하는 방법과 공통 거리 보정은 채택하지 않았다. 다음 비교에서는 여러 프레임이 같은 굽음을 관측했는지와 그 굽음이 실제 시작을 나타내는지를 함께 확인한다. [Magione](../runs/vision-corner-onset-support-20261111-01/magione/curvature-rejection.json) · [Laguna Seca](../runs/vision-corner-onset-support-20261111-01/ks_laguna_seca/curvature-rejection.json) · [Zandvoort](../runs/vision-corner-onset-support-20261111-01/ks_zandvoort/curvature-rejection.json) · [Mugello](../runs/vision-corner-onset-support-20261111-01/mugello/curvature-rejection.json)

별도 평가 후보지로 Spa와 Silverstone 1967의 설치 모델과 양방향 곡률 구간을 확인했다. 기하에서 집계한 좌·우 구간은 Spa 12·22, Silverstone 1967은 3·6이었다. 새 영상의 입고 여부는 실제 접근 프레임과 연속 채점 가능 범위로 확인한다. [설치 스플라인 확인](../runs/vision-corner-onset-support-20261111-01/installed-track-geometry.json)
