# 세 서킷의 측면 경계 학습과 Silverstone 확인 실패 · 2026-11-12

## Imola를 추가한 학습

Magione·Spa에 Imola를 더해 측면 영상 크롭 5,803장으로 경계 픽셀 위치를 학습했다. 개발 서킷을 하나씩 제외해 영상 대비, 학습한 위치 점수, 두 점수의 결합 및 기존 대비 후보를 바꾸는 데 필요한 점수 차이를 비교했다. 각 서킷의 기존 대비 방식에 대한 횡오차 비율 중 가장 큰 값을 최소화하는 조건으로 학습 위치 점수를 골랐다. 두 서킷으로 학습한 이전 모델과 기존 대비 방식의 판독을 같은 영상에서 재현했다. [학습과 동결 설정](../runs/vision-corner-side-three-tracks-20261112-01/training.json) · [프레임별 검증](../runs/vision-corner-side-three-tracks-20261112-01/verification.json)

| 평가 | 기존 측면 대비 횡오차 | 선택된 위치 모델 횡오차 |
| --- | ---: | ---: |
| Magione 제외 교차 평가 | 0.91m | 0.75m |
| Spa 제외 교차 평가 | 1.74m | 1.55m |
| Imola 제외 교차 평가 | 1.44m | 1.11m |
| Silverstone 우회전 접근의 추가 10점 | 1.67m | 1.89m |

세 서킷으로 학습하고 이전과 같은 대비 결합을 쓴 조건에서 Silverstone 추가 10점의 횡오차는 1.95m였다. 개발 영상으로 고른 조건에서 Silverstone의 우회전 방향·거리 적중은 경계와 T4/T5 모두 2/13이었다. Vallelunga의 같은 후보에서 T4/T5를 추가하면 우 적중 3→6/38, 전체 적중 6→10건, 직선·범위 밖 경보 15→11건, 내부 경보 6→13건이었다. Monza 보존 라벨 989장에서는 우 적중이 3→4/42, 범위 밖 경보가 12→25건이었고 좌 동시 적중은 두 방식 모두 0/24였다. 이 조건은 기존 고정 판독기에 적용하지 않았다. [Silverstone·Vallelunga 재생](../runs/vision-corner-side-three-tracks-20261112-01/result.json) · [Monza 전이](../runs/vision-corner-side-three-tracks-20261112-01/monza.json)

## 물리 경계로 바꿔 확인한 병목

Silverstone의 1,200프레임을 다시 재생하면서, 물리 메시에서 노면 전환이 확인되고 카메라에 투영된 경계 좌표를 각 조건에 맞춰 넣었다. 정답 방향과 거리는 재생 뒤 채점에만 썼다. 물리 경계와 렌더 좌표를 쓴 아래 결과는 원인을 확인하는 오프라인 진단이다. 픽셀 투영에는 가림 여부를 따로 판정하지 않은 표본이 포함된다. [진단 코드](../runs/vision-corner-side-three-tracks-20261112-01/geometry_intervention.py) · [13장별 후보와 경보](../runs/vision-corner-side-three-tracks-20261112-01/geometry-intervention.json)

| Silverstone 우 접근 13장 | 맞는 후보가 있는 장 | 맞는 후보를 고른 장 | 경계 최종 동시 적중 | T4/T5 최종 동시 적중 |
| --- | ---: | ---: | ---: | ---: |
| 기존 경계 | 7 | 6 | 2 | 2 |
| 누락된 오른쪽 점만 물리 좌표로 채움 | 8 | 6 | 2 | 2 |
| 관측된 오른쪽도 물리 좌표로 교체 | 10 | 9 | 3 | 3 |
| 좌우 관측을 물리 좌표로 교체 | 10 | 9 | 1 | 1 |

동시 적중은 방향과 시작 거리 ±5m를 함께 맞힌 경우다. 오른쪽을 교체하고 맞는 후보를 고른 9장 중 6장은 두 확인기 모두 점수 문턱에서 탈락했다. 양쪽을 교체하면 맞는 후보를 고른 9장 중 8장이 같은 이유로 탈락했다. 이 장들에는 코너 내부 차단이 걸리지 않았다. 실제 영상에서의 경계 정확도와, 경계가 개선된 후보의 확인 점수를 함께 학습할 근거다. [확인 점수와 탈락 사유](../runs/vision-corner-side-three-tracks-20261112-01/geometry-confirmation-audit.json)

Mugello와 Zandvoort의 기존 영상 각 1,000장에서 별도 물리 경계 라벨을 검증했다. 측면의 오른쪽 표본은 각각 3,651점과 4,035점, 왼쪽 표본은 3,498점과 2,769점이다. 이번 측면 모델 학습에는 이 두 영상의 라벨을 사용하지 않았다. 다음 확인기 학습에서는 개발 서킷의 실제 영상 경계 후보와 물리 경계로 교정한 후보를 함께 비교하고, 코너 방향·거리와 오경보를 동일한 기준으로 채점할 수 있다. [Mugello 라벨](../runs/vision-corner-extra-side-labels-20261112-01/mugello-train-labels.json) · [Zandvoort 라벨](../runs/vision-corner-extra-side-labels-20261112-01/ks_zandvoort-train-labels.json)
