# 측면 경계의 연속 추적과 영상 위치 학습 · 2026-11-12

## 인접 단면을 함께 고르는 실험

측면 카메라의 기존 대비 후보 중 연속된 누락 단면을 한 묶음으로 선택했다. 양쪽에 현재 관측이 있으면 그 위치를 고정했다. 도로 쪽으로 갑자기 튀거나 단면 사이의 기울기가 크게 바뀌는 경로는 점수를 낮췄다. Magione·Spa에서 기존보다 관측 수가 줄지 않는 설정 가운데 서킷별 횡오차의 최대 상대값을 최소화했다. Silverstone·Vallelunga에는 고정한 설정을 썼다. 작은 후보 문제 25개에서 동적 계획법의 점수가 전수 탐색과 일치했다. [연속 추적과 개발 평가](../runs/vision-corner-side-chain-20261112-01/training.json) · [검증](../runs/vision-corner-side-chain-20261112-01/verification.json)

개발 평균 횡오차는 Magione 0.94→0.70m, Spa 1.73→1.66m였다. Silverstone 우회전 접근에서 새로 채운 10점은 1.67→1.71m였고 우회전 방향·거리 적중은 두 판독 모두 2/13이었다. Vallelunga에서 연속 추적 후 같은 후보에 T4/T5를 추가하면 우 적중은 4→6/38, 직선·범위 밖 경보는 14→11건, 내부 경보는 5→10건이었다. [프레임별 재생](../runs/vision-corner-side-chain-20261112-01/result.json)

Silverstone의 누락 10점에 대해 이전 최대 네 프레임의 오른쪽 경계를 국소 속도·요율로 현재 좌표에 옮겼다. 7점은 실제 위치 1.25m 이내의 과거 추정이 한 번 이상 있었다. 현재 보이는 이웃과 1.25m 이내로 일치하는 조건까지 통과한 정확한 추정은 0점이었다. 물리 위치는 이 진단의 채점에만 사용했다.

## 측면 영상의 물리 경계 위치 학습

기존 위치별 후보 점수와 별도로, 측면 영상에서 경계선이 나타날 픽셀 위치를 학습했다. 왼쪽 영상은 가로 반사해 오른쪽 영상과 방향을 통일했다. Magione·Spa의 측면 영상 크롭 4,178장과 카메라에 투영한 물리 노면 경계를 사용했다. 반대 서킷에서 학습한 모델로 각 개발 서킷을 채점해, 경계 위치 모델과 기존 대비를 함께 쓰는 설정을 골랐다. 영상에 투영된 물리 경계의 가림은 따로 판정하지 않았다. [모델과 학습](../runs/vision-corner-side-heatmap-20261112-01/train_heat.py) · [서킷 제외 평가](../runs/vision-corner-side-heatmap-20261112-01/training.json)

| 평가 | 기존 측면 대비 | 픽셀 경계 학습과 대비 |
| --- | ---: | ---: |
| Magione 개발 횡오차 | 0.91m | 0.87m |
| Spa 개발 횡오차 | 1.74m | 1.69m |
| Silverstone 우회전 접근의 추가 10점 횡오차 | 1.67m | 1.79m |
| Silverstone T4/T5 우 동시 적중 | 2/13 | 1/13 |
| Vallelunga T4/T5 우 동시 적중 | 6/38 | 5/38 |

픽셀 경계 학습 조건에서 Silverstone의 같은 후보에 T4/T5를 추가하면 우 적중은 1건으로 같았고, 직선·범위 밖 경보는 17→14건이었다. Vallelunga는 우 적중 3→5건, 직선·범위 밖 경보 12→9건, 내부 경보 7→12건이었다. 기존 고정 판독기에 두 실험을 적용하지 않았다. [영상 비교](../runs/vision-corner-side-heatmap-20261112-01/result.json)

## 추가 개발 영상의 물리 경계 라벨

Imola 1,200장과 Laguna Seca 1,000장의 기존 녹화에서 물리 노면 경계 라벨을 추가로 추출했다. 등록된 유효 노면과 비유효 노면의 연결을 확인해 라벨을 붙였다. 등록되지 않은 물리 메시가 표본 부근의 지면과 겹치면 해당 단면의 학습 라벨을 제외하도록 했다. Laguna Seca의 기존 엄격한 추출기는 등록되지 않은 물리 메시에서 종료됐다. 새 추출에서는 그 메시의 노면 종류를 가정하지 않았다. [추가 라벨 추출 코드](../runs/vision-corner-extra-side-labels-20261112-01/export_guarded_labels.py)

Imola 측면 카메라의 확인된 투영 표본은 오른쪽 3,574점과 왼쪽 4,970점이다. Laguna Seca는 오른쪽 14점과 왼쪽 70점으로 표본이 적었다. 두 파일의 영상 순서, 해시, 투영 범위를 검사했다. 두 라벨은 이번 픽셀 경계 학습에서 사용하지 않았다. [Imola 라벨 근거](../runs/vision-corner-extra-side-labels-20261112-01/imola-train-labels.json) · [Laguna Seca 라벨 근거](../runs/vision-corner-extra-side-labels-20261112-01/ks_laguna_seca-train-labels.json) · [출력 검증](../runs/vision-corner-side-heatmap-20261112-01/verification.json)
