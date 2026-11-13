# 전후좌우 영상과 코너 판독 가중치 검토 · 2026-11-11

Magione 원본 1,887개 캡처를 재생한 결과, 전·좌·우·후 영상이 모두 코너 판독 점수에 들어갔다. 세 시점과 두 영상 분기가 모두 가용한 1,879개에서 카메라별 기여도를 더해 실제 점수를 재구성했다. 최대 차이는 2.56×10⁻¹⁵였다. 판독 방향과 경보는 보존된 실행과 일치했다. 각 원본의 처음·중간·마지막에서 네 BMP를 실제로 디코딩해 320×180 크기와 화면별로 다른 해시를 확인했다. [카메라별 가중치와 무력화 결과](../runs/vision-corner-camera-weight-audit-20261111-01/result.json).

## 각 화면에서 뽑는 특징

각 화면의 RGB를 밝기로 바꾸고, 화면 전체에 펼친 721개 육각 수용점에서 주변 픽셀을 가중 평균한다. 각 카메라는 독립적인 FlyVis 상태를 이어간다. T4a–d와 T5a–d의 활성에 tanh를 적용해 화면의 4×4 위치 구역별로 평균낸다. 같은 화면·시점·채널 안에서 16개 구역의 평균을 뺀 값이 판독층으로 들어간다. 이 입력에는 차선이나 연석 같은 물체 이름이 없다. 화면의 위치와 운동 채널 반응이 있다. [영상 투영 코드](../src/flybrain_drive/flyvis_camera.py), [특징 추출 코드](../scripts/extract_next_corner_flyvis.py).

한 영상 분기의 입력은 **세 시점 × 네 카메라 × 여덟 운동 채널 × 16구역 = 1,536개**다. 카메라 순서는 전→좌→우→후다. 반전 분기는 원본의 좌·우 카메라를 교환하고 각 화면을 좌우 반전한 뒤 같은 신경망을 별도 상태로 처리한다. 판독층은 반전 결과의 좌·우 점수를 원래 방향으로 되돌리고 원본과 50%씩 평균한다. 코너 방향, 코너 존재 여부, ordinal 거리, 방향·거리 결합 판독이 이 구조를 사용한다. [반전 순열](../src/flybrain_drive/retinal_reflection.py), [판독 코드](../scripts/infer_frozen_next_corner.py).

## 화면별 수치

| 화면 | 방향 | 코너 존재 | ordinal 거리 | 방향·거리 결합 |
| --- | ---: | ---: | ---: | ---: |
| 전 | 36.1% | 36.4% | 33.3% | 27.6% |
| 좌 | 23.7% | 20.7% | 21.9% | 25.5% |
| 우 | 23.7% | 20.7% | 21.9% | 25.5% |
| 후 | 16.5% | 22.3% | 22.9% | 21.4% |

표는 표준화 스케일과 구역 평균 제거를 반영한 유효 계수의 제곱합 비중이다. 방향은 좌·우 각각의 점수에서 배경 점수를 뺀 계수를 사용했다. 좌·우 비중은 반전 분기의 대칭 때문에 같다. 실제 영상에서 계산한 좌·우 점수 기여도는 각 화면의 특징값에 따라 달라진다. [계산 방법과 원본 수치](../runs/vision-corner-camera-weight-audit-20261111-01/audit.py).

| 원본 분기 | 방향 계수가 가장 큰 특징 | 좌−배경, 우−배경 계수 | 두 분기에서 해당 화면 특징을 0으로 만들 때 방향 점수 1위가 바뀐 프레임 | 경보 유무가 바뀐 프레임 |
| --- | --- | --- | ---: | ---: |
| 전 | 가장 오래된 시점 T5d · 위에서 2행, 왼쪽에서 3열 | +1.056, +1.147 | 642/1,879 | 268 |
| 좌 | 최신 시점 T5c · 4행 3열 | +0.713, +1.175 | 609/1,879 | 240 |
| 우 | 가장 오래된 시점 T5c · 2행 3열 | −0.746, −0.987 | 692/1,879 | 302 |
| 후 | 가장 오래된 시점 T5c · 1행 1열 | −0.550, −0.618 | 1,123/1,879 | 238 |

후방의 방향 계수 비중은 16.5%였다. 저장 영상에서 실제로 들어온 후방 특징의 평균 절대 방향 점수 기여는 좌−배경 1.235, 우−배경 1.044로 네 화면 중 가장 컸다. 이 대조는 신경망을 지난 특징을 0으로 만든 민감도 검사다. 이 수치만으로 카메라를 제거했을 때의 주행 성능이나 후방의 일반화 기여를 판단할 수 없다. 위 구역은 신경망 활성의 집계 위치이며 각 계수가 특정 원본 픽셀 하나를 가리키지는 않는다. [프레임별 점수 기여](../runs/vision-corner-camera-weight-audit-20261111-01/score-contributions.jsonl).

## 레이싱 라인에서 화면이 쓰이는 방식

현재 국소 라인 경로는 네 이미지에서 고정 카메라 기하로 차량 앞쪽 지면 후보를 투영하고, 해당 위치를 볼 수 있는 카메라의 색 또는 노면 대비로 좌우 경계를 찾는다. 네 이미지를 받지만 경계에 기여하는 화면은 관측 위치에 따라 달라진다. 평평한 차체 기준 4m 단면의 가로 후보 281개 중 좌·우 화면은 각각 106개를 볼 수 있고 전·후 화면은 0개다. 20m에서는 전방 281개, 좌·우 각각 12개, 후방 0개다. 후방은 점검한 2–32m 전방 단면에 직접 투영되지 않는다. 아카이브의 `front-far` 옵션은 12m부터 노면 대비 검색에 전방만 사용한다. [경계 투영](../src/flybrain_drive/pixel_road_boundary.py), [노면 대비 옵션](../src/flybrain_drive/surface_road_boundary.py), [투영 가시성 수치](../runs/vision-corner-camera-weight-audit-20261111-01/line-camera-visibility.json).

## GitHub의 외부 구현

| 외부 저장소 | 코드에서 확인한 방법 | 비교할 수 있는 범위 |
| --- | --- | --- |
| **Auren Lane Detection** | 현재 프레임 경계 중심점에서 원호와 부호 있는 곡률을 추정한다. 연속 증거로 코너 진입과 해제를 확인하고, 경계가 잠깐 끊기면 유지한 곡률을 현재 관측에 맞춰 다시 배치한다. | 영상 기반 방향과 곡률의 비교군. 해당 구현은 근거리 원호를 추정하며 4–20m 코너 시작 거리까지 제공하지 않는다. [원호와 상태 갱신 코드](https://github.com/parhamhmtt/auren-lane-detection/blob/main/src/worldcar/direct_curve.py) |
| **Lane Detection with Turn Prediction** | 전방 영상의 사다리꼴 영역을 위에서 본 영상으로 바꾸고 흰색·노란색 차선을 2차식으로 맞춘다. 이차항 부호로 좌·우·직선을 표시한다. | 단순한 영상 방향 기준선. 고정된 차선 색과 투영 영역을 쓰며 미터 단위 코너 시작 거리는 출력하지 않는다. [Turn_Predict.py](https://github.com/HemanthJoseph/Lane-Detection-with-Turn-Prediction-for-Self-Driving/blob/main/src/Turn_Predict.py) |
| **F1TENTH racecar_camera** | 전방 단안 영상에서 실내 트랙 테이프의 색과 윤곽 중심을 검출해 차선 추종에 사용한다. | 시각 입력에서 차선 추종까지 연결한 사례. 입력은 표시가 있는 트랙 영상이다. [영상 검출 코드](https://github.com/wenth1021/racecar_camera/blob/master/src/lane_detection_node.py) |
| **TUMFTM trajectory_planning_helpers** | 기준선과 좌우 폭을 받고, 법선 방향으로 위치를 조정하는 QP로 곡률을 줄인다. 차량 폭과 곡률 제한을 제약에 둔다. 열린 구간의 시작·끝 방향 조건도 지원한다. | 관측 경계 안의 진입·통과·탈출 경로 비교. 전체 지도와 차량 동역학을 쓰는 프로젝트의 최소시간 해는 교사·평가에 해당한다. [최소 곡률 코드](https://github.com/TUMFTM/trajectory_planning_helpers/blob/master/trajectory_planning_helpers/opt_min_curv.py), [전체 레이싱 라인 최적화](https://github.com/TUMFTM/global_racetrajectory_optimization) |
| **ac-copilot-trainer-physics** | Assetto Corsa의 `fast_lane.ai`를 읽고 절대 위치·방향·속력으로 앞쪽 목표점을 찾아 pure pursuit를 수행한다. | 같은 시뮬레이터에서 지도 기반 교사 라인과 추종을 비교할 수 있다. [AC 경로 및 추종 코드](https://github.com/pgorokh/ac-copilot-trainer-physics/blob/main/tools/ac_harness/ai_line.py) |

외부 코드 검토는 소스 확인 범위다. 이 저장소들의 성능을 Flybrain 영상에 재생해 비교하지 않았다. 사용자 게임에서 Laguna Seca를 확인했고, 설치된 CSP 상태에서도 트랙·차종·주행선 OFF를 확인했다. 상태 조회 시 세션이 실행 중이고 브레이크가 0이어서 네 화면의 실제 Laguna Seca 녹화는 아직 시작하지 않았다. [Laguna Seca 상태 영수증](../runs/vision-corner-laguna-session-20261111-01/result.json).
