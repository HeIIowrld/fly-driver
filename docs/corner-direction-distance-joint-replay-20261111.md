# 방향 판독과 연속 접근 거리의 결합 · 2026-11-11

요율·속도 기반의 방향 판독과 운동 시각을 함께 쓰는 판독을 비교했다. 기존 영상 거리와 연속 접근으로 학습한 거리를 절반씩 섞었다. 기존 경보는 방향을 유지하고 거리만 보정했다. 새 경보는 기존의 직선 또는 유무 판독 보류에서만 추가했고, 기존 영상 거리가 4–20m여야 했다. 두 거리 차이를 3m 또는 6m 이내로 제한한 조건도 시험했다. [실행 코드](../runs/vision-corner-direction-distance-joint-replay-20261111-01/joint_replay.py) · [거리 범위와 차이 검사](../runs/vision-corner-direction-distance-joint-replay-20261111-01/checks.log)

| Spa 사후 비교 | 좌 / 22 | 우 / 49 | 방향·거리 ±5m / 71 | 오경보 / 605 | 안정 예고 / 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 사전 고정 판독 | 2 | 15 | 10 | 80 | 3 |
| 몸 감각 방향과 보정 거리, 문턱 0.40 | 3 | 24 | 19 | 86 | 7 |
| 운동 시각과 몸 감각, 거리 차 ≤3m, 문턱 0.55 | 3 | 15 | 14 | 80 | 5 |
| 운동 시각과 몸 감각, 거리 차 ≤6m, 문턱 0.55 | 3 | 17 | 16 | 80 | 6 |

6m 조건에서 기존 세 코너 외에 코너 ID 31·12·17을 각각 시작 8.07m, 4.91m, 6.69m 전에 안정 예고했다. 공통 연속 6쌍의 거리 변화 오차는 5.01→4.81m였으며, 사전에 고정한 비교 기준 4.70m와 안정 예고 기준은 통과하지 못했다. [전체 점수와 안정 예고](../runs/vision-corner-direction-distance-joint-replay-20261111-01/result.json) · [프레임별 출력](../runs/vision-corner-direction-distance-joint-replay-20261111-01/rows.jsonl)

36개 비교에서 평가 서킷과 Spa를 학습 및 문턱 선택에서 제외했다. 문턱 선택용 예측도 각 서킷을 학습에서 빼고 만들었다. 두 비교에서 거리 보정을 적용하지 않은 대조 조건의 문턱이 선택됐고 평가 지표는 그대로였다. Spa에서 오경보가 늘지 않았던 6m 조건을 같은 문턱에서 평가하면 Magione의 결합 적중은 71→67개, Imola의 오경보는 45→47개였다. 고정 판독은 유지했다. [전체 7,287개 채점과 분할](../runs/vision-corner-direction-distance-joint-replay-20261111-01/result.json) · [실행 영수증](../runs/vision-corner-direction-distance-joint-replay-20261111-01/receipt.json)

최신 CSP 확인에서 Spa 내장 AI가 주행 중이었고 녹화는 비활성이었다. Silverstone 1967 독립 평가용 영상은 아직 확보되지 않았다. [독립 평가 준비](corner-silverstone-independent-preflight-20261111.md)
