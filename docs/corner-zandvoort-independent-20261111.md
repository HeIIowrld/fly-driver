# Zandvoort 독립 코너 판독 평가 · 2026-11-11

Magione 개발 재생에서 고정한 겹눈·도로 경계·경보 종료 판독을 Zandvoort에서 처음 평가했다. 런처 검색으로 Zandvoort BMW Z4 GT3 오프라인 핫랩을 열었다. 내장 AI로 이동하면서 주행선 표시를 끈 네 카메라의 원본 1,000묶음을 CSP로 기록했다. 원본 해시와 제작자 종료 영수증을 검증했다. 촬영 후 내장 AI를 끄고 일시정지했다. [고정 프로토콜](../runs/vision-corner-warning-end-runner-20261111-01/zandvoort-protocol.json) · [촬영 영수증](../runs/vision-zandvoort-independent-capture-20261111-02/result.json) · [종료 상태](../runs/vision-zandvoort-independent-capture-20261111-02/after.json)

입고한 영상은 좌 31프레임·물리 코너 9곳, 우 27프레임·물리 코너 8곳, 직선 89프레임, 범위 밖 178프레임, 코너 내부 542프레임, 정지 133프레임이었다. 부모와 후보는 동일한 원본과 FlyVis 특징을 사용했다.

| 지표 | 원래 판독 | 고정 후보 |
| --- | ---: | ---: |
| 4–20m 좌 방향 | 1/31 | 0/31 |
| 4–20m 우 방향 | 8/27 | 6/27 |
| 방향·거리 ±5m 결합 | 6/58 | 4/58 |
| 직선 오경보 | 31/89 | 16/89 |
| 범위 밖 오경보 | 58/178 | 33/178 |
| 보류 | 13 | 11 |
| 미출력 | 0 | 0 |
| 첫 안정 예고 거리 | 우 4.76m·10.48m | 우 10.48m |

공통으로 방향을 맞힌 연속 접근 4쌍의 거리 변화 평균 절대 오차는 4.706m로 같았다. 네 쌍 모두 실제로 접근 중이었고, 세 쌍에서는 예고 거리가 오히려 늘었다. 좌·우 방향, 결합 적중, 안정 예고, 거리 변화 기준에 미달해 **독립 평가에 실패했다.** [전체 비교](../runs/vision-zandvoort-independent-evaluation-20261111-01/result.json)

## 누락이 생긴 단계

- 접근 58프레임 모두에서 로컬 몸 감각과 연속 이동량을 읽었다. 프레임 간 로컬 전방 이동량 중앙값은 4.09m였다.
- 현재 영상의 양쪽 경계가 한 단면도 잡히지 않은 접근이 55개였고, 나머지 3개는 한 단면만 잡혔다. 시간 이력을 사용한 경계 굽음 후보도 58개 모두 없었다. 따라서 겹눈 점수와 결합해 복구한 경보가 없었다. [프레임별 실패 단계](../runs/vision-zandvoort-independent-evaluation-20261111-01/failure-stages.json) · [경계 입력 대조](../runs/vision-zandvoort-independent-evaluation-20261111-01/boundary-support.json)
- 경보 종료 단계는 4프레임에서 적용됐고 접근 적중이나 직선·범위 밖 오경보 집계에는 영향이 없었다. 적중 손실은 기존 혼합 판독의 출력을 경계가 복구하지 못한 결과다. [종료된 경보](../runs/vision-zandvoort-independent-evaluation-20261111-01/ended-warnings.jsonl)
- 대표 접근 두 장에는 포장과 잔디의 경계가 육안으로 보인다. 기존 표면 대비 검출이 실패한 이유를 색도 기준과 투영 위치로 분리하려면 같은 위치의 픽셀 값을 추가로 비교해야 한다. [네 시야 원본 예시](../runs/vision-zandvoort-independent-evaluation-20261111-01/boundary-support-samples.png)

이 평가는 사전에 고정한 첫 Zandvoort 평가로 보존한다. 이 원본을 이후 개선에 쓰면 다음 독립 평가는 다른 새 서킷에서 진행한다.
