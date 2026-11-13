# 차량 이동으로 영상의 굽음 위치 확인 · 2026-11-11

Magione, Laguna Seca, Zandvoort, Mugello의 네 방향 영상 4,887묶음을 재생했다. 차량 기준으로 정렬한 현재 영상과 과거 영상에서 양쪽 도로 경계의 접선 방향 변화를 비교했다. 직선으로 출력한 프레임에만 새 경보를 추가했고, 방향·시작 거리와 정답은 기존 공통 채점기로 비교했다. 이번 비교는 이미 결과를 본 개발 자료에 한정된다. [직접 재생 코드](../runs/vision-corner-short-baseline-20261111-01/replay.py) · [인접 단면 합산 코드](../runs/vision-corner-pooled-curvature-20261111-01/replay.py)

## 짧은 구간의 영상 굽음

세 단면에 걸친 각 경계의 접선 방향 변화로 굽음을 계산했다. 2m 간격의 인접 두 위치에서 같은 방향이 현재와 과거 영상에 반복돼야 한다는 조건은 네 서킷에서 새 방향 적중을 만들지 못했다. 두 위치의 굽음을 먼저 합산하면 Mugello에서 좌 4→6, 우 1→16, 결합 2→12개가 됐지만 직선·범위 밖 오경보도 5→98개로 늘었다. 굽음 직전의 양쪽 직선 관측이나 다음 프레임의 상대 이동 일치를 요구하면 그 추가 적중은 사라졌다. [Magione](../runs/vision-corner-pooled-curvature-20261111-01/magione/result.json) · [Laguna Seca](../runs/vision-corner-pooled-curvature-20261111-01/ks_laguna_seca/result.json) · [Zandvoort](../runs/vision-corner-pooled-curvature-20261111-01/ks_zandvoort/result.json) · [Mugello](../runs/vision-corner-pooled-curvature-20261111-01/mugello/result.json)

## 같은 굽음 위치의 연속 확인

가장 가까운 굽음만 택하면 다음 프레임에서 다른 후보를 고를 수 있어, 여러 후보를 유지하고 이동으로 예측한 위치와 대조했다. 한 프레임에서 0.75m 이상 움직여야 하는 비교에서는 Magione 접근 143프레임이 전부 탈락했다. 프레임 간 이동량 중앙값은 Magione 0.41m, Laguna Seca 4.33m, Zandvoort 4.09m, Mugello 4.52m였다. 이어지는 관측에서 직전 위치와 처음 위치를 각각 투영해 둘 다 1m 이내일 때만 같은 굽음으로 유지했고, 여러 프레임에서 이동량을 합산해 0.75m 이상일 때 출력했다. 한 프레임에서 출력을 강제하거나 미래 프레임을 참조하지 않았다. [후보 위치 대조 코드](../runs/vision-corner-motion-association-20261111-01/replay.py) · [누적 이동 코드](../runs/vision-corner-motion-association-20261111-01/accumulated.py)

| 서킷 | 기존 좌·우·결합 | 누적 이동으로 추가 후 | 직선·범위 밖 오경보 | 영상 유무·거리 조건도 적용한 오경보 |
| --- | --- | --- | --- | --- |
| Magione | 13·62·71 | 13·62·71 | 27→36 | 27 |
| Laguna Seca | 4·0·4 | 4·0·4 | 12→13 | 13 |
| Zandvoort | 3·10·8 | 3·11·8 | 61→65 | 61 |
| Mugello | 4·1·2 | 4·1·2 | 5→17 | 7 |

누적 이동은 Magione에서 한 프레임 이동이 0.75m보다 작은 12개에 경보를 추가했지만, 실제 방향 적중은 0개였다. 영상 유무·거리 조건까지 적용하면 네 서킷 모두 좌·우·결합 적중이 기존과 같았다. 이 결과로 같은 굽음의 재관측은 확인할 수 있지만 실제 코너 시작을 안정적으로 골랐다는 근거는 확보하지 못했다. 이 방식의 좌·우 반전, 결측, 느린 이동, 누적 위치 이탈, 양방향 동률 대조를 확인했다. [전체 채점 결과](../runs/vision-corner-motion-association-20261111-01/accumulated-result.json) · [변경 프레임](../runs/vision-corner-motion-association-20261111-01/accumulated-changes.jsonl)

기존 판독을 유지한다. 다음 개발 비교에서는 코너 안이나 먼 거리에서도 반복되는 굽음에 대해, 굽음 이전 도로 관측과 T4/T5의 운동 시각 변화가 접근 여부를 구분하는지 살핀다. Spa와 Silverstone 1967은 별도 평가 후보지이며 이번에는 설치 자료 외 새 영상을 사용하지 않았다.
