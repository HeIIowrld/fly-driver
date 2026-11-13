# Silverstone 우회전의 화면 경계와 지도 투영 · 2026-11-12

전방 원본을 먼저 보고 오른쪽에서 읽을 수 있는 표면 전환을 한 차례 수동 표시했다. 지도 투영은 표시를 저장한 뒤 계산했다. 네 물리 코너의 접근 13프레임에서 잔디·아스팔트 경계 3장, 어두운 띠의 전환 4장, 포장된 갓길과 주행면의 경계가 모호한 6장을 확인했다. 어두운 띠는 화면에서 추적할 수 있는 전환이며 공식 주행 경계로 확정하지 않았다. 픽셀 좌표는 320×180 원본 기준이다. [수동 표시](../runs/vision-silverstone-visible-edges-20261112-01/manual-right-boundary.json) · [원본 대조](../runs/vision-silverstone-visible-edges-20261112-01/inventory.json)

| 화면 전환 | 프레임 | 비교한 지도점 | 표시선과의 최단거리 중앙값 |
| --- | ---: | ---: | ---: |
| 잔디·아스팔트 | 3 | 18 | 2.60px |
| 어두운 띠 | 4 | 25 | 1.07px |
| 포장된 갓길에서 모호 | 6 | 0 | — |

비교는 표시한 선의 세로 범위 안에 들어온 점만 사용했다. 화면에서 경계가 거의 가로로 누워 있어 같은 높이에서 재는 가로 오차는 각각 22.40px, 10.70px였다. 실제 표시선까지의 짧은 거리와 가로 오차는 다른 값이다. 이 작은 표본에서 지도 경계 전체가 크게 어긋났다고 볼 수는 없다. [지도 대조 결과](../runs/vision-silverstone-visible-edges-20261112-01/result.json) · [잔디 경계 488번](../runs/vision-silverstone-visible-edges-20261112-01/comparison-488.png) · [어두운 띠 620번](../runs/vision-silverstone-visible-edges-20261112-01/comparison-620.png) · [포장 갓길 808번](../runs/vision-silverstone-visible-edges-20261112-01/comparison-808.png)

표시한 선에 직각으로 ±3–5px 떨어진 영상 띠에서 Lab 밝기 차 중앙값은 잔디·아스팔트 28, 어두운 띠 17이었다. 앞선 지도 기준 좌우 띠 검사에서는 누락 지점의 밝기 차 중앙값이 2였다. 두 표본의 위치와 채취 방향이 달라 직접적인 개선량은 계산할 수 없다. 화면에서 거의 가로로 보이는 경계는 좌우로만 샘플링하면 대비가 약할 수 있으므로, 이미지에서 경계를 가로지르는 방향으로도 조사할 근거가 생겼다. [직각 방향 대비 샘플](../runs/vision-silverstone-visible-edges-20261112-01/manual-normal-contrast.json)

양쪽 경계로 적합할 수 없는 프레임에서는 바깥쪽 한 경계가 충분히 이어지고, 직선과 일정 곡률보다 공통 시작점 모델로 잘 맞는 경우 경보를 추가했다. 현재와 직전 프레임의 방향과 차량 이동 보정 거리도 함께 맞아야 하는 조건을 비교했다. 좌·우 방향에 같은 규칙을 적용했다. Silverstone 두 조건 모두 좌 5/11·우 0/13·결합 4/24였다. 오경보는 기준 3건, 단일 경계 6건, 시간 확인을 더한 조건 3건이었다. Spa에서도 우 적중이 늘지 않았다. [한쪽 경계 코드](../runs/vision-silverstone-visible-edges-20261112-01/outer.py) · [두 서킷 결과](../runs/vision-silverstone-visible-edges-20261112-01/outer-result.json)
