# 화면 경계를 가로지르는 대비로 코너 판독 · 2026-11-12

앞서 수동 표시한 Silverstone 전방 영상의 같은 지점에서 각각 ±3–5픽셀의 가로 띠와 경계에 직각인 띠를 비교했다. 밝기 차 중앙값은 잔디와 아스팔트 전환 27점에서 7→28 Lab L, 어두운 띠 전환 36점에서 6→17 Lab L였다. 수동 표시가 가능한 두 유형을 본 사후 진단이다. [짝지은 대비 측정](../runs/vision-corner-image-normal-20261112-01/paired-image-contrast.json)

판독에서는 수동 좌표 대신 현재 영상의 Lab 기울기로 샘플 방향을 고르고, 그 방향의 양쪽 픽셀 띠를 비교했다. 기존 검출이 비었던 좌우 경계만 채웠다. 인접 단면의 경계 좌표가 1.4m 이내일 때만 받는 조건도 시험했다. 양쪽 경계가 같이 굽는 시작점을 계산하는 규칙을 그대로 사용했다. Silverstone과 Spa의 기존 영상 각 1,200묶음을 사후 비교했다. [실행 코드](../runs/vision-corner-image-normal-20261112-01/image_normal.py) · [전체 결과](../runs/vision-corner-image-normal-20261112-01/result.json)

| Silverstone 비교 | 좌 / 11 | 우 / 13 | 방향·거리 동시 적중 / 24 | 오경보 |
| --- | ---: | ---: | ---: | ---: |
| 기존 양쪽 경계 | 5 | 0 | 4 | 3 |
| 경계를 가로질러 빈 단면 채움 | 3 | 1 | 4 | 8 |
| 이웃 단면 확인 뒤 채움 | 6 | 0 | 4 | 7 |

새로 맞힌 우회전은 코너 ID 0의 623번 프레임이었다. 시작 거리는 실측 6.02m, 예측 8m였다. Spa에서도 가로지르는 대비 방식으로 우 적중이 0→1, 결합 적중이 0→2, 오경보가 3→18로 변했다. [Silverstone 프레임별 채점](../runs/vision-corner-image-normal-20261112-01/ks_silverstone1967-frames.jsonl) · [Spa 프레임별 채점](../runs/vision-corner-image-normal-20261112-01/spa-frames.jsonl)

기존 판독이 보류였던 프레임에만 새 경보를 추가하면 Silverstone 좌 5/11, 우 1/13, 결합 5/24, 안정 예고 2/7, 오경보 9건이었다. 보류는 873→860이었다. 기존 연속 적중 두 쌍은 그대로였고 새 우회전 적중은 한 프레임이었다. Spa의 보류 복구는 좌 3, 우 1, 결합 2, 오경보 19였다. Spa 보류 복구의 기준 행은 저장한 경계 캐시와 새 원본 재생의 상세 행에서 재구성했고, 기준 채점의 적중·오경보 집계가 원본 재생과 일치했다. 두 사후 비교에서 우 적중을 일부 복구했으나 새 오경보가 늘어 배포 후보로 채택하지 않았다. [보류 복구 채점](../runs/vision-corner-image-normal-20261112-01/fallback-result.json) · [추가 경보](../runs/vision-corner-image-normal-20261112-01/fallback-alerts.jsonl)
