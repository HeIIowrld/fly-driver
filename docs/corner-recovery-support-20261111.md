# 경보 복구의 영상 거리·이동 증거 · 2026-11-11

Magione, Laguna Seca, Zandvoort, Mugello, Imola의 저장 영상 6,087묶음에서 고정 후보의 경보 복구를 다시 계산했다. 경보 유무·방향·프레임 ID를 보존 결과와 대조했고, 일부 실행의 거리·점수 반올림 차이에는 절대 오차 10⁻¹⁰만 허용했다. 복구 여부를 바꾼 프레임은 복구 직전의 출력으로 되돌린 뒤 기존 공통 채점기로 비교했다. [감사 코드](../runs/vision-corner-recovery-support-20261111-01/audit_restores.py) · [수치 대조 코드](../runs/vision-corner-recovery-support-20261111-01/audit_restores_tolerance.py) · [전체 결과](../runs/vision-corner-recovery-support-20261111-01/result.json)

| 서킷 | 복구 경보 | 맞은 방향 접근 | 직선·범위 밖 | 기타 |
| --- | ---: | ---: | ---: | ---: |
| Magione | 2 | 0 | 0 | 2 |
| Laguna Seca | 12 | 3 | 6 | 3 |
| Zandvoort | 29 | 3 | 12 | 14 |
| Mugello | 0 | 0 | 0 | 0 |
| Imola | 6 | 0 | 6 | 0 |

다섯 서킷에서 복구된 49개 중 방향을 맞힌 접근은 6개였다. Laguna의 좌 3개, Zandvoort의 좌 1개·우 2개다. 이 중 5개는 방향과 거리 ±5m를 함께 맞혔다. 직선·범위 밖은 24개였고, 나머지 19개는 정지 또는 코너 안쪽이었다. [Laguna 프레임](../runs/vision-corner-recovery-support-20261111-01/ks_laguna_seca/restored.jsonl) · [Zandvoort 프레임](../runs/vision-corner-recovery-support-20261111-01/ks_zandvoort/restored.jsonl) · [Imola 프레임](../runs/vision-corner-recovery-support-20261111-01/imola/restored.jsonl)

복구할 거리와 별도로 학습한 영상 거리 추정, 직전 경보를 상대 이동으로 옮겼을 때의 거리, 영상 경계 방향을 대조했다. 한 서킷씩 제외하고 나머지 네 곳에서 좌·우·결합 적중과 안정 예고 횟수를 지킨 조건 중 오경보가 적은 규칙을 골랐다.

- Imola를 제외한 비교에서 고른 6m 거리 일치 조건은 Imola 오경보를 45→44개로 줄였고 적중·안정 예고 횟수는 같았다.
- Zandvoort를 제외하고 고른 4m 조건은 Zandvoort 우회전 적중을 10→9개로 줄였고, 오경보는 61개 그대로였다.
- 다섯 서킷을 모두 본 뒤 6m를 공통으로 적용한 개발 비교에서는 적중·안정 예고 횟수가 유지됐고 Imola 오경보 한 건이 줄었다. 고정 판독기에 반영하려면 별도 자료의 검증이 필요하다. [규칙별 채점 및 선택](../runs/vision-corner-recovery-support-20261111-01/result.json) · [선택 규칙의 변경 프레임](../runs/vision-corner-recovery-support-20261111-01/removed-by-selection.jsonl)

Spa에서 비교할 전체 후보와 혼합 단계는 이전에 고정한 설정을 유지한다. Windows 연결 해제 상태가 계속돼 이번에도 녹화는 시작되지 않았다. [Spa 독립 평가 설정](corner-spa-frozen-20261111.md)
