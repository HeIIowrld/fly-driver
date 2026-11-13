# T4/T5를 이용한 추가 코너 경보 확인 · 2026-11-12

경계 판독이 보류했을 때 이미지 기울기로 복구한 동일한 후보만 비교했다. 다섯 개발 서킷의 영상으로 굽는 구간을 학습했고, Spa와 Silverstone은 학습에서 제외했다. 경계가 제안한 시작 거리의 2m 앞부터 4m 뒤까지 굽음의 평균이 제안 방향과 일치하고 정규화 점수가 0.25 이상이면 경보를 받는 조건을 사용했다. 동일 학습 방식과 문턱에 경계만 넣은 모델과 경계에 세 프레임 T4/T5를 더한 모델을 비교했다. 두 서킷을 이미 본 뒤의 개발 진단이다. [확인 코드](../runs/vision-corner-motion-confirmation-20261112-01/confirm.py) · [경계만 쓰는 대조군](../runs/vision-corner-motion-confirmation-20261112-01/paired.py) · [짝지은 판정](../runs/vision-corner-motion-confirmation-20261112-01/paired-result.json)

| 서킷·경보 확인 | 좌 적중 | 우 적중 | 방향·거리 동시 적중 | 직선·범위 밖 오경보 |
| --- | ---: | ---: | ---: | ---: |
| Silverstone 기존 경계 | 5 | 0 | 4 | 3 |
| Silverstone 복구 경보 전부 | 5 | 1 | 5 | 9 |
| Silverstone 경계 모델로 확인 | 5 | 0 | 4 | 4 |
| Silverstone 경계+T4/T5로 확인 | 5 | 1 | 5 | 5 |
| Spa 기존 경계 | 2 | 0 | 0 | 3 |
| Spa 복구 경보 전부 | 3 | 1 | 2 | 19 |
| Spa 경계 모델로 확인 | 3 | 0 | 1 | 7 |
| Spa 경계+T4/T5로 확인 | 3 | 1 | 2 | 8 |

경계 모델과 짝지은 비교에서 T4/T5는 두 서킷 각각에서 우회전 누락 한 건을 방향과 거리 동시 적중으로 바꿨다. Silverstone은 직선 오경보 한 건을 없애고 새 오경보 두 건을 냈다. Spa는 오경보 두 건과 접근 중 반대 방향 경보 한 건을 없애고 새 오경보 세 건을 냈다. 새 오경보를 포함하면 두 서킷의 총량은 각각 한 건 늘었다. [프레임별 오류 변화](../runs/vision-corner-motion-confirmation-20261112-01/paired-confirmation.jsonl)

Silverstone에서 T4/T5 단독 곡률의 시작 제안은 20m였다. 영상 경계가 제안한 8m 주변의 굽음 방향은 지지했고 실제 시작은 6.02m였다. 두 입력의 시작 거리가 5m 이내로 일치해야 한다고 요구하면 이 우회전을 놓쳤다. Spa에서도 시작 거리까지 서로 일치해야 한다는 조건에서는 새 적중을 얻지 못했다. [신호와 시작점](../runs/vision-corner-motion-confirmation-20261112-01/proposals.jsonl) · [전체 확인 조건](../runs/vision-corner-motion-confirmation-20261112-01/result.json)

추가로 복구한 단면이 시작점 전후에 각각 두 개 이상 있어야 하는 규칙을 Silverstone 사례를 본 뒤 세웠다. Silverstone에서는 우 적중 한 건을 유지하며 추가 직선·범위 밖 오경보 6건을 걸렀다. Spa에서는 우 적중 한 건을 유지했지만 기존 대비 오경보가 3→10건이었다. 이 두 수치는 규칙을 만들 때 결과를 본 서킷의 사후 진단이다. [단면 수 사후 진단](../runs/vision-corner-motion-confirmation-20261112-01/bracketed-recovery-diagnostic.json)

같은 단면 수 규칙을 고정하고 다른 다섯 개발 서킷의 6,087묶음을 원본에서 다시 읽었다. 합산 우 적중은 13→15, 방향·거리 동시 적중은 5→6이었다. 직선·범위 밖 오경보는 기준 59건, 복구 경보 전부 108건, 단면 수 확인 90건이었다. Silverstone에서 오경보 증가가 없었던 결과가 다른 서킷에서 반복되지 않아 채택하지 않았다. [개발 서킷 결과](../runs/vision-corner-motion-confirmation-20261112-01/development-bracket.json) · [개발 프레임별 제안](../runs/vision-corner-motion-confirmation-20261112-01/development-bracket-proposals.jsonl)
