# Silverstone 1967 결합 판독 추가 비교 · 2026-11-11

Silverstone 영상을 받기 전에 도로 경계·운동 시각과 몸 감각을 결합한 후보의 가중치와 문턱을 고정했다. 학습에는 Magione, Laguna Seca, Zandvoort, Mugello, Imola만 사용했다. 학습에서 제외한 Spa의 사후 결과를 보고 경보 문턱 0.10과 이동 기반 거리 보정의 최대 2m 조건을 추가 비교 대상으로 선택했다. 개발 서킷 제외 검증에서는 채택 조건을 충족한 문턱이 없었다. Silverstone의 기존 주 평가와 합격 기준은 기존 고정 영수증을 따른다. [추가 후보 고정 기록](../runs/vision-silverstone1967-combined-exploratory-20261111-01/receipt.json) · [주 평가](corner-silverstone-independent-preflight-20261111.md)

Spa 원본 1,200묶음에서 도로 경계를 다시 추출해 후보를 재생했다. 방향 적중은 좌 5/22, 우 19/49였고, 방향과 거리 ±5m 동시 적중은 18/71이었다. 직선·범위 밖 오경보는 80/605, 미출력은 0이었다. 공통 접근 6쌍의 거리 변화 오차는 3.09m로 기준을 통과했다. 안정 예고는 연속 채점이 가능한 24곳 중 7곳으로 사전 기준을 충족하지 못했다. [Spa 재생 결과](../runs/vision-silverstone1967-combined-exploratory-20261111-01/spa-rehearsal/result.json)

추가 후보 실행기는 주 평가에서 입고를 마친 Silverstone 캡처와 주 평가의 고정 해시를 대조한 뒤 실행한다. Spa 결과를 입력으로 주면 거절하는 것을 확인했다. 프로젝트 폴더에서 아래 명령의 세 변수에 각각 캡처, 완료한 주 평가, 새 결과 경로를 지정한다. Silverstone 실측 영상과 독립 점수는 아직 없다. [실행기](../runs/vision-silverstone1967-combined-exploratory-20261111-01/run.ps1) · [잘못된 서킷 거절 기록](../runs/vision-silverstone1967-combined-exploratory-20261111-01/admission-check.log)

```powershell
& .\runs\vision-silverstone1967-combined-exploratory-20261111-01\run.ps1 -CaptureDir $capture -PrimaryOutput $primary -Output $exploratory
```

## Spa에서 안정 예고가 끊긴 원인

공통 채점기의 180ms 이내 연속 프레임 조건으로 좌 8곳과 우 16곳을 다시 나눴다. 안정 예고를 놓친 17곳 중 13곳에서는 연속 채점에 쓰이는 어느 프레임에서도 방향을 맞히지 못했고, 3곳에서는 한 프레임씩만 맞혔다. 두 프레임 연속으로 방향이 맞았으나 거리 오차로 탈락한 곳은 우회전 한 곳이었다. 좌회전은 2곳이 안정 예고에 성공했고, 나머지 6곳은 연속 채점 프레임에서 방향 적중이 없었다. [코너와 해당 프레임](../runs/vision-silverstone1967-combined-exploratory-20261111-01/spa-rehearsal/stability-diagnosis.json)
