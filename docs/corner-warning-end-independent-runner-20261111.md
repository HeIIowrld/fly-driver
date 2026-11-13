# 경보 종료를 포함한 독립 평가 실행기 · 2026-11-11

Magione 개발 비교에서 선택한 경보 종료 조건을 기존 결합 실행기 뒤에 연결했다. 현재 예고 거리가 6m 이하일 때만, 첫 예고 거리에서 연속 캡처의 로컬 이동을 빼고 1m 여유를 더한 값이 4m 미만이면 보류한다. 시계·세대·프레임이 끊기거나 몸 감각이 없거나 방향이 바뀌면 누적을 초기화한다. 이 단계에는 지도와 정답 라벨을 전달하지 않는다. 기존 실행기와 프로토콜은 별도 실행으로 보존했다.

Magione 원본 전체 1,887개에 적용한 결과 두 경보가 보류됐다. 한 프레임은 4–20m 실제 접근, 한 프레임은 범위 밖이었다. 원래 판독 대비 좌 3/64→13/64, 우 62/79 유지, 방향·거리 ±5m 적중 45→55개, 직선·범위 밖 오경보 27개 유지, 미출력 0개였다. 공통 62쌍의 거리 변화 오차는 0.399→0.391m, 실제 접근 중 역행은 22회로 같았다. 기존 개발 검증의 사전 기준을 모두 통과했다. [Magione 전체 재현](../runs/vision-corner-warning-end-runner-20261111-01/magione/result.json) · [종료된 경보](../runs/vision-corner-warning-end-runner-20261111-01/magione/ended-warnings.jsonl)

## Zandvoort 촬영과 채점

Zandvoort에 동일한 판독과 원래 독립 합격 기준을 고정했다. 실제 영상이 없는 입력은 자료 자격 단계에서 거부됐다. 촬영 준비 검사에서는 BMW Z4 GT3 오프라인·정차·브레이크·일시정지, 주행선과 AI 꺼짐, CSP 소스 일치가 확인됐다. 현재 트랙은 Laguna Seca라서 Zandvoort 촬영을 시작하지 않았다. [촬영 준비 검사](../runs/vision-corner-warning-end-runner-20261111-01/zandvoort-preflight/result.json) · [독립 설정](../runs/vision-corner-warning-end-runner-20261111-01/zandvoort-protocol.json)

새 세션이 준비되면 `scripts/capture_independent_corner.py`에 `zandvoort-capture-protocol.json`과 새 실행 폴더를 지정한다. 이 명령은 네 카메라 원본을 최대 1,000묶음까지 기록하며 차량을 움직이지 않는다. 촬영이 끝나면 생성된 `archive-map.json`을 `scripts/evaluate_fused_corner_warning_end.py`의 입력으로 사용한다. 신규 서킷의 실제 이동 영상, 양방향 물리 코너 입고와 독립 합격은 아직 확인하지 않았다.
