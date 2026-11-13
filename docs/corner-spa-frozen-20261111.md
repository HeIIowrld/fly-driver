# Spa 독립 평가 사전 고정 · 2026-11-11

Spa에서 사용 가능한 설치 모델과 `fast_lane.ai` 해시를 확인했다. 기하에서 집계한 좌·우 곡률 구간은 12·22개다. 새 Spa 영상을 받기 전에 기존 고정 판독기, 경계·거리·복구 후보, 카메라 설정과 공통 채점 기준을 고정했다. 주요 비교는 전체 후보와 기존 판독의 대조다. 혼합 판독 단계의 결과도 같은 캡처와 기준기로 별도 채점하도록 기록했다. [후보 설정](../runs/vision-spa-independent-protocol-20261111-01/candidate.json) · [녹화 설정](../runs/vision-spa-independent-protocol-20261111-01/capture.json) · [입고 및 비교 계획](../runs/vision-spa-independent-protocol-20261111-01/admission-review.json)

저장 과정의 1,200그룹과 네 방향 영상 및 시간 기록을 모두 검증한 뒤, 좌우 각각 두 곳 이상에서 두 프레임 연속 채점이 가능한지 확인한다. 같은 입고 검사기를 Imola 완성 원본에 적용해 기존 입고 프레임의 해시를 재현했다. Imola 원본을 Spa 설정으로 넣으면 거절되는 것도 확인했다. [입고 검사기](../scripts/audit_corner_capture_admission.py) · [Imola 검증 결과](../runs/vision-imola-approach-capture-20261111-02/reusable-admission/result.json) · [Spa 평가 실행기](../runs/vision-spa-independent-protocol-20261111-01/evaluate_spa.py)

Windows 사용자 세션이 연결 해제된 상태였다. Computer Use에서 AC 창 접근을 재확인했으나 `GetCursorPos failed: 0x80070005`가 반복됐다. 게임의 마지막 CSP 상태는 Imola 일시정지, AI 운전 꺼짐, 카메라 비활성이었다. 이번 확인 시점까지 Spa 녹화와 채점은 시작되지 않았다.

## Imola의 개선이 나온 단계

| 단계 | 좌·우·결합 적중 | 직선·범위 밖 오경보 | 안정 예고 접근 횟수 |
| --- | ---: | ---: | ---: |
| 기존 판독 | 13·2·11 | 46 | 3 |
| 혼합 판독 | 14·5·15 | 39 | 4 |
| 경계·거리·복구 포함 | 14·5·15 | 45 | 5 |

Imola의 추가 방향·결합 적중은 혼합 판독 단계에서 이미 나타났다. 이후 복구한 경보 6개는 모두 범위 밖이었다. 그중 다섯 개는 같은 좌 코너까지 104.9–117.2m 남은 프레임이었다. 경계·거리·복구를 포함한 단계는 혼합 단계 대비 공통 거리 변화 오차를 3.30→3.08m로 줄였고 안정 예고 접근을 한 번 더 만들었다. 다른 개발 자료에서는 전체 단계가 Magione·Laguna Seca·Zandvoort의 방향·거리 동시 적중을 각각 20·3·4개 늘렸다. 두 설정을 함께 비교하기로 한 이유다. [Imola 변경 프레임](../runs/vision-imola-approach-diagnostic-20261111-01/component-rows.jsonl) · [다섯 서킷 단계별 채점](../runs/vision-spa-independent-protocol-20261111-01/development-stages.json)


### 독립 녹화 재개 조건

Windows 사용자 세션이 계속 연결 해제 상태다. 현재 배포된 CSP 소스의 해시는 사전 고정 설정과 일치한다. 이 소스의 서킷 전환 API가 허용하는 경로는 Monza↔Magione과 Magione 재실행이다. Imola에서 Spa로 바꾸는 검증된 API 경로가 없다. 기존 원본 인벤토리와 이후 녹화 경로에서 Spa 또는 Silverstone 1967 영상도 확인되지 않았다.

Windows에 다시 연결한 뒤 Spa 오프라인 핫랩을 열고 사전 고정한 녹화·입고·채점을 실행한다. 재확인 시점의 Spa 후보와 녹화 설정 해시는 고정 기록과 일치했다.


## 평가 완료

Windows 재연결 후 Spa 네 방향 영상 1,200묶음을 수집하고 고정 비교를 완료했다. 두 후보의 사전 기준 미충족 결과를 별도 기록했다. [Spa 독립 평가](corner-spa-independent-20261111.md)
