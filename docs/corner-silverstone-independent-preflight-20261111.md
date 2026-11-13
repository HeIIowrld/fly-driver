# Silverstone 1967 독립 평가 준비와 촬영 간격 · 2026-11-11

Silverstone 1967 설치의 여섯 모델 파일과 AI 경로 해시를 확인했다. 설치 기하의 곡률 구간은 좌 3곳과 우 6곳이다. 실제 연속 접근 표본은 촬영 후 확인한다. 기존 저장본의 아카이브 목록에서 Silverstone 1967을 찾지 못했고, 이번 준비에서도 이 서킷 영상을 열거나 촬영하지 않았다. [설치 및 타이밍 검사](../runs/vision-corner-dense-capture-preflight-20261111-01/result.json)

## 요청 간격과 실측

Spa의 기존 1,200묶음에서 렌더 완료 사이 간격은 중앙값 120.02ms, 95백분위 121.29ms였다. 저장 시각 기준 180ms를 넘긴 간격은 네 번이었다. 동일 원본의 연속 채점 가능 코너는 좌 8/9, 우 16/17이었다. [타이밍과 접근 표본](../runs/vision-corner-dense-capture-preflight-20261111-01/result.json)

현재 설치된 CSP는 몸 상태를 함께 보존하는 네 방향 외부 카메라 녹화에 90ms 이상의 요청 간격과 정차·브레이크·일시정지·AI 꺼짐을 요구한다. Spa의 기존 AI 주행 상태에서 시도한 90ms 요청은 카메라가 시작되기 전에 종료됐다. 저장된 그룹은 0개였고, 종료 영수증에서 카메라 작업이 끝났으며 조작 메시지는 0개였다. 실제 90ms 저장 간격은 확인되지 않았다. [실패 및 종료 영수증](../runs/vision-corner-dense-capture-preflight-20261111-01/spa-90ms/result.json)

## 독립 평가에 고정한 설정

- Silverstone 1967, BMW Z4 GT3, 오프라인, 주행선 표시 끔, 전·후·좌·우 영상과 몸 상태를 1,200묶음 수집한다. 요청 간격은 90ms다. [촬영 설정](../runs/vision-silverstone1967-independent-protocol-20261111-01/capture.json)
- Spa에서 고정했던 4–20m 접근, ±5m, 180ms 이내 연속 두 프레임과 기존 합격 기준을 그대로 적용한다. 실제 표본의 물리 코너 좌우 각각 두 곳 이상에서 연속 채점이 가능해야 한다. 미출력과 보류도 전체 캡처의 행으로 남긴다. [채점 설정](../runs/vision-silverstone1967-independent-protocol-20261111-01/base.json) · [입고 기준](../runs/vision-silverstone1967-independent-protocol-20261111-01/admission-review.json)
- 기존 시각 판독 비교를 주 평가로 둔다. 이동으로 갱신한 거리에서 새 영상 거리 쪽으로 최대 2m 수정한 설정을 별도의 실험 조건으로 미리 고정했다. 이 거리 갱신은 다른 개발 서킷에서 일관된 개선을 얻지 못했다. [고정 영수증](../runs/vision-silverstone1967-independent-protocol-20261111-01/lock.json) · [평가 실행기](../runs/vision-silverstone1967-independent-protocol-20261111-01/evaluate_silverstone.py)

고정 해시와 시각 판독·채점 기준을 검증했다. Spa 원본을 Silverstone 평가에 넣으면 입고 단계에서 거절된다. 실행 중인 엔진의 현재 상태는 Spa 내장 AI 주행으로, Silverstone 사전 검사에서 서킷·정차·일시정지 조건이 충족되지 않았다. Silverstone에서 정차·브레이크·일시정지 상태를 확보하면 카메라를 열고 첫 그룹 저장을 확인한 뒤 주행을 시작한다. [검증](../runs/vision-silverstone1967-independent-protocol-20261111-01/checks.log) · [현재 사전 검사](../runs/vision-corner-dense-capture-preflight-20261111-01/silverstone-preflight/result.json)


## 주 평가 실행 환경

기존 Windows ROCm Python, 저장된 FlyVis 소스, Datamate 호환 패키지를 연결했다. Radeon RX 9070 XT를 확인했고 주 평가 실행기의 고정 해시 검사를 통과했다. 프로젝트 폴더에서 다음 명령에 새 캡처와 결과 경로를 지정한다. 추가 결합 후보는 주 평가의 입고와 채점을 마친 뒤 실행한다. [주 평가 실행기](../runs/vision-silverstone1967-independent-protocol-20261111-01/run-evaluation.ps1) · [추가 비교](corner-silverstone-combined-exploratory-20261111.md)

```powershell
& .\runs\vision-silverstone1967-independent-protocol-20261111-01\run-evaluation.ps1 -CaptureDir $capture -Output $primary
```

## 촬영 시작 위치와 정차 프레임의 영향

고정한 Silverstone 기하에서 좌 3곳과 우 5곳의 채점 접근 구간은 각각 16m였다. 우 ID 6은 앞 코너와 가까워 4–20m 접근으로 등록되는 구간이 약 1m였다. 이 코너에서는 90ms 요청에서도 보통 속도로 두 프레임을 연속 확보하기 어렵다. [설치 기하와 구간](../runs/vision-silverstone1967-capture-coverage-20261111-01/result.json)

시작 위치를 서킷 전체의 720개 지점으로 바꾸고, 고정 요청 간격 90ms·1,200묶음에서 카메라 첫 렌더와 마지막 렌더가 같은 접근 구간에 들어오는지 계산했다. 일정 속도·일정 간격·스플라인 중심 주행을 가정한 모의 계산이다.

| 가정한 이동 속도 | 정차 0묶음 | 정차 100묶음 |
| --- | ---: | ---: |
| 108km/h | 594/720 | 553/720 |
| 144km/h | 720/720 | 702/720 |
| 165.4km/h, Spa 실측 중앙값을 일정 속도로 적용 | 720/720 | 720/720 |

값은 좌우 각각 두 물리 코너 이상에서 연속 채점이 가능한 시작 위치 수다. 90ms는 같은 묶음 수에서 120ms보다 한 접근 구간을 더 자주 관측할 수 있고 주행을 담는 시간은 36초 짧다. 촬영 준비 영수증을 확인한 뒤 주행을 시작하면 정차 중 소모하는 묶음을 줄일 수 있다. 실제 입고에서는 네 카메라 기록, 원본 시간 간격, 좌우별 표본을 다시 검사한다. [48개 모의 조건](../runs/vision-silverstone1967-capture-coverage-20261111-01/result.json) · [계산 코드](../runs/vision-silverstone1967-capture-coverage-20261111-01/coverage.py)

## 녹화 실행기

고정 영수증에 기록한 버퍼 녹화기와 설정 파일의 SHA-256을 확인한 뒤 녹화기를 실행한다. 프로젝트 폴더에서 첫 명령의 새 출력 경로로 실제 세션 상태를 검사한다. 두 번째 명령은 준비된 세션에서 별도의 새 경로로 녹화한다. 출력 폴더의 `ready.json`은 새 네 방향 카메라 묶음을 확인한 뒤 생성된다. 이 신호 뒤 주행을 시작한다.

```powershell
& .\runs\vision-silverstone1967-independent-protocol-20261111-01\run-capture.ps1 -Preflight -Output $preflight
& .\runs\vision-silverstone1967-independent-protocol-20261111-01\run-capture.ps1 -Output $capture
```

실행기 검증에서 고정 설정과 설치한 CSP 소스가 일치했다. 실시간 상태의 서킷은 Spa였고 AI 운전 중이어서 준비 검사에서 멈췄다. 이 검사에서 카메라 요청이나 차량 조작을 보내지 않았다. [녹화 실행기](../runs/vision-silverstone1967-independent-protocol-20261111-01/run-capture.ps1) · [실시간 준비 검사](../runs/vision-silverstone1967-capture-launcher-preflight-20261111-01/result.json)

## 2026-11-12 촬영 및 평가

촬영과 원본 입고를 마쳤고, 주 평가와 추가 결합 후보를 모두 채점했다. 접근 좌 11·우 13프레임 모두 직선으로 판독됐다. [독립 평가 결과](corner-silverstone-independent-20261112.md)
