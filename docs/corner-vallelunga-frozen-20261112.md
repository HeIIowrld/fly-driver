# Vallelunga Extended 독립 코너 평가 준비 · 2026-11-12

Vallelunga Extended를 새 서킷으로 선택했다. 레이아웃 설정·모델 파일·AI 경로 해시를 기록했다. 경계 패치 모델과 경계 확인 모델은 기존 다섯 개발 서킷의 자료로 고정했다. Vallelunga, Silverstone 1967, Spa는 학습에 사용하지 않았다. 국소 곡률 문턱은 0.002m⁻¹, 경계 패치 문턱은 0.55, 두 확인 모델의 공통 문턱은 0.25다. 이전 Silverstone 사후 문턱 0.33도 확인했으나 이번 평가에는 기존 0.25 조건을 고정했다. [설정과 모델 해시](../runs/vision-vallelunga-local-corner-protocol-20261112-01/lock.json) · [설치와 공통 기준](../runs/vision-vallelunga-local-corner-protocol-20261112-01/base.json)

두 확인 모델은 같은 영상과 같은 경계 후보를 사용하며, 결합 모델에만 세 프레임 T4/T5 응답을 넣는다. 둘 다 직접 확인된 경보가 다음 한 프레임에서 비면 차량의 이동만큼 거리를 갱신하고 현재 확인 점수를 다시 요구한다. 공통 평가에서 방향·거리 동시 적중은 실제 4–20m 접근에서 거리 오차가 5m 이내인 경우다. 안정 예고에는 180ms 이내의 연속 두 프레임 적중이 필요하다. 미출력과 보류도 전체 타임라인에 남긴다. 방향·거리·오경보와 안정 예고를 경계 기준 및 경계 확인 대조군과 각각 비교한다. 지도의 절대 위치는 촬영 입고와 평가에 쓰인다. [평가 실행기](../runs/vision-vallelunga-local-corner-protocol-20261112-01/evaluate.py)

기존 Silverstone 녹화 1,200묶음에서 새 실행기의 아홉 판독 조건이 기존 채점과 모두 일치했다. 이 재생은 새 서킷을 읽지 않는 호환성 검사다. Vallelunga 실행은 네 카메라의 실제 녹화 영수증과 레이아웃이 지정된 활성 레이스 설정의 전후 해시를 요구한다. CSP는 트랙 ID를 확인하며, 레이아웃은 녹화 전후의 활성 레이스 설정으로 확인한다. [Silverstone 재생 검사](../runs/vision-vallelunga-local-corner-protocol-20261112-01/silverstone-rehearsal/result.json) · [입고 기준](../runs/vision-vallelunga-local-corner-protocol-20261112-01/admission-review.json) · [레이아웃과 녹화 검증](../runs/vision-vallelunga-local-corner-protocol-20261112-01/capture_layout.py)

## 실행

프로젝트 디렉터리에서 아래 실행기를 사용한다. 촬영 시작 조건은 Vallelunga Extended, BMW Z4 GT3, 오프라인, 주행선 끔, AI 끔, 브레이크를 밟은 정차·일시정지다. 첫 사전 검사 당시에는 Silverstone 1967이 열려 있었다. [당시 CSP 사전 검사](../runs/vision-vallelunga-local-corner-protocol-20261112-01/engine-preflight/result.json)

```powershell
$runner = 'runs/vision-vallelunga-local-corner-protocol-20261112-01/run-evaluation.ps1'
$race = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Assetto Corsa/cfg/race.ini'
$capture = 'runs/vision-vallelunga-local-corner-capture-20261112-01'
$output = 'runs/vision-vallelunga-local-corner-evaluation-20261112-01'
& $runner -Mode capture -RaceIni $race -Output $capture
& $runner -Mode evaluate -RaceIni $race -CaptureDir $capture -Output $output
```

녹화기가 첫 버퍼를 열면 기존 CSP 수집 절차에 따라 주행을 시작한다. 실행기는 90ms 요청 간격으로 전후좌우 영상 1,200묶음을 받는다. 입고 검사를 통과한 뒤에만 원본과 반사본의 FlyVis 특징을 추출하고 채점한다.

GPU 추출 경로도 Silverstone의 621·622번과 필요한 앞 프레임을 다시 읽어 확인했다. 기존 특징과 새 특징 사이의 최대 절대 차이는 원본·반사본 모두 1.2×10⁻⁷이었다. 전체 1,200프레임 재생에서는 기존 특징을 사용했고, 경보와 경계 배열을 프레임별로 대조해 일치했다. [프레임별 재현 검사](../runs/vision-vallelunga-local-corner-protocol-20261112-01/parity.json) · [GPU 특징 재추출](../runs/vision-vallelunga-local-corner-protocol-20261112-01/gpu-feature-rehearsal.json)


## 저장 영상과 채점 입고

Vallelunga Extended에서 BMW Z4 GT3를 사용했다. 주행선 표시는 껐고 영상 수집 주행은 게임 내장 AI로 진행했다. 전후좌우 영상 1,200묶음이 저장됐고 녹화가 정상 종료됐다. 시작 4–20m의 접근 표본은 좌 23개와 우 38개였다. 표본이 있는 물리 코너 16곳 중 3곳은 연속 두 프레임이 부족했다. 안정 예고를 채점할 수 있는 코너는 좌 5곳과 우 8곳이다. [녹화 영수증](../runs/vision-vallelunga-local-corner-capture-20261112-01/result.json) · [입고와 코너별 표본](../runs/vision-vallelunga-local-corner-evaluation-20261112-05/admission/result.json)

녹화 실행기가 버퍼 프로필을 전달하지 않아 녹화에는 8슬롯 MMF 저장기가 쓰였다. 첫 채점 실행기는 32슬롯 영수증을 요구해 입고에서 멈췄다. 실제 저장기의 기존 검증기로 원본 1,200묶음의 네 영상과 메타데이터 해시, 저장 완료 응답을 재검사했다. 복구 실행기는 저장 검증기만 선택하며 촬영 전에 고정한 모델, 문턱 및 채점 기준을 그대로 호출한다. [첫 입고 실패](../runs/vision-vallelunga-local-corner-evaluation-20261112-01/result.json) · [원본 무결성 확인](../runs/vision-vallelunga-local-corner-storage-review-20261112-01/storage-verification.json) · [복구 실행기](../runs/vision-vallelunga-local-corner-storage-review-20261112-01/evaluate_recording.py)

## 독립 평가 결과

1,200묶음 전체를 채점했다. FlyVis 특징은 앞의 두 묶음을 시간 이력으로 사용해 1,198묶음에서 나왔다. 판독 보류와 미출력을 전체 분모에 포함했다. 안정 예고는 채점 가능한 물리 코너 13곳을 분모로 표시했다. 고정 합격 판정은 표본이 잡힌 물리 코너 전체를 요구하며, 채점 가능 13곳으로 좁혀도 결합 모델은 한 곳에서만 안정 예고했다. [고정 평가 결과](../runs/vision-vallelunga-local-corner-evaluation-20261112-05/result.json) · [1,200프레임 판독](../runs/vision-vallelunga-local-corner-evaluation-20261112-05/ks_vallelunga-frames.jsonl)

| 판독 방식 | 좌 방향 | 우 방향 | 방향·거리 동시 적중 | 직선·범위 밖 오경보 | 안정 예고 | 판독 보류 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 경계 기준선 | 2/23 | 0/38 | 2/61 | 6/425 | 1/13 | 653 |
| 경계 확인과 한 프레임 유지 | 4/23 | 1/38 | 5/61 | 34/425 | 2/13 | 569 |
| 경계+T4/T5 확인과 한 프레임 유지 | 5/23 | 2/38 | 5/61 | 45/425 | 1/13 | 482 |

세 방식 모두 미출력은 0이었다. 결합 모델은 경계 확인보다 좌·우 방향을 각각 한 프레임 더 맞혔지만, 동시 적중은 한 건을 얻고 한 건을 잃어 5건이었다. 안정 예고는 9번 좌회전에서 끊겼고, 13번 좌회전에서 시작 5.66m 전에 확인됐다. 두 비교에서 공유한 연속 방향 적중은 한 쌍뿐이며 거리 변화 오차는 2.45m로 같았다. 오경보, 안정 예고, 거리 변화 기준에서 두 비교 모두 불합격했다. 코너 내부를 별도로 보면 경보가 49→130건으로 늘었다.

우회전 38프레임 중 확인 점수를 적용하기 전 국소 경계 후보는 27프레임에서 경보를 내지 못했고 8프레임에서 좌회전을 제시했다. 3프레임에서만 우회전을 제시했고 이 가운데 2프레임이 거리까지 맞았다. 결합 확인은 경계 대조군의 잘못된 방향 경보 7건을 보류로 바꿨다. 우회전 후보가 없거나 방향이 틀리는 문제는 확인 점수 앞에서 이미 발생했다. 다음 개발 진단은 기존 개발 서킷에서 우회전 후보 누락과 반대 방향 후보를 분해하고, 방향과 시작 거리를 각각 대조하는 것이다. 이 Vallelunga 결과는 고정 설정의 독립 평가 기록으로 남긴다.

GPU 그래프 실행은 17번 프레임 부근에서 멈췄다. GPU에서 실행하는 FlyVis 원래 적분 함수로 바꿔 동일 영상의 정지 구간을 통과했다. Silverstone 621·622번에서 그래프 경로와 대조한 최대 특징 차이는 원본 8.94×10⁻⁸, 반사본 7.45×10⁻⁸이었다. 검증은 이 두 프레임에서 수행했다. 원래 적분 함수로 원본·반사 특징을 추출한 뒤 해시를 확인하고 고정 채점기를 실행했다. [실행 경로 대조](../runs/vision-vallelunga-local-corner-storage-review-20261112-01/eager-parity.json) · [채점 재실행](../runs/vision-vallelunga-local-corner-storage-review-20261112-01/evaluate_eager_resume.py)
