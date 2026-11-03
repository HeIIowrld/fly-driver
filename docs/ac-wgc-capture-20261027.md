# AC 창 전용 백그라운드 영상 수집 시도

갱신: 2026-10-27, Asia/Seoul. 목적은 사용자 작업을 방해하는 창 활성화 없이 게임 영상을
받아, 실제 AC의 비전/몸 감각 기록을 만드는 것이다. 이전 Esc 중지 뒤 앱 입력은 재개하지 않았다.

기존 MSS 경로는 화면에 보이는 client 영역을 읽으므로 전면 창 검사를 유지한다.
별도 선택 경로로 [windows-capture2.0.1의 명시적 HWND API](https://github.com/NiiightmareXD/windows-capture/blob/main/windows-capture-python/windows_capture/__init__.py)를
사용하는 Windows Graphics Capture 수집기를 추가했다. 설치한 버전의 실제 생성자·프레임·
스레드 종료 API를 검사했다. 의존성은 `windows-capture==2.0.1`·`opencv-python==5.0.0.93`이고,
선택 설치는 `pip install -e .[wgc]`다. 현재 Windows Python3.12.10·numpy2.5.4 환경을 사용했다.

## 입력·기록 경계

- AC 엔진의 HWND·PID·실행 파일·창 geometry를 읽고, 정확히 그 HWND만 넘긴다.
  `window_name` 부분 일치나 monitor 캡처로 대체하지 않는다.
- 마우스·키보드 입력·창 이동/포커스 변경·보안 설정 변경은 하지 않는다.
- 영상은 전체 창 RGB다. client/camera crop·FOV·미러 반사 보정은 아직 하지 않는다.
- native callback 안에서 BGRA를 복사해 수명이 짧은 native buffer를 나중에 읽지 않는다.
- 창 생산자/geometry 변경, capture native time 중복/역행은 거절한다. 유한 수신 기한·
  프레임 수·저장 용량을 적용하고 최신 값 대기열은1개로 제한한다.
- 물리 매핑의 전/후 표본과 영상 callback 수신 구간을 같은 호스트 시계로 묶는다.
  이는 host receipt bracket일 뿐 엔진 노출·시뮬레이션 시각의 동기화가 아니다.
- API의 native timespan은 원시 정수로 보존한다. 단위·엔진 frame ID·실제 렌더 freshness·
  가림 동작은 검증 전이며 actor 스트림을 만들지 않는다.

## 초기 실패와 전면·비활성 조건 비교

`runs/ac-wgc-20261027-t30/`와 `runs/ac-wgc-20261027-t30b/`에 실패를 보존했다.
두 번째는 물리 표본을 먼저 읽고 capture 세션을 시작해 초기 수신 순서 문제를 배제하도록
바꿨으나,3초 동안 callback0·영상0으로 시간 초과였다. capture worker는 종료되지 않은 상태였고,
창은 `acs.exe` PID42060·HWND1379464·client1280×720으로 확인됐다. 다른 창의 픽셀은 받지 않았다.

사용자가 직접 AC를 앞에 둔 뒤 `runs/ac-wgc-foreground-20261027-t30c/`에서 실제 영상5장을
받았다. 이후 창이 비활성인 상태의 `runs/ac-wgc-background-after-foreground-20261027-t30d/`도
3장을 받았다. 전면 여부는 각 수신 시점에 조회했으며 각각 모두 true / 모두 false였다.
두 실행 모두 증가하는 native capture 시각, 서로 다른 RGB 해시, 영상 전후 물리 packet ID를
확인했다. NPY 파일·RGB 내용·물리 원시 prefix의 SHA-256도8장 모두 다시 대조했다.

전면 영상은 실제 AC 콕핏과 연습 메뉴 overlay임을 눈으로 확인했다. 전체 창1602×940 픽셀과
논리 client1280×720은 DPI와 창 테두리 때문에 다르며, 카메라 crop이나 거울 보정 결과가 아니다.
HUD·메뉴를 actor 입력으로 전달하지 않는다. 비활성 표본3장은 백그라운드 수신의 성공 사례지만,
장시간 연속성·최소화·실제 주행 중 렌더 속도·노출 시각 동기는 검증하지 않았다.
초기 실패의 원인은 확정하지 않았으며, 전면에 한 번 둔 뒤의 성공을 재현 조건으로 기록한다.

마우스·키보드·포커스 조작과 액추에이터 전송은0건이다. 현재 결과는
`configs/ac-wgc-result.json`에 기록했다. 다음은 비활성 연속 수신율과 camera crop·미러 투영·
센서 시계 보정이다. 실제 주행 학습용 동기 actor 스트림은 아직 만들지 않았다.

### 비활성 연속 수신 시험

후속 `runs/ac-wgc-cadence-20261027-t31/`에서 요청10Hz·100장을 측정했다.
100장 모두 수신 시점에 비활성이었고,10.813초 동안 실제 수신율은9.156Hz였다.
수신 간격 중앙값·95백분위·최댓값은 모두0.125초다. capture callback은175회였고,
물리 표본의 호스트 수신 하한보다 오래된 프레임74개는 버려100개를 반환했다.
RGB 해시는100개 모두 달랐으며 native capture 시각도 증가했다. 프레임마다 물리 원시
prefix와 호스트 수신 구간을 보존하고, 저장량을 줄이기 위해 RGB 원본은 처음·마지막만 저장했다.
100쌍의 물리 원시 해시와 시간 순서, 두 끝점 이미지 해시를 다시 검증했다.

마지막 이미지를 눈으로 확인한 결과 실제 Z4 GT3의 연습 메뉴·정차 콕핏이었다.
이 결과는 해당 상태의 약11초 연속 수신을 뒷받침한다. 목표25Hz, 장시간 안정성,
최소화 또는 실제 주행 중 영상 갱신을 검증한 결과는 아니다. RGB 해시 변화만으로
도로 장면이 움직였거나 올바른 노출 시각에 대응한다고 판단하지 않는다.

같은 시험은 다음 명령으로 실행한다. 각 실행은 새 출력 경로를 사용한다.

```powershell
python -m scripts.probe_ac_wgc_cadence --frames 100 --rate 10 --output runs/new-ac-cadence
```

또한 수신 대기열에 오래된 프레임이 계속 들어와도 유한 수신 기한을 지키도록 수정했다.
이는 데이터 누락을 숨기지 않기 위한 경계 수정이며 게임 조작을 추가하지 않는다.

### 목표25Hz 요청과 게임 시간 동시 기록

`runs/ac-wgc-cadence-25hz-20261027-t33b/`에서25Hz를 요청해100장을 받았다.
모두 비활성이었고6.25초 구간의 실제 수신율은15.84Hz였다. 수신 간격 중앙값·95백분위는
약0.063초, 최댓값은0.125초였으며 RGB 해시는 모두 달랐다. 앞의10Hz 요청 실험과 요청률이
다르므로9.16→15.84를 같은 조건의 최적화 효과라고 해석하지 않는다. 아직25Hz 수신을
달성하지 못했으며 실제 주행 중 성능과 병목 원인은 미확정이다.

`record_ac_wgc --graphics-clock` 선택 경로를 추가했다. 각 영상의 physics 전/후보다 바깥에서
native graphics 시간을 읽고 원시 prefix·해시·호스트 수신 시각을 함께 저장한다.
`runs/ac-eye-body-clock-20261027-t33/`에서 비활성 영상20장·native prefix80개를 받았다.
원시 prefix를 다시 decode해 기록 필드와 비교하고, 영상 해시·단조 증가 sequence·호스트
수신 순서를 검사했다. 저장·검사한20장 모두 일치했다. 그룹 전체 호스트 구간은 중앙값
16ms·최대141ms였으며, 이 값은 수집 대기를 포함하고 노출 동기 오차나 지연의 상한이 아니다.

이 구간의 네 타이어 원시 온도는 모두 일정했다. 하중 원시값은 변했지만 정차 진동을
하중 이동 학습 사례로 취급하지 않는다. HUD·메뉴·FOV·미러 보정과 온도/하중 단위·축·노출 시계가
남아 있으므로 기록을 학습용 actor 데이터로 승인하지 않았다.

```powershell
python -m scripts.record_ac_wgc --frames 20 --rate 5 --graphics-clock --output runs/new-eye-body-clock
python -m scripts.audit_ac_eye_body_recording runs/new-eye-body-clock --output runs/new-eye-body-clock/audit.json
```

현재 정책의25Hz 제안과 실제 카메라 수신 주기는 분리해서 검토한다. 새 영상이 없으면
source별 age·validity로 표시하며 복제 프레임을 새 노출로 올리지 않는다. 최종 주행에 필요한
지연·누락 허용치는 실제 주행 기록과 폐루프 시험으로 정한다.

정확한 창 지정·BGRA 복사/수명·누락 기한·생산자 변경·시각 역행의 대역 검사3개와 기존 캡처
검사13개는 Windows에서 통과했다. 다음 읽기 전용 검사 예시는 다음과 같다.

```powershell
$env:PYTHONPATH='src'
python -m scripts.record_ac_wgc --frames 5 --rate 5 --output runs/new-ac-wgc-probe
```
