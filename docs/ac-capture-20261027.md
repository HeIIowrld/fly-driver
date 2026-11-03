# AC 영상 수집: 캡처 구간과 원본 RGB 기록

2026-10-27, Asia/Seoul. 직전 단계는 [읽기 전용 센서 연결](ac-bridge-20261027.md)이다.
이번 단계는 실제 게임 연결 전 필요한 영상 기록기를 추가했다. **실제 AC 영상·조작·완주는
검증하지 않았다.** 이전 목표 턴은 통신 구현·실행 증거를 남긴 진행으로 분류한다.

## 이번에 바뀐 것

`ac_capture.py`와 `scripts/record_ac_capture.py`는 선택한 원본 AC 프로세스의 창만 읽는다.
PID·실행 파일명 `acs.exe`/`acs_x86.exe`·표시/최소화 상태·foreground·client 영역을 확인한다.
다른 창이나 모니터 전체로 대체하지 않고 창을 이동하거나 focus를 바꾸지도 않는다.
캡처 전후 창 신원/크기가 바뀌면 이미지를 버린다. 창이 화면 밖으로 나간 경우도 거절한다.

원본 uint8 RGB는 NPY로 저장한다. 검은 픽셀을 유지하고 파일/RGB SHA256을 따로 남긴다.
별도 JSONL에는 캡처 ID, 수집 시작·종료 monotonic 시각, 창 신원·영역을 기록한다.
ID는 **화면 읽기 횟수**다. 같은 RGB라도 새 수집이면 다른 ID를 가지며 이것을 새 게임
렌더 프레임의 증거로 해석하지 않는다. `engine_frame_id`·simulation time·camera calibration은
미확인 상태이며, render freshness·occlusion·body synchronization·actor stream은 false다.
합성 시험 provider는 각 표본에 `acquisition_kind=synthetic_fixture`를 표시한다.

프레임 수·수집률·저장량을 제한한다. 실패하면 이미 기록된 NPY/JSONL과 실패 원인을 남긴다.
HUD·미니맵·driving line·다른 overlay를 자동 제거하지 않으므로 이 기록은 곧바로 actor에
넣는 데이터가 아니다. foreground를 전후 확인해도 중간의 가림이나 렌더링 지연을 증명하지 못한다.

MSS10.2.0을 선택 의존성으로 고정했고 Windows/WSL 검토 가상환경에 설치했다.
영역 grab과 RGB 바이트 API는 [공식 API](https://python-mss.readthedocs.io/latest/api.html),
설치 버전은 [공식 배포 메타데이터](https://pypi.org/project/mss/10.2.0/)를 근거로 했다.
원본 Gym의 screen_capture/worker와 MIT LICENSE도 고정 commit
`ef0fa502bdd257fc2affad9361c02b3dbdfe1632`에서 보존하고 Git blob SHA1·SHA256을 대조했다.
외부 Gym 코드는 실행하지 않았다.

manifest의 소스 해시는 이제 src/configs에 더해 scripts·embedded AC 앱도 포함한다.
캡처 라이브러리 버전도 기록하므로 실제로 실행한 입구와 설치 앱을 추적할 수 있다.

## 실행 방법

```powershell
# 프로젝트 Python 환경에서 한 번 설치
python -m pip install -e ".[capture]"
$env:PYTHONPATH='src'

# 창 신원만 검색. 화면 픽셀은 읽지 않는다.
python -m scripts.record_ac_capture --list-windows --output runs/new-ac-window-discovery

# windows.json의 HWND를 사용. 예시 숫자를 실제 확인한 값으로 바꿔야 한다.
# 실행 후5초 안에 직접 AC 창으로 전환한다.
& {
    Start-Sleep -Seconds 5
    python -m scripts.record_ac_capture --hwnd 0x12345 --frames 10 --rate 10 --output runs/new-ac-rgb
}
```

캡처가 시작될 때 AC가 foreground가 아니면 거절한다. 위5초는 사용자가 창을 전환할
여유이며 자동 focus 조작은 없다. 기록 중 다른 창을 선택하면 중단될 수 있다.

몸 센서는 [TCP 기록기](../integrations/ac/README.md)로 별도 수집한다. 두 기록을 나란히
실행하는 것만으로 동기화가 성립하지 않는다. 같은 호스트인지, 각 clock의 대응과 오차는
무엇인지, 영상은 어느 simulation 구간을 관측하는지 실제 AC에서 확인해야 한다.

## 검증과 원본 증거

| 범위 | 결과 |
|---|---|
| Windows 캡처·AC 통신 선택 검사 | 35통과, 알려진 UDP 실패1건 제외 |
| WSL 같은 검사 | 33통과, Windows 전용3건 건너뜀; UDP 포함 |
| 정적 검사 | 전체 src/scripts/tests/integrations 통과 |
| 실제 Win32 다른 프로그램 창 | 숨겨진 owned 창의 프로세스를 조회하고 캡처 전에 거절 |
| MSS RGB 경로 | 실제 라이브러리의 합성 BGRA→RGB 변환·검은 픽셀 검증; desktop grab 없음 |
| Windows MSS backend | 초기화·모니터 영역 API 확인 성공, grab 호출0·픽셀0 |
| 원본 기록/재생 | 합성4프레임·25,088바이트, 파일/RGB 해시 대조 |
| 실제 AC 창 검색 | 0개, 픽셀 수집0, 조작0 |

이번에는 관련 선택 검사만 수행했다. 직전 전체184통과 결과와 섞어 새 전체 검사 수로
표시하지 않는다. 창 이동·PID/크기 변경·foreground 소실·역행 시각·잘못된 shape·저장량
초과·부분 결과 보존을 확인했다. 새 source snapshot과 실행 로그 등 **195파일을 수집해
SHA256을 대조**했다. `runs/ac-capture-20261027`에 있으며 native 검색은
`runs/ac-window-discovery-20261027-t15b`에 있다.

## 다음 실제 게임 관문

필요한 외부 정보는 실제 AC가 설치된 PC와 경로다. 현재 확인한 Steam 경로·실행 창에는 없다.
PC/설치가 확인되면 센서100표본과 RGB10장을 먼저 기록해 단위·축·카메라/HUD·pause/reset·
clock 대응을 검증한다. 그다음 적용 조작 확인과 저속 구간 학습으로 진행한다.
자율 속도·제동과1랩 완주, 기관 생리학·회로 부호 검증은 여전히 남아 있다.
