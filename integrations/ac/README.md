# AC 읽기 전용 센서 수집 앱

이 폴더의 앱은 읽기 전용이다. 실제 AC는 설치·실행했고 네이티브 공유메모리에서250표본을
받았다. SDK 앱은 실제 내장 Python에서 socket import 실패를 확인해 로컬 파일 전달 대안을
추가·배치했다. 수정 앱의 재로딩은 미검증이다. 학습기는 게임 내 Python과 분리한다.

Windows의 첫 원시 기록은 앱 없이 다음처럼 할 수 있다. 주행 엔진이 기존 물리/정적 매핑을
발행해야 하며, 없는 매핑을 새로 만들지 않는다. 출력은 매번 새 폴더를 지정한다.

```powershell
$env:PYTHONPATH='src'
python -m scripts.record_ac_native_physics --samples 100 --output runs/new-ac-native-raw
```

이는 단위·시계·영상 동기가 검증된 actor 스트림이 아니다.
[실제 계측·제한](../../docs/ac-native-physics-20261027.md)을 먼저 확인한다.

## 설치와 첫 기록

아래는 선택적인 SDK 앱 경로다. 실제 설치된 원본 Assetto Corsa의 연습 세션에서 사용하며,
네트워크 확장이 없는 내장 Python에서는 로컬 파일 전달로 전환한다.

1. 원본 Assetto Corsa 설치 경로를 확인한다. `apps/python` 아래에 이 폴더의
   `flybrain_telemetry` 디렉터리를 복사한다. 최종 파일은
   `apps/python/flybrain_telemetry/flybrain_telemetry.py`다.
2. 게임의 Python 앱 설정에서 활성화하고 연습 세션에서 앱을 켠다.
   가능하면 `127.0.0.1:2348`에서 TCP 연결 하나를 받는다. 네트워크 확장을 불러올 수 없으면
   `%LOCALAPPDATA%/flybrain/telemetry/`에 최신 값 파일 두 개만 쓴다. 외부 PC에는 공개하지 않는다.
3. 저장소의 Python 3.11 이상 환경에서 다음 명령으로 최대100표본만 기록한다.
   실행마다 새 출력 폴더를 사용한다.

```powershell
$env:PYTHONPATH='src'
python -m scripts.record_ac_tcp --samples 100 --output runs/new-ac-raw-recording
# 로컬 파일 전달을 사용한 경우:
python -m scripts.record_ac_file --samples 100 --output runs/new-ac-file-recording
```

게임 Python에 Torch나 이 프로젝트 패키지를 설치할 필요는 없다. 앱 오류는 AC의
Python 로그에서 확인한다. 포트 충돌·접속 실패 시 기록기는 실패 결과와 수집한 부분을 남긴다.
설치·실행과 수정 앱 배치는 수행했으며, 수정 앱 재시작과 실제 SDK 전달 수신은 남아 있다.

## 데이터와 제한

- SDK에서 읽은 회전속도·휠 회전·서스펜션·조향·페달·속도·랩 수를 원값으로 보낸다.
  경로·세계 위치·AI line은 읽지 않는다. 랩 수와 속도는 자동으로 actor 입력이 되지 않는다.
- SDK는 `acUpdate` 주 스레드에서만 호출한다. 네트워크 스레드는 최신 표본만 전송한다.
  표본 건너뜀은 허용하되 sequence 역행·중복·producer session 변경은 수신기가 거절한다.
- `timestamp_ac`는 **송신 프로세스의 perf_counter**다. 수신 PC의 monotonic 시계와
  빼서 지연을 계산하면 안 된다. `elapsed_seconds_candidate`는 deltaT 누적 후보일 뿐이다.
  pause/reset/replay/랩 경계 검증 전에는 simulation time으로 봉인하지 않는다.
- SDK 단위·축 부호·`Steer`의 의미는 차량별로 검증해야 한다. 실제 계측 근거가 없는
  변환을 기본 SI 센서로 등록하지 않는다. 누락값은 null이며 0으로 복구하지 않는다.
- 영상·노출 ID·노출 시각·적용 조작 확인은 제공하지 않는다. 기존 Gym 이미지 버퍼도
  노출 메타데이터가 없어 바로 겹눈 actor로 연결할 수 없다.
- 50Hz는 송신 시도 상한이다. 실제 새 표본율·지연·안정성은 게임에서 측정해야 한다.
  이 앱만으로 자율주행이나 몸 감각 동기화가 완성되지 않는다.

  순수 Python 문법은3.3 기준으로 검사했고 SDK 대역과 현대 Python의 localhost TCP에서
검증했다. embedded Python 3.3 실행·SDK 필드 지원·앱 활성화는 실제 게임에서 확인할 항목이다.
앱을 끄고 제거하려면 세션을 종료한 뒤 설정을 해제하고 복사한 앱 디렉터리만 제거한다.

프로토콜 조사와 검사 기록은 [AC 연결 검토](../../docs/ac-bridge-20261027.md)에 있다.

별도 [AC 창 영상 기록기](../../docs/ac-capture-20261027.md)도 준비했다. 원본 RGB와 수집
구간을 기록하지만 렌더 프레임·simulation time·몸 센서와의 동기화는 아직 보장하지 않는다.
