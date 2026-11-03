# 실제 AC의 조향·페달 전달 준비

갱신: 2026-10-27, Asia/Seoul. 네이티브 물리 센서 기록과 설치된 vJoy 장치 조회는 성공했다.
실제 게임 축 적용은 미검증이다. 센서 공유메모리에 조작값을 쓰는 방식은 사용하지 않는다.

## 최신 확인: 설치 완료, 장치 조회 성공

사용자가 설치 진행을 승인하고 설치 완료를 알렸다. `vjoy` 드라이버의 Running 상태와 SDK의
enabled 값을 확인했다. 장치1은 FREE였고 조회된9축의 범위는 모두0..32767이었다. 설치된
`C:/Program Files/vJoy/x64/vJoyInterface.dll`은 공식 릴리스 SDK의 동일 파일과 SHA-256
`199316aaefac95118e869c07e7f332ce492dd9a8b6dac5a3ca20b1d5bb5058a1`이 일치했다.
조회 기록은 `runs/ac-vjoy-installed-probe-20261027-t29/`에 있다.

DLL의 조회 함수는 실행했으나 장치 획득·축 전송은0건이다. 이후 사용자 Esc로 Computer Use가
중단돼 실제 게임 축 설정과 적용값 검증은 진행하지 않았다. 아래 드라이버 부재와 미실행
기록은 설치 전 상태다. 현재 설치 동의를 다시 기다리는 상태가 아니다.

## 정차 축 시험 준비와 현재 실행 차단 요인

`scripts/probe_ac_axes.py`와 `configs/ac-axis-probe.json`을 추가했다. 기본 실행은 읽기 전용
사전 점검이며, `--execute`가 없으면 장치를 획득하거나 축을 전송하지 않는다. 별도 실행 모드는
정차 기준·작은 좌우 조향·제동1/0.5/0·가속0.1과 제동 유지·마지막 중심 요청의8단계다.
각 단계8표본으로 유한하며, 요청 전/후 원시 physics와 요청 영수증을 함께 기록한다.

실행 전 정확한 profile hash·장치 GUID/index·DLL hash·차량/트랙·단일 게임 창을 확인한다.
매 요청 전 profile과 창 생산자를 재확인하고, 새 센서 누락·속도 초과·처음 중립 피드백 불일치면
시험을 종료한다. 종료 시 중심/가속0/제동1을 요청한 뒤 장치를 놓는다. 이 종료 요청이나 SDK
수락은 실제 게임 적용을 보증하지 않으며 적용 지연·축 단위·방향은 결과에서 따로 검증한다.
중립 요청을 보낸다는 이유로 잘못된 축 반전이 해결됐다고 가정하지 않는다.

`runs/ac-axis-preflight-20261027-t36/`의 실제 읽기 전용 점검은 다음을 확인했다.

- Z4 GT3·Monza와 원래 `acs.exe` 창1개, vJoy 장치1 FREE·GUID 일치.
- 실제 controls.ini SHA-256은 `952ebd7bede93c5904011f6db32b7b7fa9c8be2bb8534d170cad338a00a4155e`.
- 준비한 vJoy 후보 SHA-256은 `96d142e0dde91630c6eab235ed5da60f1b6d613dfffb324f2301b6294a03a048`.
- 실제 파일의 `INPUT_METHOD=KEYBOARD`이며 후보 profile 적용·게임 재로딩은 안 됐다.
- 장치 획득·축 전송0건. 실제 적용 검사 완료로 표시하지 않는다.

현재 남은 필수 실행은 이전 Esc 중단 뒤 자동 조작 재개 응답, 중립 유지 상태에서 후보 profile
적용/게임 재로딩, 정차 축 시험의 실제 계측이다. 같은 재개 질문은 이미 사용자에게 제시했으며
응답이 오기 전에는 profile을 바꾸거나 `--execute`를 실행하지 않는다.
대역 장치의 유한 요청·원시 표본 기록·사전 guard 실패·센서 누락/속도 초과/중립 불일치 정리
검사5개와 Ruff를 통과했다. 실제 적용 결과는 아니다.

```powershell
python -m scripts.probe_ac_axes --output runs/new-axis-preflight
```

기존 AC Gym의 [고정 버전 설치 안내](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/INSTALL.md)는
Windows에서 vJoy를 조작 전달 경로로 사용한다. 현재 PC의 일반적 vJoy DLL 경로3곳은 없고,
이름에 vJoy·ViGEm이 포함된 등록 드라이버 서비스도 발견하지 못했다. 이는 조사한 경로·이름의
결과이며 모든 가능한 제어 방식의 부재를 증명한 것은 아니다.

## 검토·준비한 설치 파일

[BrunnerInnovation 공식 릴리스 v2.2.2.0](https://github.com/BrunnerInnovation/vJoy/releases/tag/v2.2.2.0)에서
설치 파일과 SDK를 받았다. 릴리스는 Windows10/11 지원과 서명을 명시한다. 실제 파일의
Windows Authenticode 검사는 `Signature verified`, 서명자는 `Brunner Elektronik AG`였다.

- 파일: `vJoySetup_v2.2.2.0_Win10_Win11.exe`,5,302,312바이트.
- SHA-256: `ef569a3105cd301b89580f18f60c66b339e95296acf2c0dfcaf4b4bbf8ab68fe`.
- 로컬 준비 경로: `C:/Users/jeff4/AppData/Local/flybrain-control-prerequisites-20261027/`.
- 원본·다운로드/서명 기록: `assets/external/vjoy-control-20261027/`.
- 설치 준비 영수증: `runs/ac-control-prerequisite-20261027-t29/receipt.json`.

설치 파일·DLL은 실행하지 않았다. 새 가상 조이스틱 드라이버를 추가하는 설치는 별도 사용자
동의를 받아야 하고, Windows 관리자/보안 확인은 사용자가 직접 처리해야 한다. 테스트 서명 모드나
Windows 보안 설정을 바꾸는 절차는 사용하지 않는다.

## 설치 뒤 검증할 계약

SDK 헤더의 cdecl·32-bit LONG/BOOL 규격과 장치 소유권/축 조회/갱신 함수를 검토했다.
설치 뒤 실제 장치 ID와 X/Y/Z 축 범위를 조회하고, 조향[-1,1]·가속[0,1]·제동[0,1]을
게임 설정의 축·반전과 일치시켜야 한다. 다른 프로그램이 소유한 장치는 가져오지 않는다.
드라이버가 요청을 받았다는 사실과 실제 게임의 steer/gas/brake 적용값을 구분한다.

첫 확인은 정지 상태에서 가속0·제동 유지·조향 중심을 적용하고, 네이티브 센서에서 적용값을
확인하는 것이다. 이후 작은 조향과 저속 가감속, 포커스 변화에 따른 적용 여부를 확인한다.
현재 이 검사·축 매핑·자율 조작은 모두 미완료다. 모델의 가속·제동 학습 또한 별도 과제다.

`ac_vjoy.py`에 SDK cdecl·108바이트 전체 조이스틱 프레임·실제 축 범위 조회·축 반전·장치
소유권·종료 시 가속0/제동 유지 요청을 구현했다. 대역 SDK2검사와 ABI/범위/부분 실패/재획득
검사를 통과했으며, 설치 파일·실제 DLL 실행·가상 장치 제어는 아직 하지 않았다.
