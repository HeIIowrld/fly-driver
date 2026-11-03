# 네 시점 GPU 공유 텍스처 전송 검증 — 2026-10-28

이전 턴의348m 경로 실패 및 카메라 지연 진단은 진전으로 분류한다. 이번에는 영상 전송 경로를 실제로 바꾸고 정차/짧은 주행에서 검증했다. 전체 서킷 주행과 최종 모델 성능은 여전히 미검증이다.

## 검토한 경로

공식 CSP SDK의 [`GeometryShot:accessData`](https://github.com/ac-custom-shaders-patch/acc-lua-sdk/blob/7cf60f5a406820ee4e39f3a62ed121fa36274001/lib_scene.lua#L1623)와 `sharedHandle(true)`를 확인했다. 설치 SDK에서도 같은 API를 확인했다. 원본과 보충 UI 소스의 해시는 `assets/external/csp-camera-api-20261028/async-data-provenance.json`에 보존했다.

비동기 GPU→CPU callback 경로 t850은4그룹/16장 모두 같은 shot의 PNG와 픽셀 차이0이었다. 하지만 callback 지연 중앙318.427ms,최대327.139ms라 현재150ms 입력 한도에 맞지 않았다. 이 경로를 주행용으로 채택하지 않았다. 실제 RGB 변환 중앙0.311ms,파일 저장0.364ms와 callback 대기는 별도다.

직접 읽기는 Microsoft의 [`OpenSharedResource`](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11device-opensharedresource), [`CopyResource`](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-copyresource), CPU READ staging texture와 nonblocking Map을 사용한다. 원본 게임 텍스처는 복사의 source일 뿐이며 destination은 별도로 만든 텍스처다. 공식 Microsoft header에서 COM 슬롯·구조체 배치·GUID를 대조하고 `assets/external/d3d11-shared-texture-api-20261028`에 원본/해시를 보존했다.

기본 D3D 장치는 Intel Graphics였고 t852는 OpenSharedResource invalid argument로 종료됐다. 소유 lease를 닫았고 게임/다른 앱을 변경하지 않았다. DXGI adapter1이RX9070XT임을 실제 조회한 뒤 t853에서4시점×4회 픽셀 차이0,예열 뒤 단일 시점 읽기 중앙1.130ms·최대2.111ms를 확인했다. 장치 선택을 명시하고 GPU 메타데이터를 기록한다.

## 소유권과 프레임 수명

`shared_immutable_v1`에서는 Lua가 네 시점을 렌더하고 각 텍스처를 그대로 유지한다. Python이 네 텍스처를 모두 자신의 CPU 버퍼로 복사하고, 원래 request ID·generation·capture ID가 일치하는 ACK를 보내야 다음 렌더가 가능하다. 마지막 프레임도 ACK 전에는 source를 해제하지 않는다. 중단 파일·기존 deadline·원래 카메라 예산은 유지한다.

Python은 등록된320×180 RGBA8/BGRA8만 받고 stride와 포맷을 확인한다. 원래 렌더 시작 시각을 유지하며 수신 시각으로 바꾸지 않는다. RGB는 호스트에서 무손실 BMP로 저장해 원본 픽셀 재생 감사에 사용한다. 모델의 네 시점·망막 투영·노출·몸 감각·구조적 그래프를 바꾸지 않았다. 단,실제 frame age와25Hz 목표 수집 간격은 기존PNG/30ms 경로와 달라진다.

원본 메타데이터·GPU복사 시각·handle·텍스처 설명·ACK의 연결을 `verify_shared_frame`으로 검증한다. 공유 자원은최대8개만 캐시하고 종료 때 모두 반환한다. 비동기 callback의318ms 지연을 입력 한도 확대로 허용하지 않았다.

## 정차 검증

| 실행 | 범위 | 결과 |
|---|---|---|
| t855 | 연속8그룹,PNG를 복사 전에 저장 | 32장 RGB 차이0 |
| t856 | PNG 생성 없는120그룹 | 완료,모든120개 frame/copy/ACK 연결 검증 |
| t866 | PNG를 호스트 복사 후 생성,정차/paused | 32장 RGB 차이0 |
| t868 | PNG를 호스트 복사 후 생성,unpaused/brake hold | 32장 RGB 차이0 |

공유 업데이트의 GPU 완료 순서가 중요하므로,복사 전 PNG 저장이 GPU 작업을 강제로 완료시킨 효과에 기대지 않도록 t866/t868에서 검증 순서를 바꿨다. 원래 shot을 바꾸지 않고 호스트 복사 뒤 PNG를 생성해 비교한다. 짧은 검증 범위의 픽셀 일치이며 모든 미래 GPU 스케줄 상황의 동기화를 무조건 보증하는 주장은 아니다.

t857 timing 감사에서 t856의 도착 시 이미지 나이 상한 중앙20.071ms·최대51.180ms,시점별 GPU 복사 중앙1.081ms·최대4.171ms였다. 과거PNG t822의 도착 나이 중앙43.942ms·최대85.997ms와 비교할 수 있으나 정차/주행·수집기간·시계 범위가 다른 실행이므로 통제된2배 개선이라고 주장하지 않는다. 현재 측정은 실제 host시각과 같은 source clock bounds를 사용한다.

## 기존 모델을 연결한 실제 주행

t859 API 준비·t860 읽기 전용 사전검사 후 t861에서806 모델을 새 경로로 운전시켰다.257개 요청 뒤20.015297km/h에서 상한20을 넘겨 중단했다. 바퀴 이탈0·유효 랩 상태였고 native 정지/브레이크/FREE1을 확인했다. 카메라 오류는 없었지만60m 목표에는 도달하지 못했다.

t863은257회 요청을 원본 BMP·몸 감각에서 재생해 최대 차이 `[0,0,0]`과 공유 frame/copy/ACK를 검증했다. 실제 입력의 나이/샘플 분포가 바뀐 조건에서 기존 모델이 과속한 기록이며,전송 개선이 모델 주행 개선을 보장하지 않는다. 원래 시각을 일부러 늦추거나 속도/도로 기준을 완화하지 않는다.

## 새 경로에서 교사 자료

t873은 실제 AI 소유권 관측 뒤 student vJoy를 반환하고 독립 native 수신을 시작했다. 그러나 heartbeat JSON을 읽은 뒤 경로의 stat을 따로 조회하는 구간에서 FileNotFoundError로 조기 종료됐다.128개 raw native 수신의 해시·packet ID·앞뒤 같은 실행의 packet 범위를 검증했다. 실제 수신 간격 중앙15.461ms였다.10ms는 목표 주기이고 실제100Hz 달성을 주장하지 않는다. 조기 종료 자료를 성공30초 자료나 학습 봉인으로 바꾸지 않았다.

상태 내용과 수정 시각을 같은 열린 파일 핸들의 json.load/fstat으로 읽도록 수정했다. 상태의2초 신선도·같은generation/epoch·AI 소유권 확인은 유지한다. 파일 경로가 교체되는 테스트를 추가했다. t874는학생251개 요청을 원본으로 재생해 차이0을 확인했다. 종료 후 AI off·pause·native brake hold와FREE1을 확인했다.

고정 런타임 t875,API 준비 t876·사전검사 t877 뒤 t878에서 같은30초 교사 수집을 진행한다. 실제 완료와 원본 감사/봉인 전에는 새 학습 자료라고 세지 않는다. 기존 모델과 데이터,실패 기록은 모두 보존한다.


## 완료된30초와 관측 주기 재확인

t878은30초 AI 구간과 정지 처리를 완료했다. 학생249·대기3·AI733,총985개 관측과 독립 native1993표본을 얻었다. t879의 학생 재생과t880의 전체 감각 배열·학생/대기 신경 재생은 오차0이었다. 다만 AI 구간에84.601ms 관측 간격1개가 있어80ms 기준을 넘으므로 학습 봉인에 채택하지 않는다. 삭제·보간·시각 변경·인계 지점의 state reset은 하지 않는다.

AI 관측 루프가 actor 입력에 쓰지 않는 SDK 진단 JSON의 갱신도 기다리고 있었다. 학생 운전의 SDK 진단/물리 guard는 유지하고,AI 관측만 같은 native reader에서 직접 표본을 읽어 `expert_raw_native` 이벤트에 원본 bytes/수신 시각을 저장하도록 분리했다. 원본 감사와 라벨용 index 대응도 이 명시적 이벤트를 검증한다. 카메라·몸 감각 입력과 oracle 분리는 유지한다.

관련265개 테스트와Ruff 후,t881 고정 런타임·t882 API 준비·t883 사전검사를 거쳐t884를 실행한다. 같은 소유 handle의 종료 결과를 확인한다. 추가 학습과 모델 승격은 아직 하지 않았다.


## 최신 실행의 종료 결과

t884는AI 인계 전에 학생788개 요청 뒤 inference80.3504ms가80ms 기준을 넘어 중단됐다. 입력 나이는125.828ms로150ms 이내였지만 두 기준을 모두 만족해야 하므로 요청을 추가하지 않았다. 최고15.647269km/h,SDK 경로112.070530m였고 타이어/속도 기준을 완화하지 않았다. 관측789개의 최대 간격55.549ms·80ms 초과0개이나 AI 표적이 없으므로 교사 학습 자료는 아니다. t885는788개 학생 요청의 원본 입력·공유copy/ACK 재생 오차0을 확인했다. SDK 진단을 기다리지 않는 AI 관측 분기는 구현/단위 검사됐지만 이 실행에서는 AI 활성화 전 종료돼 실제 검증하지 못했다.

현재265개 관련 프로토콜·영상·학생·고빈도 native·표적 검사와Ruff가 통과했다. 이번 턴에는 새 모델 학습을 하지 않았고 기본 모델을 바꾸지 않았다. 기존806은새공유경로에서 과속 실패와추론시간 초과 기록이 있으므로 이 경로의주행 자격을 부여하지 않는다.

다음 작업은 `scripts/record_ac_engine_ai_probe.py`의 기존 정차 출발 AI15km/h·30초 수집기에 동일 공유 전송을 연결하는 것이다. 이 수집기는이미SDK 진단 없이native 몸 감각을 읽는다. 새 영상 나이/대비 분포의 실제 전문가 자료를 먼저 확보하고,원본 감사·미래 적용 제어 표적·동일 감각 보정 확인 뒤 GPU 학습한다. 학생의16~18.5km/h 인계 조건이나80ms 추론/관측 기준을 낮춰 성공시키지 않는다. 이후 새 모델을60m→240m→곡선320m→480m→전체 서킷으로 다시 검증해야 한다.

최종 t886 API pause와t887 읽기 전용 확인에서background·paused·AI off·camera idle·expert inactive·gas0·brake1·속도0.000190km/h였다. 모든 소유 실행·학습·감사 handle은 종료됐다. 전체 서킷 완주·일반 제동·타이어/하중/셋업 최적화는 미완료이며 목표는 활성 상태로 유지한다.

최종 기록 시각: 2026-10-29T00:05:47.926909+09:00 (Asia/Seoul). 파일 이름의20261028은 작업 시작일이다.
