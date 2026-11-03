# 겹눈·몸 기관·정책 연결 검증 — 2026-10-27

목표는 실제 초파리 기능 회로와 기관별 몸 피드백으로 Assetto Corsa를 비전 기반 운전하는 것이다. 이번에 확인한 것은 **그 목표의 감각→순환 정책 연결 부품**이며, 학습 성공이나 AC 완주가 아니다.

## 구현한 경로

`MuJoCo 원본 양안 렌더링 → 방향/수용각 기반 겹눈 표본 → 감각 계약 → 몸 기관 부호화 → 공통 감각 인코더 → GRU 또는 합성 희소 순환 회로 → 기존 차량 의도/제어기 → 실제 물리 갱신`

- [compound_eye.py](../src/flybrain_drive/compound_eye.py): 눈별 광학축·수용각·시점·출처의 hash, Gaussian 수용각의 3×3 접평면 적분 근사. 원영상 밖/무효 픽셀은 무효로 남긴다. 깊이 없이 다른 위치의 시점으로 재투영해 stereo를 만드는 요청은 거절한다. 현재 eye map은 **합성 방향 격자**다.
- [mujoco_sensory.py](../src/flybrain_drive/mujoco_sensory.py): 실제 원영상과 기존 V 관측의 계측 채널만 변환한다. 경로 정답 채널이 유효하면 거절한다. 같은 노출의 시각/ID/값은 유지하고 카메라 소실은 누락 표본으로 표시한다. MuJoCo의 물리 토크를 AC throttle/brake로 이름만 바꾸어 넣지 않는다.
- [sensory_policy.py](../src/flybrain_drive/sensory_policy.py): eye map hash·배열·노출 시각 검증, 오래된/누락된 눈·몸 입력 마스킹, 기관 특징과 가용 가속도/휠 신호의 결합, 지속 신경 상태와 episode reset. 일반 학습형 시각 인코더이며 측정된 시각엽 회로는 아니다.
- [policy.py](../src/flybrain_drive/policy.py): 기존 legacy/retina_v1 모델과 별도로 `sensory_encoder` 설정을 받는 순환 정책 경로. 기존 checkpoint의 기본 인코더는 유지한다.

순환 시퀀스 재계산과 역전파도 검증했다. 이는 optimizer를 사용해 주행을 학습시킨 결과와 다르다. 기존 BC/DAgger/PPO 수집기에 새 감각 변환을 통합하는 작업은 아직 남아 있다.

## 실행 환경과 검증

Windows의 MuJoCo DLL 차단 상태는 바꾸지 않았다. 별도 WSL Ubuntu 환경과 프로젝트 전용 `/home/hj/.venvs/flybrain-20261027`을 준비하고 CPU에서 검증했다. 시스템의 venv 지원 패키지와 전용 환경의 의존성을 설치했다. 원격 GPU 작업은 제출하지 않았다.

전체 테스트는 **110 passed, 17.45초**였다. 실제 물리/카메라, 새 감각 경로, 기존 학습·체크포인트·패키징 검증을 포함한다. [전체 실행 로그](../runs/sensory-integration-20261027/pytest-verification.log). 소스 정적 검사도 통과했다. 실제 게임·GPU 성능·생물학적 정확도의 검증은 아니다.

현재 소스/스크립트/테스트의 SHA-256이 WSL 검증 복사본과 같은지 확인했고, 아래 결과16개 파일의 회수 hash도 검증했다. [소스 출처](../runs/sensory-integration-20261027/source-provenance.json), [회수 영수증](../runs/sensory-integration-20261027/collection-receipt.json). WSL 복사본에는 테스트 도구가 요구하는 별도의 로컬 Git snapshot을 만들었으며, 원본 저장소의 커밋과 별도다. 실행 출처는 canonical revision과 파일별 hash로 추적한다.

## 실제 영상·물리 진단

| 모델 | 물리 제어 단계 | 원본 새 노출 | 학습 update | 저장 모델로 행동 재생 최대 오차 |
|---|---:|---:|---:|---:|
| GRU |100|25|0|0.0|
| 합성 희소 회로 |100|25|0|0.0|

두 실행은 실제 렌더링과 물리를 사용했다. 초기 가중치는 무작위이고 목표 완주 관문은 통과하지 않았다. 짧은 이동 거리로 두 모델의 성능 우열을 판단하지 않는다.

- [GRU 결과](../runs/sensory-integration-20261027/runs/sensory-gru/result.json), [합성 회로 결과](../runs/sensory-integration-20261027/runs/sensory-graph/result.json).
- 각 폴더에 `eye-maps.json`, `actor-sequence.npz`, `source-exposures.npz`, `trace.json`, `untrained-checkpoint.pt`, manifest가 있다.
- 진단 스크립트: [diagnose_sensory_policy.py](../scripts/diagnose_sensory_policy.py). 새 출력 폴더를 요구하며 HTML을 만들지 않는다.

```bash
PYTHONPATH=src MUJOCO_GL=egl python -m scripts.diagnose_sensory_policy \
  --kind gru --steps 100 --output runs/new-sensory-gru
PYTHONPATH=src MUJOCO_GL=egl python -m scripts.diagnose_sensory_policy \
  --kind graph --steps 100 --output runs/new-sensory-graph
```

## 실제 연결 데이터 확보

이전 프로젝트의 보존 실행 패키지에서 **MaleCNS v1.0 Traced 선택 그래프**를 데이터로만 복구했다. 165,122개 뉴런·25,563,197개 집계 연결이며 원래 graph SHA-256과 일치한다. CSR의 row는 postsynaptic, column은 presynaptic이다. 과거 코드를 런타임 의존성으로 가져오지 않았다.

[공식 MaleCNS 다운로드](https://male-cns.janelia.org/download/)의 뉴런 주석도 받아 과거 원자료 hash와 대조했다. 전체211,577행 중 Traced165,122행이다. 출처·해시·로컬 위치는 [복구 데이터 등록부](../configs/malecns-recovered-source.json)에 있다. 데이터와 원본 영수증은 `assets/external/malecns-pilot-recovered-20261027/`에 보존한다.

초기 [키워드 후보 목록](../runs/sensory-integration-20261027/annotation-inventory.json)에서 PFL2 12개, PFL3 24개, EPG 50개, DNa02 2개, DNa01 4개, haltere 관련 키워드205개가 매칭됐다. 여러 주석 필드를 대상으로 한 후보 검색이므로 **최종 유형·기관 포트 선정 결과가 아니다.** FeCO/femoral 키워드가 없다는 결과도 그 기관 경로의 부재를 뜻하지 않는다. 유형 별칭·좌우·관절/수용기·연결을 검토해야 한다. [집계 스크립트](../runs/sensory-integration-20261027/annotation_inventory.py)를 함께 남겼다.

이번 정책 진단에 위 실제 그래프를 사용한 것은 아니다. 실제 시각/CX/DN·상행/국소 기능 부분 그래프와 감각 포트는 계속 검토해야 한다.

## 다음 단계

1. 새 감각 변환을 교사 데이터 수집·반복 DAgger·학습 데이터에 연결해 저속 조향 기준선을 학습한다.
2. 확보한 실제 주석과 연결에서 후보의 좌우·유형·시각 위치·감각/하행 경로를 확인하고, 경계 손실·도달성·포트 manifest를 갖춘 작은 기능 회로를 만든다.
3. AC 설치 위치가 확인되면 실제 게임의 영상·몸 계측·조작 bridge를 검증한다. 현재 점검한 등록 Steam 라이브러리에는 엔진이 없었으며 다른 위치의 설치는 미확인이다.

상위 단계와 완료 기준은 [연구 계획](research-plan.md)을 유지한다. 실제 기능 회로를 사용하는 AC 무개입1랩은 아직 미달성이다.
