# 공개 교사 속도 단계·랩 검증과 후보 비교

> **이 문서는 원인 수정 전의 과거 단계 기록이다.** 이후 유효 전체 랩과 랩타임을 확인했다. 최신 상태와 차종을 포함한 후보 해석은 [교사 완주·지연 해결 보고서](teacher-recovery-20261102.md)를 따른다. 아래의 “전체 랩 0/미확정”은 당시 결과다.

기준 날짜: 2026-11-02, Asia/Seoul. 사용자가 요청한 공개 교사 평가다.
초파리 회로의 학습·자율 주행 결과가 아니다.

## 목표와 정보 경계

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다.
차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른
레이싱 라인/랩타임을 학습한다. 변속·셋업 적응과 상대풍 감각도 목표에 포함한다.
이번 교사는 지도·world pose·미래 곡률을 사용하지만 학생 actor에는 전달하지 않는다.

## 평가 계약

- 기존 BMW Z4 GT3/Monza와 타이어 설정을 유지한다. 공장 기본 셋업·원 학습 셋업 일치는 미확인이다.
- 카메라를 끄고 교사만 평가한다. 원본 SDK/native/graphics와 조작을 기록한다.
  영상과 몸의 물리적 동기를 검증하거나 학습용 시연을 생성하는 시험이 아니다.
- 목표 속도 20→40→60→80→120→160→220→280km/h. 목표+20km/h가 중단 한도다.
  제한기는 기존 정책의 페달 요청을 변경하며 목표 속도의 일시 초과가 가능하다.
  높은 한도는 그 속도에 실제 도달했다는 뜻이 아니다.
- 기존 첫 시험은 단계당 12초, 이후 속도별 반응을 분리하는 시험은 단계당 3초다.
  마지막 단계 최대 400초, 최대 2개의 전체 랩을 기록한다. 한 조건의 반복 안정성은 별도다.
- 조향 보정 요청 배율 0.6은 같은 세션의 측정 ±270 SDK 응답을 따른다.
  원본 감각 간격 최대 80ms, SDK/native 수신 차 최대 50ms, 경로 gap 최대 3m와
  기존 3개 이상 바퀴 이탈 중단 기준을 유지한다. 모든 바퀴 이탈은 별도로 기록한다.
- 시작은 정지·전진 기어·중립 조작의 실제 응답을 요구한다. native/CSP AI 양쪽 꺼짐,
  카메라/probe 꺼짐, 동일 세션·프로필·장치와 background 상태를 계속 검사한다.
  창 활성화·키 입력을 하지 않는다. 재시작은 `scripts/control_ac_session.py`의 CSP API만 쓴다.
- `live-02` 뒤 추가한 자동 변속 계약: native autoShifterOn=1, 이전 전진 기어,
  10km/h 이상에서 SDK/native 중립 전환을 최대 300ms 허용한다. 이때 정책 예측 대신
  직전 조향과 가속/제동 0의 타력 요청을 보낸다. 역방향·변속 비활성·지속 중립은 중단한다.
  실제 중립 값을 지우거나 회로/교사 이력을 초기화하지 않는다.
- `live-04` 뒤 추가한 수집 계약: 30ms 미만의 새 SDK 원본은 기록하고 실제 다음
  패킷을 기다린다. 시각을 다시 쓰거나 원본을 보간하지 않는다. 80ms 초과는 계속 실패다.
- `live-07`부터 전체 랩용 별도 조건에서는 기준 레이싱 라인과의 거리 한도를 12m로
  명시한다. 3m는 트랙 경계가 아니므로 이 값 초과를 실제 트랙 이탈로 쓰지 않는다.
  native 3바퀴 이상 이탈·속도·시각·제어권 중단은 유지한다. 너무 이른 원본에서도
  역방향/3바퀴 이상 이탈을 검사하고 한 바퀴라도 이탈하면 clean 자격을 잃는다.
- 첫 결승선 통과는 부분 랩으로 버린다. 이후 연속 진행률 약 1랩과 native lap 증가,
  iLastTime을 함께 요구한다. 바퀴 이탈 0의 clean lap과 AC 공식 유효 랩 flag는 구분한다.
  현재 timing prefix는 공식 유효성 flag를 제공하지 않아 `official_valid_lap_verified=false`다.
- 에피소드 내 이전 조작/관측 이력을 유지하고 장치 획득/명시적 재시작에서만 새 이력을 만든다.
  종료 원인·요청·드라이버 수락과 실제 적용은 구분한다. 모든 기록은 training_admissible=false다.

## 실제 시험

원본은 저장소 `runs/teacher-speed-curriculum-20261102/`에 있다.

| 실행 | 관측 결과 | 종료 원인 |
|---|---|---|
| preflight-01 | 차량·제어권·조향 보정 사전 검사 통과, 조작 0 | 읽기 전용 완료 |
| live-01 | 요청 266/피드백 265, 최고 19.912km/h | source 간격 83.3085ms, 12초 단계 미완료 |
| live-02 | 요청/피드백 299, 20/40/60/80 단계 완료, 120 목표에서 최고 93.910km/h | 고회전 자동 변속 중 중립; 기존 짧은 pilot 기어 gate로 중단 |
| live-03 | 조작 0 | 직전 제동 정지 후 gap -5.615m로 시작 gate 불통과 |
| reset-01 / start-01 / first-gear-01 | CSP 재시작·출발 API 완료, 가속 없이 1단 확인 | 각각 완료 |
| live-04 | 요청 7/피드백 6, 최고 1.229km/h | source 간격 19.9805ms가 기존 20ms 최소보다 짧음 |
| live-05, 기준 540만 | 요청/피드백 344, 120 목표 단계 완료, 최고 119.118km/h, 변속 타력 6회, 바퀴 이탈 0 | 160 목표 단계에서 gap -3.046m, native 트랙 이탈 아님 |
| live-06, 후보 700만 | 요청/피드백 404, 160 목표 단계 완료, 최고 142.570km/h | 220 목표 단계에서 기존 3m 기준선 한도 초과 |
| live-07, 후보 700만·12m 조건 | 요청 131/피드백 130, 최고 39.447km/h | source 간격 실패, 전체 랩 시도 미완료 |
| live-08, 기준 540만·12m 조건 | 요청 388/피드백 387, 최고 137.547km/h | 너무 이른 원본에 native 기어 4·바퀴 4개 이탈: 트랙 이탈 중단 |
| live-09, 후보 700만·12m 조건 | 요청 438/피드백 437, 최고 151.220km/h | native 기어 4·바퀴 3개 이탈: 트랙 이탈 중단 |
| live-10, 후보 400만·12m 조건 | 요청 90/피드백 89, 유효 조립 피드백 최고 36.858km/h | source 간격 81.6115ms로 중단; 마지막 미조립 native에는 38.384km/h 기록 |

`live-02` 목표별 실측 최고 속도는 20.872 / 43.558 / 59.000 / 78.874 / 93.910km/h다.
관측된 중립 직전 native autoShifterOn=1, RPM 8,475, 기어 2였고 이후 기어 1·gas=0이었다.
기어 인코딩은 SDK/native 모두 1=중립, 2=전진 1단이다.

540만/700만의 3m 조건 비교는 각각 한 번이며, 차량 위치·셋업·타이어를 재시작으로
대체로 맞췄지만 시작 타이어 온도/실측 압력·원본 수집 주기/시스템 부하와 실제 조작 이력은 다르다. 후보가 더 높은 속도와
뒤쪽 진행률까지 갔다는 관측을 유효 랩 우월성으로 승격하지 않는다.
완주·랩타임은 아직 확인하지 못했다. 자동 변속·gap·랩 증거 관련 단위 검사와 기존
pilot 회귀를 합쳐 17개 통과했다. 출력 차이 검증과 실제 폐루프 비교는 서로 다르다.

### 최종 판정과 독립 감사

**현재 설정·단계적 속도 제한에서 세 정책 모두 전체 랩을 기록하지 못했다. 랩타임 비교는 불가능하다.**
540만/700만은 실제 트랙 이탈이며 400만은 수집 실패로 검증이 잘렸다.
36.858km/h를 400만 정책의 속도 한계나 성능 순위로 쓰지 않는다.
3m 조건의 `live-06`에는 바퀴 이탈이 최대 2개였으므로 이 구간도 zero-tyre clean은 아니다.
`live-01/02`의 최대 이탈도 2개다. 이전 시점의 별도 저속 교사 50결정 이탈 0과 혼동하지 않는다.

`scripts/audit_teacher_curriculum.py`가 `audit-01/result.json`에 10개 실행의 이벤트 파일 SHA와
모든 SDK wire/native prefix/graphics prefix 해시를 대조하고 source 간격·조작/피드백 수·
진행률·바퀴 이탈을 재계산했다. source 간격은 명시적 history reset에서 분리한다.
너무 이른 원본의 바퀴 이탈도 감사에 포함했다. 장치 획득 후 종료한 모든 실행에 가속 0·
조향 0·브레이크 1의 close 수락 기록이 있다. 시작 거부 `live-03`은 조작 0이다.

감사 시 서로 다른 native packet 3개에서 정지·가속 0·브레이크 유지와 vJoy FREE를 확인했다.
native/CSP AI 양쪽 false, camera/probe false, session 오류 없음, post_stop_verified=true다.
수집된 모든 기록은 학습 부적격으로 유지했고 학생 optimizer 업데이트 0이다.
최초 위치는 재시작 시험들에서 spline 약 0.85643으로 유사하지만 온도/압력은 완전 일치하지 않는다.
예를 들어 `live-08/09/10` 시작 core 온도는 각각 약 26.76/26.50/26.46°C,
압력은 약 16.12/16.08/16.07이며 연료는 모두 native 30이다. 설정을 바꾼 것이 아니고
재시작 뒤 경과 시간에 따라 물리 상태가 달라져 완전히 통제된 반복 비교라고 주장하지 않는다.

다음은 수집 간격 실패를 별도로 안정화하고, 실제 고속 이탈 구간의 원 학습 셋업·보조 기능·
legacy 관측/조작 주기 및 저속 curriculum으로 달라진 이력을 점검해 일치 조건의 반복 랩을 확보하는 것이다.
그 전에는 700만 후보로 교사를 확정 교체하거나 학생 학습을 시작하지 않는다.

## 공개 후보 조사

| 후보 | 제작자가 보고한 범위 | 공개·호환성 확인 | 비교 판단 |
|---|---|---|---|
| AssettoCorsaGym SAC step 5,400,000 | BMW Z4 GT3/Monza 예상 112.1초 | 현재 기준 가중치 SHA `3f30cbb1…157e84`; 로컬 완주 미재현 | 기준 교사 유지; 112.1초를 로컬 결과로 쓰지 않음 |
| 같은 학습의 step 4,000,000 / 7,000,000 | 개별 checkpoint 우월성 보고 확인 안 됨 | 각 663,951 bytes 내려받고 해시 고정, 동일 입출력 정책으로 로드 | 직접 비교 가능한 추가 후보. 더 긴 학습=더 빠른 교사로 단정 불가 |
| Vector-Q Transformer SAC | Mazda Miata/Monza 완주율 80% | 전체 Git tree에서 `docs/demo/inference_ckpt.pt` 26,380,237 bytes 공개 확인. demo 관측 216개, 75×50 입력 | 다른 차량이므로 BMW 교사와 랩 순위 비교 불가. 후보의 공개 가중치가 없다는 판단은 하지 않음 |
| progressive DisCor/F1 | Monza 평균 92.95초, 10랩 평가 보고 | 사용 예제 차종 f1_2026_audi_fom. 저장소의 BMW 데이터는 기존 20241108_SAC 계열 | 다른 차종이므로 112.1초와 순위 비교 불가 |
| AssettoCorsaRL SAC/VAE | F1/Monaco 대상 | 문서는 checkpoint 실행 예시 제공, README의 AC 이식은 roadmap 미완료로도 표기 | BMW/Monza에서 재현 가능한 우수 교사 근거 미확인 |
| SamuelOrtizV/Assetto-Corsa-RL | 영상·분산 학습 framework | 실행/학습 구조는 공개, 같은 차량/트랙의 우수 랩 및 가중치 짝은 확인 못함 | 도구 후보이며 즉시 교체할 검증된 교사는 아님 |

현재 조사로 **기준 교사보다 낫다고 검증된 동일 조건 공개 교사는 찾지 못했다.**
조사 범위에서의 결론이며 모든 공개 모델에 대한 부재 증명은 아니다.

동일 `live-02`의 299개 관측을 세 정책에 넣은 오프라인 비교는
`model-comparison-offline-01/result.json`과 `predictions.npz`에 보존했다.
세 정책은 모두 유한한 3축 출력을 냈다. 기준 대비 평균 절대 delta 차(조향/가속/제동)는
400만 모델 0.28962/0.15163/0.34342, 700만 모델 0.35040/0.09528/0.25317이다.
기준 교사가 만든 관측에 대한 조건부 출력 비교이며 후보의 폐루프·완주·랩타임 순위가 아니다.

추가 가중치는 `assets/external/acgym-teacher-candidates-20261102/`에 있다.
400만 SHA `c7e0dd8a2d2715b8b6fdaa207f0747647d6d0cd6acc83210bddf0b94cb91ba12`,
700만 SHA `80cf52ab1c61c54920a9f0da1b3eab906fa2a0bebb640ae9b91374f1cc442549`.
원본 다운로드 URL·시각·크기는 `provenance.json`, 공개 목록은
`public-survey/hf-monza-model-inventory.json`, GitHub 조사 응답은 `public-survey/github-evidence.json`.
두 추가 가중치는 API가 선언한 LFS SHA-256과도 일치한다 (`public-survey/download-verification.json`).
재귀 모델 목록 감사는 `public-survey/recursive-model-inventory.json`이다. Vector-Q repo는
revision `45d33fa3b33252031d91384b0f47fc449a276672`의 전체 트리를 검사해 demo 가중치를 찾았다.
일반 checkpoint/model 폴더의 placeholder만 보고 가중치 부재로 결론 내리지 않았다.

## 출처

- [기준 SAC 예제와 예상 랩타임](https://github.com/dasGringuen/assetto_corsa_gym#benchmarks)
- [같은 차량/트랙의 공개 가중치 목록](https://huggingface.co/datasets/dasgringuen/assettoCorsaGym/tree/main/data_sets/monza/bmw_z4_gt3/20241108_SAC/model/checkpoints)
- [기존 체크포인트/최신 인터페이스 호환 이슈 및 제작자 설명](https://github.com/dasGringuen/assetto_corsa_gym/issues/24#issuecomment-3515132040): 제작자는 인터페이스 변경 가능성을 지적하고 동작하던 commit `e1af2fa6dafadf2f3b38b68a9f0a47e8340c1059`를 안내했다. 우리 재구성은 원본 상태 231개로 별도 검증한 legacy 관측/코덱을 사용하며 이 댓글만으로 현재 호환성을 확정하지 않는다.
- [Vector-Q 성능과 차량](https://github.com/chandankeelara/transformer-based-autonomous-racing-agent#results), [확인한 checkpoint 폴더](https://github.com/chandankeelara/transformer-based-autonomous-racing-agent/tree/main/assets/checkpoints), [model 폴더](https://github.com/chandankeelara/transformer-based-autonomous-racing-agent/tree/main/assets/trained_models)
- [Vector-Q의 실제 공개 가중치와 demo 설명](https://github.com/chandankeelara/transformer-based-autonomous-racing-agent/tree/main/docs/demo)
- [F1 progressive 정책 결과·차종](https://github.com/tanmaiy18/assetto_corsa_gym-main)
- [AssettoCorsaRL의 설명·roadmap](https://github.com/AssettoCorsaRL/AssettoCorsaRL), [checkpoint 실행 예시와 foreground 요구](https://assettocorsarl.github.io/AssettoCorsaRL-DOCS/quickstart/)
- [SamuelOrtizV의 framework](https://github.com/SamuelOrtizV/Assetto-Corsa-RL)
