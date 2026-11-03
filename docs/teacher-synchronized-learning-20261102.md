# 700만 SAC 교사 시연의 정합 검증과 단일 비행 회로 학습 — 2026-11-02

이번 요청의 **정합 적격 교사 시연 수집→실제 오프라인 학습→봉인 평가→독립 감사**를 완료했다. **4epoch·optimizer 13,064회**이며 체크포인트를 저장했다. 학생 자율 완주를 확인한 결과는 아니다.

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 장기 대전제는 [PROJECT_CHARTER.md](PROJECT_CHARTER.md)를 유지한다. 전후좌우 운동 시각, 비행 유래 몸 제어, 조향·가감속·변속·셋업/타이어 적응과 빠른 레이싱 라인·랩타임, 차종 일반화가 목표다. 지도·world pose·절대 heading·정답 경로는 교사/평가 전용이다. 이번 학습 출력은 조향/가속/제동3축이며 변속·셋업 적응이 학습됐다고 주장하지 않는다.

아래 실행 경로의 공통 루트는 `runs/teacher-synchronized-20261102/`다. **700만은 공개 SAC의 학습 스텝 체크포인트를 뜻하며 시연700만 개를 수집했다는 뜻이 아니다.** 차종/코스는 BMW Z4 GT3/Monza다. 다른 공개 교사와 차종 후보의 이전 비교는 [teacher-recovery-20261102.md](teacher-recovery-20261102.md)에 보존했다.

## 적격 시연과 시간 정합

| 독립 재시작 시연 | 유효 전체 랩 | 전체 랩 바퀴 이탈 | 몸 원본 복사→최악 render 상한 | 원천→실제 조작 최대 | 사용 |
|---|---:|---:|---:|---:|---|
| qualified-series-04/episode-02 | 1:52.515 | 0 | 14.8804ms | 37.2255ms | 학습1 |
| qualified-series-06/episode-02 | 1:52.314 | 0 | 10.6410ms | 35.1792ms | 학습2 |
| qualified-series-11-memory-receive/episode-01 | 1:52.455 | 0 | 12.0986ms | 35.9602ms | 봉인 평가 |

각 `qualification/result.json`과 `lap-proof/result.json`이 native/SDK 랩 카운터·타이머·약1랩 진행률·SDK LapInvalidated=0·전체 랩 라벨100%커버리지·정지/제어권 반환을 독립 확인했다. 초기 준비 부분 구간과 유효 전체 랩을 구분하며, 모든 수집 시도가 무이탈이었다고 주장하지 않는다. 첫 두 증명서 SHA는742a00c6…/f3c938e1…, 세 번째는f3a95b1441c0cb6706c30ec58173a92432b8f526106ea86711e4dc2ff80daa6c이다.

정합 판정은 원본 QPC native 복사 bracket, CSP/host clock 교집합, 시야별 렌더 전후 동일 physics timestamp·몸/카메라 pose, 원본 픽셀 광류의 기하 오차와 ±1/2실제 프레임 pose 이동 대조를 사용했다. 사전25ms 기준은 clock 불확실성까지 포함한 가장 늦은 가능한 render에 적용했다. 세 시연 모두 네 시야의 실제 pose 오차가 시간 이동 대조보다 작았다. 몸50ms·영상150ms·결정40ms 기준은 늘리지 않았다.

검증 범위는 **bounded 가상카메라 pose/native-copy 정합**이다. 정확한 GPU 노출 순간·물리 tick atomicity는 미검증이다. 원본 runtime false flag는 그대로 보존하며 별도 감사 증명서가 실험적 오프라인 학습 적격성을 판정한다.

`student-data-02/03/04`는 동일한 frozen author FlyVis로 실제 시야별 사건 순서와 ≤5ms held-input 적분을 처리했다. 부적격 준비 픽셀4묶음은 frozen 신경 상태에도 넣지 않았다. 원본 native ≥10ms 관측과 원래 교사 결정 시각을 합쳤으며 native-only 사건은 명시적 미라벨로 손실에서 제외했다. 보간·재시각 부여·에피소드 내부 수치 reset은 없다.

| 데이터 | 모델 사건 | 전체 랩 감독 라벨 | held-body 최대 나이 | 역할 |
|---|---:|---:|---:|---|
| student-data-02 | 16197 | 2813 | 12.1939ms | 학습 |
| student-data-03 | 16146 | 2808 | 14.8982ms | 학습 |
| student-data-04 | 16293 | 2812 | 11.8968ms | 평가 |

`student-data-audit-01/02/03`은 원본 바이트→body21 값/16valid·5false 마스크, 원래 라벨·시각, 시야별 과거 몸 관측, 비특권 actor 필드를 대조했다. 가속3축과 pitch/roll 회전율은 미검증 마스크false를 유지했다. 지도·world pose·절대 heading·teacher observation은 actor 배열에 없다. 절대 시각/packet 번호는 감사 인덱스이며 신경 입력은 경과시간과 허용 감각이다.

## 지연 원인, 수정, 실패 보존

| 원본에서 확인한 병목 | 변경 | 증거/한계 |
|---|---|---|
| SDK/QPC138.4604ms 공백 안에 session 상태 동기 저장118.7555ms·script.update121.3569ms | 주기적 상태만 single-inflight/no-queue `io.saveAsync`; 원본 생성초/CSP clock/sequence 유지 | source-render-cause-before/after. 수정 후 짧은 계측의 heartbeat 최대1.6694ms·비동기 제출0.0188ms. 이후 추가 병목을 별도 발견 |
| stop 파일 존재 확인201.0456ms·업데이트201.7574ms | 독립 host-owned MMF stop 슬롯·요청ID/generation/체크섬·once-only; 명시적 unpaused MMF 수집 중 불필요한 lifecycle/clock-ping 파일 poll 중지 | qualified-series-08-long-timing 원본. 준비의 owned resume와 정지 뒤 영수증 순서는 유지 |
| 별도 상태 조회600.733ms, 이벤트 flush397.115ms | 프로필/중단 경로를 먼저 비동기 directory notification에 등록하고 초기SHA/존재 확인 후 변경/중단 래치 | notification-preflight-01: 실제NAS 중단 감지23.0818ms·최대조회2.0236ms. 원래 집계에서는 어느 개별 조회가600ms였는지 확정하지 않음 |
| 영상1189가 이미 렌더됐으나 MMF 관측이약64ms 늦음; GPU복사는약2ms | 파일/창 조회를 원본500ms guard가 있는 side observer로 분리. 영상 목적지 검사/저장은 배타적 xb 비동기 worker, 수신은 메모리 관측. 유한 카메라 cyclic GC 종료 뒤 처리 | series10 label1개 영상154.7654ms→100%coverage 불충족, 학습 제외. series11 새 수집은100%coverage |

느린 OS I/O 자체가 사라졌다고 주장하지 않는다. 마지막 정상 수집의 side observer 최대96.2002ms·상태 observer 최대97.1409ms는 중요 수신/제어 경로 밖에서 발생했고 기존500ms 원본 관측 guard 안이었다. 오류·알림 overflow·저장 queue 실패·누락은 명시적으로 실패하며 오래된 감각에 새 시각을 붙이지 않는다. directory notification의 비동기 완료/overflow 처리는 [Microsoft 일차 문서](https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-readdirectorychangesw)를 따랐다. session 상태의 생성초는 보수적 freshness 검사에만 쓰며 물리 센서 정합 증거로 쓰지 않는다.

SDK/native 전후 동일 prefix와 공통 QPC, inline nonblocking TCP, TCP 중 호환 파일 I/O quiescent fence, 독립 native 기록, 원본QPC health, timer/counter 전이 확인,120FPS 제한도 유지했다. 기존 타이어/차량 물리는 바꾸지 않았다.

series10의 정상완주1:52.457 뒤 남아 있던 옛1200-count validator가1566그룹을 거부했다. `timing-validator-recovery-10`은 사전1600/90ms/150초 범위·65909행·overflow0·실제 소비 오류없음을 원본 해시로 확인했고 원래 실패 결과는 바꾸지 않았다. 그러나 `qualification-10-recovered`에서 실제 영상150ms 위반 라벨1개를 확인해 **이 시연은 여전히 학습 제외**다. 최종 평가는 새 정상 수집series11이다. 다른 실패/이탈/불일치도 모두 보존했다.

검사 기록은 `mmf-stop-and-original-status-tests.txt`43개, `notification-metadata-tests.txt`15개, `long-timing-validator-tests.txt`30개, `memory-receive-tests.txt`42개 통과다. 일부 검사는 서로 중복되므로 합계를 고유 검사 수로 주장하지 않는다. [당시 단계별 기록](teacher-synchronized-learning-20261102-history.md)과 원본 run 디렉터리를 함께 따른다.

## 실제 단일 회로 학습

`student-fit-plan-v2.json`과 `student-split-01.json`을 optimizer0에서 봉인했다. 시간순 첫2적격 에피소드5621라벨만 학습, 세 번째 독립 에피소드2812라벨은 평가 전용이다. 평가로 조기 종료/모델/설정을 선택하지 않았다. seed780205, Adam.lr.001, 고정4epoch,8모델사건 TBPTT, 가중MSE[4,1,1], gradient norm1, 실제 업데이트**13,064회**다. detach는 수치 상태 reset이 아니다.

기존9728뉴런·6비행근 출력의 화학 topology/dynamics와 보호된 haltere 경계는 고정했다. 학습 대상73797개 파라미터는 공학적 시각·opponent temporal·시야별 몸 문맥·차량 감각 되먹임 어댑터와 근육→조향/가속/제동 readout이다. frozen author FlyVis의 실측 시각 topology/author-trained weights와 공학적 camera lattice/feature pool을 구분한다. readout 기준 평균/std는 학습 시연만으로 계산하고 scale floor.001을 고정했다. speed80m/s·wheel300rad/s 정규화는 차량 대응 가설이다.

| epoch | 학습 시연1 가중MSE | 학습 시연2 가중MSE | 누적 업데이트 |
|---|---:|---:|---:|
| 1 | 0.100645 | 0.086344 | 3266 |
| 2 | 0.073120 | 0.056383 | 6532 |
| 3 | 0.043985 | 0.038303 | 9798 |
| 4 | 0.035656 | 0.033899 | 13064 |

`student-fit-01/epoch-01~04.pt` 및 최종`unified-sac-student.pt`를 저장했다. 실제 실행 소스는 manifest hash와 대조해 `executed-sources/`에 보존했다. source revision/dirty 상태·버전·원본 hash는 run manifest에 기록했다.

## 봉인된 한 에피소드 평가와 독립 감사

가중MSE는 세 축 squared error에[4,1,1]을 곱한 평균이다. 조향은 교사 요청steering/.6의[-1,1], 가속·제동은 각[0,1] command fraction이며 물리 가속도/도로 바퀴 각도의 오차가 아니다.

| 조건 | 가중MSE | 조향 MAE | 가속 MAE | 제동 MAE |
|---|---:|---:|---:|---:|
| 학습 전 zero readout | 0.174121 | 0.068052 | 0.473628 | 0.468973 |
| 학습 평균만 예측 | 0.086794 | 0.068274 | 0.307866 | 0.141968 |
| 학습 후 전체 감각 | 0.042298 | 0.048229 | 0.155467 | 0.116819 |
| 시각 입력0 clamp | 0.044339 | 0.050774 | 0.162983 | 0.118393 |
| 몸 입력/학습 되먹임 clamp | 0.122931 | 0.080624 | 0.480658 | 0.219526 |

전체 감각 모델의 가중MSE는 학습 평균 상수 기준보다 **51.3% 낮았다**. 시각 clamp는 오차를4.8% 늘렸고, 몸 clamp는 **2.91배**로 늘렸다. 이번 한 시연에서는 몸 감각 의존성이 훨씬 크며 시각 기여는 작다. 이는 정보 clamp 비교이고 생리학적 lesion/세부 뉴런 기능이나 범용 주행의 증거가 아니다. 한 에피소드의 시계열2812라벨을2812개의 독립 일반화 시험으로 취급하지 않는다.

`student-fit-audit-01/result.json`은 체크포인트·데이터·split/source hash와 saved prediction 지표를 독립 재계산했다. **고정 모델 상태45개가 원본과 정확히 일치**, 보호된 haltere 경계도 일치했다. 변경을 확인한 감각 파라미터는 visual.weight/bias, temporal.opponent_projection.weight, view_context.projection.weight, body_feedback.projection.weight다. temporal.projection.weight는 이번 fit에서 변하지 않았다. 감사 자체의 optimizer/차량 제어/actor 배치는0이다.

## 원본 식별과 현재 상태

| 항목 | SHA-256 |
|---|---|
| 7M SAC teacher | 80cf52ab1c61c54920a9f0da1b3eab906fa2a0bebb640ae9b91374f1cc442549 |
| 설치 CSP v6 | a99ef883eb2a37ce40db66c482817c9781d5754eb35039065ec62bb138f663ed |
| frozen FlyVis author weights | d8a57a022aa18ca338599be713af9db29b89cde4b00098ffe42b78f0d583e46a |
| fixed motor 초기 checkpoint | 01a7a68372b76c68336f0403da48702a7f2f6f181f3e2567aecc6e9bd0e43116 |
| fit plan | 56f8ce451b3e0cae6743ec12613297cc71cfd8e7f2b15f41064473f3d548923f |
| prefit split | a18241d5cc71ceaba41cfe70f2e1794a30f5486726c45f9b375000053837f164 |
| 최종 학습 checkpoint | 24a8f4e03614636fe6eede71137faae161c8e82dcc1f212dc4bd4a75fb5203d4 |
| heldout predictions | fd1741a68de8394885df7d86ea199efb92778e63828c938fd3d012ff25870f4e |

실행 git revision=9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda, dirty=True이며 uncommitted 실행 소스의 개별 hash와 보존본을 manifest/receipts로 구분한다. 입력 배열 hash는 student-data-02 ae9e66454ba04f3c4a83d78a02cb1293889e134d0b1a736c8b77a6476e4afbe6, -03 05809a09319cfc96ad390de315e89b281bd20036c2494394d1fb939057021b5f, -04 b2f366c1b36df7561a3078a94b861b456bb03280ac9123dc69c5f880a3260990다.

`student-final-state-01.json`(22:55)과`student-final-state-02.json`(23:06)의 재조회는 엔진 PID42156/HWND5573972·paused·camera/AIoff·vJoy FREE1·AC 비전면을 확인했다. native는 paused frozen baseline임을 명시하고 새 물리 표본이라고 주장하지 않았다. 마지막 live 정지 증거는 series11 `qualification/lap-proof/result.json`이다. 교사/카메라/학습 제어 프로세스는 종료됐다. AC 데스크톱 입력0, API/background 인터페이스만 사용했다.

다음 연구는 학습된 actor의 실제 차량 폐루프/유효 랩·실시간40ms 예산 검증, 약한 시각 경로와 비행 부품 반응의 세부 인과 검사, 미검증 몸 축 등록, 차종/타이어/셋업/변속 적응 및 새로운 코스에서의 라인·랩타임 평가다. 정확한 GPU 노출 순간·초파리 전체 생리·학생 자율 완주·차종/셋업 일반화는 **아직 미검증**이다. 이번 오프라인 시연 학습을 그 장기 목표의 완료로 바꾸어 부르지 않는다.
