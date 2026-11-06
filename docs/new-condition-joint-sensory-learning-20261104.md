# 새 실제 고온 시연의 고정 평가와 공동 감각 학습 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·변속·타이어/셋업/차종·검증된 상대풍을 유지한다. 지도/world pose/절대 heading/교사 명령은 actor 입력에 추가하지 않는다. 측정 구조·기능 근사·학습한 경계를 구분한다.

## 최종 완료 — 공동 학습과 전체 원본 runtime 독립 재생

`joint-fit-01` 원래 handle38076 exit0으로 **823 optimizer 업데이트**와 세 전체 고정 재생을 완료했다. 두 projection 모두 실제로 변했고 최대 변화는 optical .0100000007/native .0100000016으로 float32 허용 오차 안의 .01 bound다. 원래 model51 상태·연결·DN·모든 근육 readout/감각 정규화를 보존했다. 새 체크포인트 SHA는 `42ad488c94a5ff6684387156b4cd94461cd48e9ae7651120ba516f722ebd2f02`이며 별도 연구 후보로 저장했다. 기존 모델을 덮어쓰지 않았다.

아래는 **RMSE 감소율**이며 양수는 감소, 음수는 악화다. 표의 학습/개발 역할을 일반화 시험으로 바꾸지 않는다.

| 기록 | 전체 조향 | 진입 전 조향 | 전체 가속 | 전체 제동 | 진입 전 제동 |
|---|---:|---:|---:|---:|---:|
| 기존 냉간 TRAIN | +1.254% | +1.763% | +1.949% | +2.166% | +3.623% |
| 새 고온 TRAIN | +1.583% | +3.780% | +2.211% | +2.152% | +4.248% |
| 소비된 nominal 개발 | +1.068% | -0.290% | +2.055% | +2.280% | +4.008% |

개발 진입 전 조향은 **.062626708→.062808074(약 .290% 악화)**다. 전체 조향 .070204230→.069454711(약1.068% 감소)과 페달 개선만으로 이를 숨기지 않는다. 전체/진입 전 동시2% 개선 gate는 **false·미승격·actor_qualified=false**다. 새 고온 TRAIN의 진입 전 가속도 약1.35% 악화했다. 제동 일부 개선과 안정적인 시각 선행 조향/실제 차량 성능은 다르다. 두 입력 경계를 함께 학습했으므로 각각의 개선 몫이나 온도 적응 성공을 분리해 주장하지 않는다.

`joint-fit-audit-01` handle33404 exit0은 frozen head/DN·normalization·projection bound·근육→조작 수식·원래 full-lap/pre mask 지표를 독립 확인했다. 이어 학습과 다른 **NativeFlightDecisionRuntime**에 저장 가중치를 연결하고 세 원본 native byte 궤적 **16,425+16,172+16,269=48,866사건**을 모두 재실행했다. 세 기록 모두 저장 출력과 최대차 **0**이다. 출력 몸 나이 최대 약16.003ms·추론 최대11.083ms이며 몸50ms/시각150ms 조건을 완화하지 않았다. 회로 reset은 episode 시작뿐이며 중간 수치 상태를 보존했다. 이 검증은 실제 학생 차량 제어나 자율 랩 성적이 아니다.

![새 고온 TRAIN의 교사·기존·공동 후보 출력과 실제 감각](../runs/thermal-control-improvement-20261104/newhot-joint-traces-01.png)

그래프에서도 강한 제동/조향 peak의 부족과 release 차이가 남아 있다. 이 그림은 학습 기록의 조작 예측이며 새 코너 성적이 아니다.

## 새로 확인한 운동 시각 공간 손실

현재 `PerViewFlyvisStream`과 원래 전처리는 각 방향의 밝기와 modeled T4a–d/T5a–d를 **4×4 Gaussian pool**로 요약한다. 앞서 복원한256차원 공간 성분은 정적 sampled 밝기였다. T4/T5의 pre-pooling 공간 정보까지 복원한 것은 아니었다.

`recover_qualified_motion_spatial`은 새 적격 영상 전체를 원래 author-trained FlyVis 가중치와 원래 시야별 촬영 시각으로 다시 실행했다. 부적격 시작4그룹은 원래처럼 제외하고, 선택되지 않은 중간 적격 영상도 원래 순서로 상태를 진행했다. 시작 steady prior는1회이며 새 수치 reset/미래 영상 조기 설치가 없다. 정적 사진으로 운동 신경 값을 만들어낸 것이 아니다. **modeled activity**이며 생체 기록과 구별한다.

`newhot-motion-spatial-01` handle87886 exit0의 결과:

- 원래 actor가 쓰는1,547영상 그룹·16,172사건의 scene 재구성 최대차 **1.1921e-7**.
- 각4시야×8cell type×721column의 `tanh(activity)`를 기존 pool의 row-space와 그 직교 잔여로 나눴다. 전체 시간 평균을 뺀 변동 에너지 중 **77.6316%**가 기존4×4 요약으로 표현되지 않는 공간에 있었다. 채널/시야별 약60.27–85.61%다.
- 잔여를 기존 pool에 다시 넣은 최대값은 **1.1224e-15**다. 원래 coarse 신호를 바꾸지 않는 별도 성분이다.
- 잔여의8×8 spatial readout인 **2,048차원** 특징과 원래512차원 motion pool, 카메라/사건 join, 직교/시간 변동 통계를 `motion-spatial.npz`에 저장했다. 8×8 readout이721column 잔여 전체를 완전히 복원한다고 주장하지 않는다.

`newhot-motion-spatial-audit-01` handle10764 exit0이 원래 camera/event join·coarse modeled motion 일치(최대6.855e-7), 잔여 평균과 fine pool의 관계·원래 pool 영공간·각 frame의 직교 에너지 항등식·시간 변동 통계를 별도로 재계산했다. 전체 raw 신경 상태를 두 번째 FlyVis pass로 독립 재현한 감사는 아니다.

**77.6%는 운전 정보 비중·시각 기여·정확도 손실률이 아니다.** 이 기록과 모델/선형 요약에서 측정한 활동 변동의 에너지다. 남은 특징이 코너 판단에 필요한 정보인지, 실제 운전을 개선하는지는 인과 실험과 학습/폐루프 평가가 필요하다. **새 운동 공간 특징은 아직 actor에 연결하지 않았다.**

## 현재 완료/미완료와 다음 작업

새 joint objective8검사와 기존 다차원 retinal nullspace3검사의 **합계11검사**가 통과했다. 이전8검사와 합산하지 않는다. full native 평가·joint fit/runtime·motion spatial의 세 독립 감사가 완료됐다. 전체 goal은 active·학생 actuator/새 simulator 조작0이며 실제 학습은 이번823회다. 모든 실제 작업 핸들은 종료했다. 최종 source/원본/결과/체크포인트 해시는 `joint-source-receipt-final-01.json`에 남긴다.

**다음 우선순위는 추가 head/몸 가중 튜닝보다 운동 시각의 세밀한 공간 성분을 검증하는 작업이다.** 다른 TRAIN/개발 원본에도 같은 dynamics/scene parity 검사를 적용해 direction별 공간 잔여를 확보하고, 기존 비행 회로의 시각 입력 경계에 연결할 후보를 만든다. 정규화/선택은 TRAIN만 사용하고 기존 body·근육 readout/회로를 보존한다. 생리 기능 근사와 학습한 어댑터를 구분하며 같은 몸·시작 상태의 좌우/접근/회전 대조로 방향·지연·인과 기여를 검증한다. 기존 기준 후보의 영상 교환6/15는 미해결 근거이며, 이번 joint 후보에 그 방향 검사를 새로 실행한 것은 아니다.

외부 ceiling/교사 curriculum 차이와 실제 온도·자세·가속도/하중/slip 지원 범위를 유지하고, 독립 정지·입력 신선도·단위/부호를 검증한 제한된 학생 폐루프로 이어간다. 가감속·차체 기울기·타이어/변속/셋업/차종/상대풍 적응, 새 코너/트랙의 유효 완주와 빠른 레이싱 라인/랩타임, 생리/R2 검증은 미완료다.

---

## 당시 실행 관측 — 아래 최종 결과로 갱신됨

관측 UTC `2026-11-03T22:22:37.141328+00:00`. 원래 tool handle38076이 실제 live임을 재확인했다. 기록 progress는 training, episode1, 사건3000/16172, optimizer414회다. 고정 학습·최종 재생/개선 판정·후보 독립 감사는 아직 미완료다. 실행 중이라는 이 문장은 다음 턴에서 원래 핸들을 재확인하기 전 현재 상태로 재사용하지 않는다. 목표active·actor_qualified=false·학생 조작0이다.

## 새 자료의 고정 평가

루트 `runs/thermal-control-improvement-20261104/`다. `newhot-cohort-01.json`은 새 실제 hot TRAIN 후보로 기존 hot TRAIN을 교체하고 기존 cold TRAIN과 소비된 nominal 개발 자료를 유지했다. 봉인된 일반화 자료가 아니다. `newhot-spatial-01`은 세 기록의 원본 RGB를 같은 망막 좌표에서 샘플링해 기존 intensity pool과2e-6 이내로 일치함을 확인하고 기존 pool에서 사라지는 grid8 공간 성분을 복원했다. T4/T5 운동 상태를 정적 영상에서 만들어내지 않았다.

`newhot-evaluation-01`은 현재 감사된 future200 optical/native/head를 새 시연16,172사건 전체에 연결했다. 출발/무감독 사건도 원래 시각과 연속 상태로 진행하고 오차는 적격 full-lap2,808라벨에만 계산한다. 외부 ceiling300km/h로 실행했으며 해당 교사 구간은 unlimited final racing stage다. 이전10/20km/h 준비 요청 비교와 구별한다.

| 새 hot TRAIN 후보의 고정 모델 | 조향 RMSE | 가속 RMSE | 제동 RMSE |
|---|---:|---:|---:|
| 전체 랩2,808라벨 | .069259 | .160061 | .149965 |
| 진입 전67라벨 | .048486 | .182545 | .341427 |
| 차속<100km/h,210라벨 | .184024 | .302864 | .207015 |
| 차속≥100km/h,2,598라벨 | .049470 | .142394 | .144372 |

진입 전은 기존 TRAIN-only steering/yaw sign−1과 지속 yaw onset 기반0.8..0.08초 전 mask를 그대로 사용했다. 미래/도로 truth는 훈련/평가 mask·target 전용이며 actor에 넣지 않는다. 재생 몸 나이 최대14.465ms·추론 최대11.704ms, 감독 온도 OOD와 가속도 관측 envelope 이탈은0이다. `newhot-evaluation-audit-01`이 원래 packet/receipt·native sidecar·근육→조작 수식(최대차2.76e-7)과 mask별 지표를 독립 확인했다. 이 감사는 별도 전체 신경 궤적 재구성을 대신하지 않는다. 실제 학생 운전 성적도 아니다.

## 고정 공동 학습 계획

`joint-fit-plan-01.json`은 source hash·cohort·기존 후보를 봉인한 한 번의 TRAIN epoch다. LR1e-4·TBPTT32·Adam·gradient clip1·기존 optical/native projection의 각 요소 변화≤.01을 사용한다. 원래 model51 상태·화학 연결·DN·여섯 근육 readout의 모든 행과 모든 감각 정규화를 고정한다. 기존 native/공간 시각 어댑터의 projection만 함께 학습한다. 두 경계를 함께 바꾸므로 각각의 개선 몫을 분리했다고 주장하지 않는다.

기존 두 TRAIN 역할에서만 현재 교사 조작을 감독한다. 전체 label 가중은 조향/가속/제동4/1/1, 진입 전20/1/4, 고정 이전 예측 anchor는0.5×[4,1,1]이다. 준비 구간 라벨은 burn-in에만 사용한다. 미래 관측·미래 교사 조작·지도·정답 라인은 actor 입력에 없다. 중간 gradient detach는 numerical reset이 아니다.

새 objective의8검사는 target/이전 예측의 gradient 분리, 진입 전 조향/제동의 감독 가중, 유한 값과 bounded projection·reference 보존을 확인했다. 기존 candidate·checkpoint·원본은 덮어쓰지 않았다. 추가 설정 탐색/개발 기반 재선택은 없다.

완료 후 세 전체 episode 고정 재생과 독립 수식/정규화/weight-bound 감사 및 원본 native runtime 전체 재구성을 수행한다. 개발 전체/진입 전 조향 각각2% 개선 및 가속/제동2% 이내 보존은 참고 gate이며 통과해도 actor를 승격하지 않는다. 새 코너·시각 방향6/15·실제 냉간/고온 적응·학생 폐루프·빠른 유효랩/라인·랩타임·생리/R2는 남아 있다.
