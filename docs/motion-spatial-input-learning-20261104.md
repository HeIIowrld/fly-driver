# 운동 시각 공간 잔여의 새 비행 입력 경계 — 2026-11-04

> **후속 2026-11-04 해석 제한:** 원래 반대 코너 쌍의 도로 횡위치/속도 혼입을 실제 원본으로 확인했다. 방향5/5를 독립 코너 인식으로 해석하지 않는다. [후속 기하·원본 영상 검증](precorner-road-pose-confound-20261104.md).

> **후속 2026-11-04:** 새 motion의 실제 camera/CPU owner 연결·531저속 shadow 출력·픽셀/몸/회로 독립 감사가 완료됐다. 이 문서의 live 미연결/다음 작업은 당시 상태다. [후속 원본·종료·미검증](motion-live-integration-20261104.md).

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 가감속·변속·타이어/셋업/차종·검증된 상대풍을 유지한다. actor에 지도/world pose/절대 heading/교사 조작을 넣지 않는다. 측정 구조·modeled neural activity·공학 변환·학습한 차량 어댑터를 구별한다.

## 최종 완료 — 새 시각 경계 학습·matched 비교·전체 runtime 감사

두 원래 fit handle98157/78527은 exit0으로 종료했다. 각 **823 optimizer 업데이트·합계1,646회**이며 새 motion projection만 변했다. 모든 기존 회로/감각 경계/모든 근육 readout/정규화는 보존했다. 이 턴 이전의 joint823회와 다시 합산하지 않는다. fine checkpoint SHA는 `fc4a2997867f75cffc6ad879379c5e365799dffdc46c68321af4188ac4130b19`, coarse checkpoint SHA는 `c863877ab87001a402ffdfb17ddec8bf7e654e58776c74fb63e687b32154fb4b`다. 두 연구 후보는 별도 저장했고 기존 모델을 덮어쓰지 않았다.

소비된 nominal 개발 기록의 적격 전체 랩2,811라벨·진입 전86라벨:

| 고정 후보 | 전체 조향 RMSE | 진입 전 조향 RMSE | 전체 가속 RMSE | 전체 제동 RMSE |
|---|---:|---:|---:|---:|
| 기준 후보 | 0.070204230 | 0.062626708 | 0.157732604 | 0.152315010 |
| 기존 coarse 복제 | 0.069168132 | 0.062178348 | 0.157343664 | 0.152236007 |
| 새 fine motion | 0.068762703 | 0.060789692 | 0.157353387 | 0.152031885 |

fine은 기준 대비 전체 조향 **2.053%**, 진입 전 **2.933%** 감소했고 가속/제동은 약.240%/.186% 감소했다. 기존 전체/진입 전 동시2% 및 페달 보존 gate는 fine만 통과했다. 그러나 matched coarse 대비 추가 감소는 전체 **.586%**, 진입 전 **2.233%**다. 사전에 정한 **두 조향 지표 모두 coarse보다1% 이상 개선** 조건에는 전체가 미달하므로 matched information gate=false·미승격·actor_qualified=false다. 기준 gate 통과와 전체 actor 자격을 같은 것으로 부르지 않는다. 새 설정을 골라 다시 학습하지 않았다.

TRAIN cold의 전체/진입 전 조향은 기준 대비 약1.706%/5.853%, 새 hot TRAIN은 약2.505%/5.300% 감소했다. 이 기록들은 학습 자료이며 새로운 코너/온도 일반화 성적이 아니다. 개발 진입 전 가속은 약.150% 악화했고, fine이 모든 페달/구간에서 coarse를 이긴 것도 아니다.

두 `*-audit-01`의 원래 handle59303/95624도 exit0이다. 각 frozen 상태·TRAIN-only 정규화·weight bound·원래 supervision/window의823업데이트·근육→차량 수식·모든 mask별 지표를 별도로 재계산했다. 이어 **MotionSpatialDecisionRuntime**에 각 저장 포트를 연결하고 **각48,866사건, 두 후보 합계97,732사건**의 세 원본 native byte 궤적을 재실행했다. 두 후보 모두 세 기록에서 저장 출력과 최대차 **0**이다. 영 포트의 원래 후보 복원도 각 episode 첫96사건에서 최대차0이며, 이를 전체 영 포트 재생으로 확대하지 않는다. binding/입력 확인 시간을 포함한 출력 몸 나이는 최대 약15.878ms로50ms 안이었다. 실제 live capture/IPC/actuation 경로의 통과를 뜻하지 않는다.

## 같은 몸·같은 시작 상태에서 새 운동 공간 성분만 바꾼 결과

`development-interventions-01`(handle82910 exit0)은 fine 후보의 개발 기록 전체16,269사건을 다시 재생하고 정상 결과와2e-6 이내 일치함을 확인했다. 각 sustained onset1.2초 전의 원래 recurrent state를 복제해 세 분기를 동일한 상태에서 시작했다. 원래 몸21/native19·각 mask·현재/촬영 당시 몸 이력·기존 scene576/static320·모든 수신/결정 시각은 고정하고, **추가된 fine motion 성분만** 유지/평균값/반대 코너 잔여로 바꿨다.

- 개발 기록 **5구간·진입 전86라벨**의 정상 조향 RMSE **.060789692**.
- TRAIN 평균 fine 특징으로 교체(새 포트의 정규화 입력0): **.062640478**, 정상보다 약 **3.045% 증가**. 5구간 중1개는 이 제거가 오차를 조금 줄였으므로 항상 유리하다고 주장하지 않는다.
- 반대 코너 fine 특징으로 교체: **.064692319**. 최대 조향 변화는 canonical **.0187341**이다.
- 반대 donor에 대한 **가설상의 방향 순서가5/5구간에서 양수**였다. margin은 약.00178–.01517이며 데이터와 함께 보존했다.

이것은 기존 coarse 시각까지 보존한 **새 특징 경계의 인과 개입**이다. prefix state 복제/몸 고정은 해당 source-bound 실행에서 hash/정상 분기 parity로 확인했고, `development-intervention-audit-01`(handle64076 exit0)은 receiver 입력 byte hash·원래 normal 궤적·mask·분기 통계를 독립 재계산했다. 이 감사에서 반사실 신경 궤적 전체를 별도로 재실행한 것은 아니다.

**5/5는 물리적으로 옳은 반사실 운전 성공률이 아니다.** donor의 도로 상대 자세/위치는 완전히 맞지 않고 독립 조작 정답도 없다. 원래 coarse/정적 이미지 특징은 그대로 둔 채 고해상도 modeled motion만 교체했으므로, 원본 픽셀 전체를 바꿔 FlyVis를 다시 돌린 실험도 아니다. 기존 기준의 전체 영상 특징 교환6/15와 다른 개입이며 그 수치를5/5로 대체하지 않는다. 익숙한 Monza 시각 패턴의 암기 가능성과 새 코너 일반화는 여전히 미검증이다.

## 완료 상태와 다음 실제 통합

이 단계는 **progress**다. modeled motion 공간 정보를 기존 비행 입력 포트에 실제로 연결한 오프라인 후보·한 번의 matched 학습·전체 native runtime 재생·같은 몸/시작 상태의 정보 개입 근거를 남겼다. 관련29검사와 세 영상 복원 감사, 두 후보 runtime 감사, 한 개입 통계 감사가 통과했다. 결과·실패 로그·source/계획/체크포인트 hash는 `runs/motion-spatial-learning-20261104/source-receipt-final-01.json`에 정리한다.

**아직 live GPU camera owner/CPU owner 사이에 새2048 motion 특징을 전달하는 통합은 하지 않았다.** 새 runtime class만으로 live 지연/새 센서 반복·실제 학생 차량 제어가 검증됐다고 주장하지 않는다. 이번 simulator/UI/학생 actuator 조작은0이다. 전체 목표active·미승격이며 한계 주행 완료가 아니다.

다음은 기존 `PerViewFlyvisStream`의 같은 author state에서 새 motion 잔여를 만들어 원래 camera identity/clock과 함께 CPU 소유자에 전달하고, 영상150ms/몸50ms와 독립 종료 계약을 실제 read-only 움직임 시험에서 확인하는 것이다. 실제 생물학적 세포/연결/생리 반응과 공학적 readout을 계속 구별한다. 동일 몸·도로 상대 자세/위치 대조와 새 실제 코너/트랙 평가도 필요하다. 추가 평균 오차 튜닝만 반복해 시각 선행 판단을 대신하지 않는다.

제한된 학생 폐루프에는 별도 actuator owner·출력 단위/부호·신선도·독립 정지·초기 감각 준비 계약이 필요하다. 조향/가감속·기울기/가속도/하중/타이어·변속/셋업/차종/검증된 상대풍 적응, 실제 유효 완주와 빠른 레이싱 라인/랩타임 및 생리/R2는 미완료다.

---

## 당시 실행 관측 — 최종 결과로 갱신됨

두 실제 tool handle98157(fine_motion),78527(coarse_duplicate)을 관측했고 두 분기 모두823회 학습 후 세 전체 기록의 고정 재생을 진행 중이다. 구체 시각/진행은 `runs/motion-spatial-learning-20261104/observed-running-01.json`과 각 progress 원본에 있다. 다음 문맥에서는 원래 핸들을 확인하며, 결과 파일/옛 PID만으로 종료나 실행 중을 추정하지 않는다. 현재 개선 판정·후보 전체 runtime 감사·같은 몸의 방향 반응 검사는 미완료다. 전체 목표active·학생 actuator0·actor_qualified=false다.

## 실제로 추가한 부품과 정보 경계

이전 단계의 정적256 망막 잔여와 달리 새 입력은 **원래 author-trained FlyVis T4a–d/T5a–d modeled activity**의 공간 성분이다. 각 시야/방향/cell column을 원래 입력 순서로 유지해8×8 잔여 readout4×8×64=2,048차원을 사용한다. 새 항은 기존 비행 회로의 visual sensory port에서만 더해지고 모든 차량 명령은 여전히 기존 여섯 근육 readout에서 나온다. 새 직접 조향/CNN/RNN 출력이나 측정된 새 생물 회로라고 부르지 않는다.

`MotionSpatialResidualPort`는 TRAIN-only2048 평균/척도·명시적 .005 floor와 clipping±5를 사용하고 .25*tanh로 추가 신호를 제한한다. projection은0으로 초기화해 기존 입력을 보존한다. 기존 model51 상태/연결·DN·정적 optical·native body·근육 readout과 그 정규화는 모두 고정하고 **새 motion projection만** 학습한다.

`MotionSpatialObservation`/`MotionSpatialDecisionRuntime`는 episode·원래 camera receipt/네 촬영 clock·원래 coarse motion reference를 확인한다. 파일 hash/카메라 사건 join은 별도 적격 데이터 loader에서 검증한다. 몸50ms/영상150ms는 기존 runtime을 유지하고, 새 binding에 걸린 시간까지 출력 몸50ms 안에 포함한다. 실패 후 numerical rewind나 자동 중립/재시도는 없다. 이 경계는 오프라인 실험용이며 live GPU/camera owner에 새 특징을 연결하거나 actuator를 갖게 한 것은 아니다.

## 세 원본의 복원과 독립 감사

루트는 `runs/motion-spatial-learning-20261104/`다. `cold-motion-01`/`development-motion-01`은 기존 cold TRAIN/소비된 nominal development를 원래 시야별 촬영 순서·author steady prior1회·연속 visual state로 재생했다. 새 hot TRAIN은 이전 완료된 `runs/thermal-control-improvement-20261104/newhot-motion-spatial-01`을 참조하며 재복제하지 않는다. original actor scene576의 수치 parity를 각 영상에서 검증하고 부적격 시작 영상은 원래처럼 제외했다.

세 독립 motion spatial 감사가 camera/event join·기존 modeled motion·직교 energy identity/시간 변동 통계를 확인했다. 기존4×4 row-space 밖의 시간 변동 에너지 비율은 cold76.908%,newhot77.632%,development77.716%다. 이는 modeled neuron column activity의 변동 에너지이며 운전 정보 비중/생리 기능 손실률/성능 감소율이 아니다. 8×8 readout이 버려진721column 성분 전체를 완전히 복원한다는 뜻도 아니다.

## 고정 matched 학습 계획

`fit-plan-01.json` source hash를 봉인한 두 분기:

- **fine_motion:** 원래 pool의 null-space에서 복원한 세밀한 운동 잔여를 추가한다.
- **coarse_duplicate:** 기존 actor가 이미 보는4×4 T4/T5 값을2×2씩 복제해 동일2,048차원으로 만든다. 새 정보 없이 같은 추가 경계/학습 예산의 효과를 비교한다. 밝기 channel은 대조군에 섞지 않는다.

각 분기 같은 seed·새 경계의 영 초기화·두 TRAIN의 full-lap mask·Adam LR1e-4·TBPTT32·gradient clip1·1epoch·가중치 변화≤.01을 사용한다. 조향/가속/제동 label 가중4/1/1, 진입 전20/1/4 및0.5×[4,1,1]의 원래 예측 anchor를 사용한다. 모든 teacher/future/truth는 loss/평가 전용이며 actor 입력이 아니다. 준비 구간은 burn-in뿐이다. 정규화는 TRAIN 두 기록의 원래 감독 사건만 사용하고 개발 자료는 선택에 쓰지 않는다.

기준은 현재 감사된 future200 optical/native 후보이며, 앞선 미승격 joint 후보로 바꾸지 않았다. 원래 cold TRAIN·새 hot TRAIN·이미 소비된 nominal 개발의 역할을 유지한다. 같은 Monza 반복이며 봉인된 새 코너 시험이 아니다.

`comparison-plan-01.json`은 개발 결과가 나오기 전 고정했다. 기존 전체/진입 전 조향 각각2% 개선과 페달2% 이내 보존에 더해, fine 분기가 matched coarse 분기보다 전체/진입 전 조향을 각각1% 이상 줄이는지를 본다. 통과해도 actor를 승격하지 않는다. 개발 결과를 보고 설정을 반복 고르지 않는다.

## 검사·실패 보존과 남은 작업

현재 관련 **29검사 통과**(`port-data-runtime-tests-02.txt`): 영 초기화, 시야/방향/공간 순서, bounded 신호/gradient, TRAIN-only 정규화·미래/개발 배제, coarse 복제 정보 보존, capture clock/episode/coarse reference 오류 거절, 전체 binding+inference deadline, rejection 후 상태를 되감지 않는 계약과 기존 native runtime을 포함한다.

첫 새 runtime 테스트3개는 실제 optical port 대신 일반 Linear를 사용한 테스트 대역에 `.projection` 구조가 없어 실패했다. 실제 `RetinalResidualPort`로 대역을 고쳤으며 실패 로그와 이전 테스트 파일을 보존했다. 실행 중인 학습/봉인 source는 바꾸지 않았다.

세 전체 고정 재생 뒤 두 후보의 근육 수식/고정 상태/정규화/weight bound와 원본 native+typed motion runtime 재구성을 독립 검증한다. 다음은 같은 몸/시작 상태에서 좌우·접근/회전 영상의 올바른 선행 반응과 신경/근육 이전 기여다. 기존 기준 후보의6/15 방향 문제는 미해결이며 이번 후보에 이미 방향 검사를 했다고 부르지 않는다. 조향뿐 아니라 가감속·하중/타이어/자세/가속도·변속/셋업/차종/상대풍, 실제 학생 폐루프·미학습 트랙 유효 완주와 빠른 라인/랩타임·생리/R2는 미완료다.
