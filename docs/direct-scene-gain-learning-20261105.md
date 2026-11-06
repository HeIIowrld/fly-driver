# 장면 시각 경로 분리·직접 포트 보정의 matched 학습 — 2026-11-05

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

기존 직접 장면 입력이 retimed 영상 방향 점수를 약화시키는 위치를 분리했고, 새42계수 보정을 실제395회 학습했다. 그러나 같은 출발 대조군보다 진입 전 조향 오차가 냉간.0136%·고온.1076%·소비 개발.0082% 커졌다. 두 조건 모두 직전 부모보다 진입 전 오차가 커서 미승격이다. 이번에 조향이 실제로 개선됐다고 판정하지 않는다. 원본·실패·최종 가중치를 모두 보존한다. 학생 제어·simulator 변경API·desktop 자동화는 이번 턴0회다.

실제 한계 주행·다른 타이어/셋업/차종 적응·상대풍/변속·빠른 유효 랩/레이싱 라인/랩타임은 미완료다. 현재 read-only actor의 .723 부모 체크포인트를 바꾸지 않았다.

## 다섯 시각 경로 분리

TRAIN10개 영상 교환에서 scene/optical_static/dn_static/dn_motion/visual_motion의32조합640branch를 비교했다. 수용자 몸/native/clock/회로 prefix는 같다. identity와 원래 미분리 전체 교환의 별도 재생 최대차0이다. 원본은 runs/visual-route-factorial-20261105/diagnostic-01 및 independent-audit-01이다.

| 경로를 donor로 바꾼 효과 | 평균 contextual margin 변화 | 양성/음성 맥락 |
|---|---:|---:|
| scene | −.001489154 | 32/128 |
| optical_static | +.000131589 | 144/16 |
| dn_static | +.000825019 | 128/32 |
| dn_motion | +.005568580 | 152/8 |
| visual_motion | +.000100298 | 132/28 |

각 경로의16맥락×10수용자 결과다. 정적 하행 투사는 대체로 도움이 됐으므로 먼저 해제하려던 준비를 중단했다. 위 값은 독립 기여율이 아니며 비선형 상호작용을 합산하지 않는다. donor는 원래 upstream 시각 이력과 다른 횡위치 등을 포함한다. 단순 코너 곡률만의 물리 반사실/생리적 결함으로 해석하지 않는다.

추가4조합80branch에서 직접 scene 투사와 나머지 scene 유래 시간·몸 결합을 나눴다. 나머지4시각 경로는 모두donor다. 둘다recipient일 때 margin .006524126, 나머지scene만donor .006000447, direct만donor .005506649, 둘다donor .005130822였다. 부모factorial의동일 두 극단과 최대차0이다. 직접 투사가 약화의 더 큰 부분에 관여했지만 나머지scene도 음성이었다. 직접 투사만이 유일한 원인이라는 증거가 아니다.

이 보조 실행의 manifest route list/scope는 이전5경로 설명을 잘못 상속했다. direct-scene-split-01/protocol-clarification.json이 실제4조합/80branch와bit의 의미를 명시하며 원본 manifest/source/result는 보존했다.

정적 Adam3그룹395/395/331은 원래 gradient에서 복원해 모든proposal차0을 확인했지만 새 actor 학습에 쓰지 않았다. 근거는 runs/joint-static-motion-learning-20261105/static-optimizer-reconstruction-01이다.

## 실제 학습한 수정

DirectScenePortGain은 기존 direct scene의42개 visual-port logit에 .5*tanh(delta*(logit−TRAINreference))를 더한다. delta 시작0·전체norm≤.5·entry≤.25·업데이트norm≤.002이며 ref는 TRAIN5659결정만 쓴다. 기존 chemical graph·시각 투사 가중치/정규화·근육 차량 판독은 고정한다. optical/fine motion을 더하기 전 원래 direct logit에만 적용한다. body·지도·world pose·교사 명령은 새 모듈의 입력이 아니다. 이는 공학적 입력 보정이며 실측 시냅스나 생리적 적응을 새로 검증한 것이 아니다.

초기1024출력 최대차0, 기존4그룹 Adam1975/1655/662/662의moment/step 완전복원, 새그룹0을 확인했다. 실제 새gain gradient의 FD 최상대오차는.3913%다. 작은step의4.47%/14.24%/24.18% 오차도 보존했다. 가상 영상에서는 gain gradient가 있지만 native/speed는grad=None·Adam이동0이다. 사전검사의 disposable2회는 실제fit 가중치에 반영하지 않았다.

frozen_direct/learn_direct는 동일 부모·seed782011·데이터·목적함수·순서로 각32597실제 결정과395업데이트(감독301/명령30/영상교환10/합성54)를 수행했다. 기존17446계수와 새42계수로 optimizer의수치 벡터는17488개다. 대조군의42개는 고정0이고 학습조건만 갱신한다. 원래 motion norm2/native 누적norm.05·entry.02/speed norm.5·power직접민감도2000은 기준을 새 부모로 리셋하지 않았다. real 이력에 부모prefix 주입0회이며 episode 이외의 numerical reset은 없다. gradient detach는 numerical reset이 아니다.

두fit의 원래 세션18630/34937은exit0이다. 최종Adam은각2370/1986/993/993/0 및2370/1986/993/993/395, 새gain norm0/.1352197이다. 모든395 Adam proposal과 마지막moment를 독립 재구성해 최대차0을 확인했다. 첫 차이는 update0의 real gain gradient였고 그때 기존4그룹 gradient/갱신과 목적함수는 같았다. 개발/예약 자극으로 마지막checkpoint를 고르지 않았다.

## 최종 가중치 평가

각48866결정의3기록·45명령·30소비 개발 영상교환·새 합성texture915320293을 평가했다. 다음은 조향 RMSE 감소율이다. 음수는 악화다.

| 조건 | 진입 전 원래 fine 대비 | 진입 전 직전 부모 대비 |
|---|---|---|
| frozen_direct | +.979% / +.560% / +.438% | −1.104% / −4.606% / −2.133% |
| learn_direct | +.965% / +.453% / +.430% | −1.117% / −4.718% / −2.141% |

learn_direct의 matched frozen_direct 대비 진입 전 감소율은 −.01357% / −.10761% / −.00819%, 전체 조향 감소율은 −.03716% / −.04883% / −.02854%다. 새gain은 이 비교에서 조향을 개선하지 않았다. 원래 fine의 낮은label오차와 당시 몸 감쇠0/30이라는 국소 부호 문제도 구분한다. 낮은재생오차만으로 더좋은물리운전자라고 판정하지 않는다.


가감속도 함께 재생했다. learn_direct의 원래 fine 대비 전체 조향 감소율은 -24.695% / -30.973% / -29.347%, 가속은 -3.079% / -7.352% / -7.817%, 제동은 +15.281% / +16.062% / +14.846%다. 음수는 악화다. 제동 기록 오차 감소를 실제 감속·타이어/하중 적응이나 빠른 유효 랩의 증거로 세지 않는다. 모든 축의 원자료와 matched 비교는 summary/evaluation audit에 보존했다.

두 조건의 실제 영상 방향은각25/30, 목표margin.002 통과17/30이다. 부모는25/30·13/30이었다. 새 합성은각18/18방향·목표0/18이다. 원래45개 원하는 yaw/speed±.01 명령의 평균 방향은두조건모두45/45다. 같은 Monza/Z4 GT3 소비 개발이며 새트랙 일반화가 아니다.

| 같은 upstream 상태의 원본RGB 한 TRAIN 사례 | frozen_direct margin | learn_direct margin |
|---|---:|---:|
| fine | .002897763 | .002880183 |
| visual_change_off | .002050385 | .002038285 |
| direct_gain_off | .002897763 | .002890131 |

부모의 원래RGB margin .001954729보다 두 연속 학습 조건이 커졌지만 새gain의 효과는 음성이었다. 방향과 반응 크기가 이 한사례에서 커진 것을 실제 조향 개선으로 대체하지 않는다. 이feature bundle은 upstream 가중치가 그대로인 조건에서 이전의 전체FlyVis상태/held retina 복제 원본을 hash 확인 후 재사용했다. 현재후보의 자체student prefix를 복원했다. retimed RGB이며 물리 반사실/새코너 정답이 아니다.

visual_change_off는 이번 fine motion→DN 증분을 부모로 돌린다. direct_gain_off만 새42계수를 끄고 다른 학습분을 유지한다. native_change_off는 실제 데이터로 바뀐native 증분만 되돌린다. speed_policy_off는 이전부터 이어온속도경계 전체를 끄므로 이번새보정만의제거가 아니다. synthetic native forward 변화가 있어도 synthetic native gradient/Adam이동0과 모순되지 않는다.

## 몸 감각·실행 경계

두 조건 모두 고정 factual command trace와 자연 actor-policy에서 yaw·speed 감쇠 방향은 45/45다. 변화 크기>1e-5 조건은 fixed yaw45/speed43, actor yaw44/speed17이다. 같은 영상/native/clock/prefix에서 감지 yaw/speed만 ±.01 바꾼 국소 검사이며 물리 폐루프 안정성 증명이 아니다.

3속도×2온도×2모델×속도 경계 on/off의 24개 독립 고정 입력을 301결정·40ms 간격으로 검사했으며 마지막 2초의 action 변동 최대는 0이다. 실제 타이어 물리나 전체 안정성의 증거가 아니다. strict CPU loader에서는 원본 native byte/camera/body join 128결정×2조건의 예측/근육 최대차0을 확인했다. 전체 3기록을 독립 CPU runtime으로 다시 실행한 증거로 확대하지 않는다. 새 모듈·strict schema·후속 목적함수의 18검사가 통과했다.

전용 desired yaw는 두 모델의 모든 48,866결정에서 0이었다. 영상은 별도 4DN 전류와 visual ports로 조향에 기여하므로, 이것만으로 시각 부재나 주된 오차 원인을 단정하지 않는다. 새 yaw 명령 회로를 이유 없이 필수화하지 않는다.

## 다음 수정의 근거: 보호 기준의 빈틈

현재 recording_losses의 보호 anchor는 원래 fine이다. 직전 .723 부모로 바뀌지 않았다. 최종 고정 TRAIN 궤적을 원래 selected window/fit/pre mask와 1.02 비율에 대입하면, 진입 전 33개 window 중 5개·17사건은 직전 부모를 기준으로 할 때 검출되지만 원래 fine 기준은 검출하지 못한다. 원래 기준 위반 14개 window와 부모 기준 위반 14개 window는 같은 집합이 아니다. 이는 보호 기준의 빈틈이며, 실제 online 학습 악화의 유일한 원인이라고 증명한 것은 아니다.

새 dual_reference_recording_losses 원형은 supervision/원래 anchor 항을 유지하면서 각 all/pre 축의 상대 오차 위반을 두 reference 중 더 엄격한 것으로 고른다. 두 reference가 같으면 원래 forward와 gradient를 유지하는 검사를 통과했다. 실제 TRAIN 14592:14624 구간에서 원래 보호0→새 보호.5522136이며 새 loss gradient의 FD 최상대오차는 .04963%다. 이 함수는 soft 목적함수이며 품질 gate나 물리 보장이 아니다. **이 목적함수를 사용한 새 학습은 아직 하지 않았다.** 초기 diagnostic bootstrap의 잘못된 plan key 참조 실패도 source와 receipt로 보존했다.

다음은 보존한 .723 부모/정확한 Adam에서 gain을 고정0으로 두고, 원래 보호와 두 기준 보호를 비교하는 것이다. 같은 epoch를 그대로 또 돌리거나 포트 용량을 무작정 늘리지 않는다. 두 원본 기준, 각 native 누적 bound, 실제 연속 상태, 몸 감쇠와 가감속을 함께 평가하고 실제 제한된 학생 폐루프 조향으로 검증을 이어가야 한다. 현재 두 새 후보는 미승격이고 current read-only actor와 기존 실패 기록은 그대로다. 빠른 유효 완주/라인/랩타임, 타이어 온도·자세·회전·가속도·하중·접지/셋업·차종·검증된 상대풍·변속 및 한계 주행이라는 전체 목표는 active다.

원본은 runs/direct-scene-gain-learning-20261105의 plan/reference/validation, 두 fit/evaluation·optimizer audit·matching audit·evaluation audit, body/RGB/held probes·runtime validation, anchor-protection diagnostic·dual-reference prototype validation, summary-01.json 및 source-receipt-final-01.json이다. texture915320293은 이번 평가로 소비했고 915320317은 예약만 했다. 학습·최종 평가·후속 검증의 원래 실행 핸들은 exit0을 확인했다. 실패 bootstrap의 exit1은 별도로 보존했다. 과거 shadow-01의 잔여 GPU/launcher 종료 문제도 별도로 관측하며 현재 학습 진행이나 정상 종료로 세지 않는다.
