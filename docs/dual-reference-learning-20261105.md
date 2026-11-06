# 두 기준 보호 목적함수의 실제 연속 학습 — 2026-11-05

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

두 기준 보호 목적함수를 실제32597결정·395업데이트 학습했다. 기록 재생에서 진입 전 조향 RMSE가 원래 보호 대조군보다2.362%/5.521%/3.209%, 직전 부모보다1.284%/1.170%/1.144% 줄었다. 반면 전체 감독 구간 조향 오차는 대조군보다.972%/1.274%/1.238% 커졌다. 이는 한계 주행이나 전체 운전 품질 성공이 아니다. 현재 새 체크포인트는 연구 후보이며 실제 학생 제어0·미승격이다.

## 변경과 대조군 재사용

새 목적함수는 감독·원래anchor 항을 유지하고, 각 all/pre 축에서 old-fine과 즉시부모 두 상대오차 위반 중 더 엄격한 값을 보호한다. soft 목적함수이며 hard 품질 보장이 아니다. parent prediction은 TRAIN loss에만 들어가고 actor 입력에는 들어가지 않는다. runtime은 이 parent prediction 배열을 읽지 않는다.

출발은 보존한.723 부모(input_adjoint, SHA723286d6f79401e266d24a7c384680f95cf3edbabf7807139bfb39177dea6812)와 기존4 Adam1975/1655/662/662다. 실패한 direct gain42개는 양쪽 모두0으로 고정했다. 기존17446개만 학습하며 optimizer에 남은42개는grad=None·step0이다. native/motion/speed의 원래 누적bound·power직접민감도2000·TRAIN정규화·chemical graph·근육 판독은 유지한다. 근육6값은 modeled MN 군 평균이며 실측 근육 힘이 아니다. 생리적 비행 기능 전체나 새 실측 연결을 구현했다고 주장하지 않는다.

원래 보호 대조군은 runs/direct-scene-gain-learning-20261105/frozen_direct-fit-01의 a47338d90bd994f86a934df0234478552adc41d837b002c811a0579b614272e8이다. 동일seed782011·데이터·기존목적함수항·395작업으로 이미 학습한 원본이다. 새로 대조군을 학습하거나 전체48866결정을 재생하지 않았다. control-reuse-01.json이 CP/학습/평가/prediction hash를 고정한다. 새texture915320317에서는 두 모델을 평가했다. 이전texture915320293의 점수를 새texture 점수로 사용하지 않았다.

## 학습과 실행 증거

초기1024출력차0·기존4Adam moment/step 일치, 실제motion gradient FD, 가상영상 native/speed grad=None·Adam이동0을 확인했다. source validation의 FD 수치 전체를 보존했다. 이후 실제학습에서 부모prefix 주입0·episode내숫자상태연속을 유지했고 gradient detach만 사용했다. 실제감독301/명령30/영상교환10/합성54로 총395업데이트다.

모든395 Adam proposal과 최종moment를 독립 재구성해 최대차0을 확인했다. 대조군과 초기모든파라미터차0·모든작업순서가 같았다. 첫clipped gradient/보호값차이는 update30의real이며 그전 파라미터 이력도 일치했다. 새gain은끝까지0이다. 최종Adam2370/1986/993/993/0, motion원점대비norm.5962704, native원래기준delta norm.0497139다. 최종CP SHA는8de4aad03027e36dc8dcd2655745f6a3769af8ee254f9095e2cdd6ef9d0c6099이다.

학습원래핸들11947 exit0, 새합성GPU55181 exit0(명시적drain/release), 전체평가34630 exit0, optimizer/matching74061 exit0, held/runtime67296 exit0, 수정한control fresh49208 exit0, body probe39512은첫body성공뒤RGB bootstrap TypeError로exit1, 수정한RGB21937 exit0이다. 평가·보호감사51693 exit0도 확인했다. 오류 실행을 정상종료로 바꾸어 기록하지 않는다.

## 최종 기록 재생

새모델48866결정의3기록을 자체연속상태로 재생했다. 아래는 같은fit mask에서 비교한 RMSE 감소율이다. 전체는 전체감독구간, 진입전은그중pre subset이다. 음수는악화다. 냉간/고온/이미소비한Monza개발기록순서이며 새트랙 일반화가 아니다.

| 기준 | 진입전 조향 감소율 |
|---|---|
| 재사용 원래보호 대조군 대비 | +2.362% / +5.521% / +3.209% |
| 직전.723 부모 대비 | +1.284% / +1.170% / +1.144% |
| 원래fine 대비 | +3.317% / +6.051% / +3.632% |

| 대조군 대비 전체감독구간 감소율 | 냉간 | 고온 | 소비개발 |
|---|---:|---:|---:|
| 조향 | −.972% | −1.274% | −1.238% |
| 가속 | +.133% | +.086% | +.311% |
| 제동 | −.012% | +.104% | −.023% |

직전부모 대비 전체조향은.643%/1.381%/1.446% 악화했고, 전체가속은1.223%/1.053%/.688%, 제동은.476%/.606%/.384% 감소했다. 원래fine 대비 전체조향은25.860%/32.577%/30.911% 크다. 원래fine의 낮은label오차와 당시 고정목표 몸 감쇠0/30의 국소부호 문제도 함께 해석한다. 특정 재생오차를 차량폐루프 성능으로 대체하지 않는다.

고정6-MN 판독을 독립NumPy로 재구성한 출력 최대차는3.09e-7 이내이고 RMSE차는5.64e-9 이내다. 45개 원하는yaw/speed 양방향 명령은모두45/45다.

## 영상과 몸 감각

같은 수용자의 body/native/clock/회로 prefix를 유지한 소비 개발 영상 교환에서 기대 방향은 25→27/30, 목표 margin .002 통과는 17→17/30이다. 새 texture317은 두 모델 모두 18/18방향·목표0/18이다. 합성 road/body/온도를 실제 타이어 물리나 생리학 실험으로 해석하지 않는다.

원본 RGB 한 TRAIN 사례에서는 같은 upstream FlyVis 전체 상태와 held retina를 복제한 기존 feature bundle을 hash 확인 후 재사용하고, 현재 후보의 자체 student prefix를 복원했다. upstream 가중치는 변하지 않았다. 새 모델의 margin은 .002252343(대조군 .002897763), visual 증분 제거 .001863603(목표 미달), native 증분 제거 .002252847, speed 경계 전체 제거 .002803313이다. 새 모델도 .002는 넘지만 대조군보다 반응 여유가 작다. 실제 물리 반사실이나 새 코너 정답은 아니다. visual_change_off는 이번 fine-motion→DN 증분만 되돌리고, speed_policy_off는 이전부터의 속도 경계 전체를 끈다. gain42는 0이라 제거 시 변화가 없다.

고정 factual command와 자연 actor-policy의 감지 yaw/speed 감쇠 방향은 각각 45/45다. 변화>1e-5 조건은 fixed yaw45/speed43, actor yaw45/speed17이다. 영상/native/clock/자체 prefix를 고정하고 현재 및 각 view body의 yaw/speed만 ±.01 바꾼 국소 검사다. 물리 안정성 증명이 아니다. 대조군은 같은 CP의 이전 원본 결과를 재사용했다.

이전 texture293의 고정 직선 입력으로 3속도×2온도×속도 경계 on/off의 12조건을 검사했고 마지막 2초 출력 변동 최대는 0이었다. 대조군의 동일 12조건도 이전 원본을 재사용했다. 실제 온도 적응이나 차량 수렴 증명이 아니다. strict CPU loader의 원본 native byte/camera/body join 128결정에서 학습 경로 대비 예측/근육 최대차0을 확인했다. 이를 전체 3기록의 독립 runtime 재생으로 확대하지 않는다. 보호 목적함수 5검사도 통과했다.

## 보호 결과의 한계와 실패 보존

같은 TRAIN 33개 진입 전 window의 최종 고정 궤적을 진단하면 원래 fine 기준 위반은 14→12, 즉시 부모 기준 위반은 14→8이다. 원래 기준이 놓치는 부모 기준 퇴행은 5window/17사건→5window/21사건으로 모두 해결되지 않았다. 이는 원래 기준에 새 궤적을 대입한 결과이며, 새 dual 항이 부모 위반을 누락한다는 뜻은 아니다. 새 항은 부모 위반도 보지만 soft이므로 위반이 남는다. 학습 때의 변하는 가중치·상태와 최종 고정 궤적은 다르므로 이 진단만으로 학습 원인을 단정하지 않는다.

control-reserved-01은 진행 기록의 time 미정의로 실패해 source/result를 보존했다. 수정 준비의 PYTHONPATH 누락에 따른 재실행 FileExistsError도 원래 도구 출력에 남았다. 원본을 덮어쓰지 않고 control-reserved-02를 실행했다. RGB bootstrap은 옛 learn_direct 키워드가 남아 TypeError로 run 생성 전에 실패했고 별도 source/receipt를 보존했다. 수정한 RGB 본 실행만 최종 결과로 사용한다. 이 실패들은 모델 재학습·제어·감각 대체를 발생시키지 않았다.

## 다음 실제 주행 검증

이번 결과는 진입 전 기록 오차 개선이며, 전체 조향·특정 영상 반응 여유·일부 window의 퇴행도 남는다. 무조건 승격하지 않는다. 다음은 이 연구 후보의 현재 NativeFlightRuntime 출력3값(canonical steering/throttle/brake)을 원본 학습 라벨·축 교정의 단위/부호/배율과 대조하는 것이다. source clock에 묶인 최신 출력 전달, 준비 구간, 중립/정지, 실패 시 종료를 검증한 뒤 유한 학생 폐루프로 이어가야 한다.

기존 pilot_ac_native_student.py는 MultiViewCircuitController를 사용하므로 그대로 현재 통합 flight actor의 주행 기능이라고 부르지 않는다. VJoySession/guard 같은 공통 경계는 재사용할 수 있지만 현재 runtime으로 연결해야 한다. head의 canonical3 출력에 flight_actuator.canonical_controls의 intent2 codec을 다시 적용하지 않는다. 기존 CSP lifecycle/background 인터페이스만 사용하고 desktop 입력을 자동화하지 않는다. 지도/world pose/정답 라인은 교사·guard·평가에만 두고 actor에는 영상과 허용된 몸 감각만 준다.

차체 기울기·회전·가속도·관성·하중·접지/타이어 상태 되먹임, 다른 타이어 온도·셋업·차종, 검증된 상대풍·변속, 빠른 유효 완주·레이싱 라인/랩타임과 한계 주행은 미완료다. 학생 제어·simulator 변경 API·desktop 자동화는 이번 턴0이며 목표는 active다. 이번 전용 run의 학습·감사·후속 검증은 종료했고 과거 shadow-01의 잔여 종료 핸들은 별도 관측으로 남긴다.

원본은 runs/dual-reference-learning-20261105의 plan·control-reuse·validation·dual_guard-fit/evaluation·각 audit·held/runtime/body/RGB 검증·보호 진단·summary-01.json·source-receipt-final-01.json이다. texture915320317은 이번 평가로 소비했고 915320341은 예약만 했다.
