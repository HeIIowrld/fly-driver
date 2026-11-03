# 다음 작업을 위한 현재 문맥

최신 [센서 저장 지연 분리·새 코너 수집](flight-publisher-worker-and-corpus-20261029.md): 게임 콜백 안의85.4ms 파일 쓰기를 실측해 최신1개 대기 슬롯의 별도 작업자로 옮겼다. 실제 배포/정차8묶음/3초75·30초740관측과 원시 차0·80ms 초과0, 몸16→21 보존, 실제 파리 시각망 동결 GPU 재생을 통과했다. 작업자 저장 최대110ms 동안 콜백 발행 최대0.34ms였으나 모든 공백의 해결을 주장하지 않는다. t1625 새 지리 구간 수집이 실제 실행 중이며 첫transit737관측/누락0과 독립 감사를 완료했다. TRAIN0/개발0, 학생 미적용/optimizer update0이다. 아래는 당시 상태 기록이다.


이전 t1597 기록 [120FPS 실측·수집 실패·내부 지연 계측](flight-framecap120-and-publisher-timing-20261029.md): t1577에서 실제120FPS와 새 엔진을 적용했다. 30초737관측/누락0/원시차0 및 동결 시각망 재생은 완료했으나 t1593 첫transit이 카메라 stale로 실패해 새 TRAIN0/개발0이다. t1596 전체414묶음 감사의 복사→ACK 최대6.3ms로214ms 공백을 설명하지 못한다. 내부 Python 센서의 단계별 최대 지연 계측을 추가했고 관련44검사는 통과했으나 게임에는 미배포다. 학생 미적용/학습 진행 없음. 아래 과거 설정·상태는 당시 기록이며 최신 판단은 연결 문서를 따른다.


이전 t1573 기록 [새 전방 눈 주행 검증·수집 실패](flight-exterior-stream-qualification-and-corpus-20261029.md): revision3 일반 스트림과 기준 native 패킷 priming을 구현했다. 3초74관측/30초739관측·영상누락0·원시차0·80ms 초과0, 실제 초파리 시각망 동결 GPU 재생/몸21 보존까지 완료했다. 관련278검사/Ruff 통과. 후속 geographic 수집은 개발1/학습0에서 첫TRAIN의2.132초 엔진 갱신 공백으로 종료1이다. 실패는 fit 불가/학생 미연결이다. 실제 약553FPS인 무제한 설정의 별도120FPS 비교안을 준비했지만 실제 설정 변경/엔진 재시작은 아직0이다. 다음은 기존 native AI-off 소유권 가드를 새 상태에서 확인하고 별도 FPS 조건을 검증하는 것이다. 아래 과거 source/generation·미구현/다음 작업은 해당 시점의 기록이다.


갱신:2026-10-29,Asia/Seoul. 새 실행 전에 원본 결과와 실제 프로세스를 다시 확인한다.
읽는 순서: README → PROJECT_CHARTER → 이 문서 → research-plan.

## 반드시 보존할 대전제

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다.
차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.**

일반운전모델·그래프통과·부분기능성공으로대체하지않는다. 몸제어기준은비행이다.
실측연결·저자가중치·기능근사·차량대응·학습한경계를구분한다.
actor는영상과허용된몸감각을받고지도/좌표/절대heading/정답라인은교사·보상·평가용이다.
설명자료는Markdown. HTML/설명사이트를만들거나갱신하거나열지않는다.
게임창활성화/클릭/키보드자동화금지. 검증된CSP/센서/vJoy경로만사용한다.
다른llama-server/다른작업프로세스를종료하지않는다.

## 현재 결론

최신 [실시간MMF/새코너자료](ac-live-mmf-v3-and-spatial-data-20261029.md): v3metadata/ACK실시간연결·원본slot보존·247검사/Ruff 완료.
t1520배포/current generation `4525688ULL:0.0012764`, Lua SHA `a9f583e54da981c5e700b186d51af09fdb9249aa6112cf89a87608cd05940386`.
t1521 3초74관측/누락0/58.91kmh, t1523 30초739관측/누락0/60.93kmh, t1522/t1524원시배열·실제slot차0.
t1526 새geographic corpus는transit3/train2완료뒤episode05가stale로실패했다(collector16119종료1). 새development0개/전체task실패보존.
완료train03/04의1479관측전체를t1527partial diagnostic에서감사:원시차0/80ms초과0/바퀴이탈0/역할변경0.
새범위0.276142~0.449609, CSP보고조향−203.279~+67.919까지다. 두train모두시작3초뒤실제brake양성0이며제동학습자료로부르지않는다.
episode05는504관측/누락1. t1529sourcegap179.748ms/게임17frame진행/metadata수신3.597ms/저장9.784ms라전송만으로안정성완료가아니다.
t1530몸16재생1479완료. t1531 실제Flyvis시각망GPU재생739/740완료(87571종료0),각26.43/26.21초,peak5.515GB,저자가중치불변/update0.
현재아직새학습/학생배치는0이고t1473시각조향개발실패는그대로다. 모든소유job terminal,마지막paused/fullbrake/0.009475kmh.
**다음우선순위는새train2+원래역할의기존자료로앞길보조표적·시각경계를학습/검증하고고정운동회로에연결하는것이다.**
신경캐시는local `runtime-runs/20261029-t1531-spatial-visual-replay/episode-03-train,episode-04-train`.
새dev는없으므로기존known development를유지한다. 기존0..30goal/체크포인트를60으로재해석하지않는다. 실패episode05/미실행dev는fit에넣지않는다.
NAS `runs/ac-live-mmf-spatial-data-20261029-t1532`;대용량raw/신경응답로컬유지. 아래8정차전용기록은이전배포시점이다.

최신 [공유메모리 정차 자격](ac-paused-mmf-qualification-20261029.md): separate metadata/ACK map 구현·230검사/Ruff 완료.
t1515 paused원자배치 종료0, 새generation `4525688ULL:0.001221`, Lua SHA `4baf7f0b6293cca9e5b24a3ed941e6ac76cab8904a630ae6799d72ed9886c342`.
t1516실제정차8묶음/32화면이동일shotPNG와차0, frame metadata파일0/ACK파일0. t1517독립원시재생도차0/종료0이다.
Python소유ACKmap/읽기전용producer map, sequence/SHA/request/generation 검증. 네 GPU복사완료뒤한번ACK, 실패그룹ACK0.
실행Lua새MMF는8정차묶음전용이며 기본recorder는아직파일경로다. request/terminal/PNG비교파일은남는다.
현재는주행MMF자격/635ms원인해결/새코너학습자료/파리조향성공을주장하지않는다. 모든소유job terminal/foregroundfalse/pausedfullbrake.
다음은별도v3실시간stream에연결하고원본metadata·ACK를bounded보존/종료후hash감사하는작업이다.
새소스/계약고정뒤3초→30초주행검증,이후geographic코너수집/파리회로학습. 과거실패plan/sourcehash를재사용하지않는다.
NAS `runs/ac-paused-mmf-qualification-20261029-t1518`,이번정차시험raw전체보존. 학생미배치·t1473시각기여실패는그대로다.

최신 [구간별 코너 수집/원본 공백](flight-spatial-corpus-and-source-interruption-20261029.md): geographic role수집기/73검사/Ruff 완료.
수집 전 train0.27~0.63/development0.67~0.84/나머지transit를 선언했다. 기존 검증/평가 위치를train에 넣지 않는다.
t1508 collector session32140/자식은 종료1. 첫transit가0.903948→0.924132/최고60.8644kmh/238관측/누락12뒤stale 실패했다.
새train/dev자료0개,fit금지. t1509원시재생차0/370묶음저장완료이나최대관측gap565.972ms이다.
t1510GPUcopy→ACK최대5.648ms/render→ACK상한43.109ms. t1511publication지연최대2.841ms/대기열1,저장최대595.828ms.
t1512원본frame309→310시각gap635.352ms/게임프레임7개진행. 호스트가다음metadata자체를기다린증거이며공통원인은미확정이다.
새async저장경로의단일30초성공을다구간안정성으로일반화하지않는다. 실패batch/같은설정재수집을반복하지않는다.
다음은설치CSPSDK에서확인한별도소유memory-mapped metadata/ACK경로의opt-in 구현·검증이다. 아직미구현/미배포.
FPS_CAP_MS0도확인했으나변경/재시작은0이다. 모든소유job terminal,paused/fullbrake/약0.009925kmh.
NAS `runs/flight-spatial-source-interruption-20261029-t1513`;원본raw로컬유지.파리시각조향실패·학생미배치는그대로다.

최신 [비동기 영상 저장](ac-async-frame-persistence-20261029.md): opt-in immutable RAM stream v2를 구현하고 관련66검사/Ruff를 통과했다.
t1499 3초59.05km/h/74관측/누락0, t1500 전체 원시 재생차0. t1502 30초교사61.1368km/h/739관측/누락0,
t1503 네 영상/몸/시각 배열 재생차0·1,000묶음 저장완료 검증. 관측 최대간격48.5582ms/80ms초과0이다.
t1505에서 모든1,000묶음 publication이 저장완료보다 먼저였고 대기열최대1, receipt→publication최대3.402ms였다.
기본legacy경로/150ms·110ms 기준은 그대로다. calibration/fit금지, action codec/정답/학생 자격은 미검증이다.
t1473 시각 조향 실패·학생 미배치는 그대로다. 이번 구간은 조향 변화가 작아 다양한 코너자료가 아니다.
모든 새 작업 종료0, 마지막paused/AI-off/camera-off/probe-off/gas0/brake1/약0.00968kmh.
다음은 경로별 train/dev를 사전 선언한 실제 좌우 코너·횡오차·다속도·제동 자료와 시각 조향 학습이다.
NAS compact: `runs/ac-async-frame-persistence-20261029-t1506`; 대용량 원시 ZIP은 로컬 유지/해시 참조.
아래 t1497까지의 “아직 구현 전/30초 실패”는 당시 결과이며, 새 opt-in 경로는 이 단락이 최신이다.

최신 [묶음 복사/저장 직렬화 지연](ac-batched-copy-and-serial-persistence-20261029.md): 관련58검사/Ruff와 같은 immutable PNG40화면 비교차0.
정지 그룹복사 중앙값5.130→3.038ms. 새 batch는 opt-in이며 기본legacy경로/150ms·110ms 기준은 유지했다.
t1487은29.39초/717관측/61.17kmh 뒤stale 실패. 모든870그룹 GPU복사 최대5.36ms/Map3.02ms라 GPU복사만으로 전체지연을 설명할 수 없다.
진단journal을종료후저장하는 v2로도 t1494는5관측/0.259kmh에서실패했다. 3초시험t1485/t1491은각74관측/누락0/원시재생차0이다.
t1495에서 frame87 저장·전처리141.067ms 동안 이미 렌더된 frame88을138.424ms 기다리게 했고, 이전 처리 종료1.466ms 뒤에야 읽은 것을 확인했다.
다음 구현은 별도 버전의 immutable RAM영상 수신과 bounded 비동기 저장 분리다. 아직 구현 완료가 아니다.
새 receipt를 실제 준비시각에 기록하며 옛 run 재시각화/가짜 durable표시는 금지한다. writer완료/hash/원시재생 전 학습 승인·학생 제어는 없다.
모든 소유 job terminal. calibration/fit금지와실패보존. 마지막paused/AI-off/camera-off/probe-off/0.00972kmh.
NAS compact: `runs/ac-batched-copy-serial-persistence-20261029-t1497`; 원시 영상 로컬유지.

최신 [조향 종료/60kmh 실제 시험](flight-steering-terminal-and-speed60-pilot-20261029.md): 모든 소유 학습/수집/감사는 terminal이다.
t1473 PID40084/session31336 종료0,24/24승인/손실0.658804→0.651948. 조향0.067739/0.099931은 평균/0기준보다 나쁘고 시각평균치환이 더 낫다. 학생 미승격이다.
t1478에서60 소스를 배포했고 현재 generation은 `4525688ULL:0.0011667`, source/deployment 일치다.
t1479 3초58.69kmh/74관측/누락0 통과. t1481은61.14kmh에 도달했으나19.01초/466관측/누락5 뒤camera stale로 실패(도구90426 종료1)했다.
role calibration/fit금지. t1482 원시감사 완료는30초 성공이 아니다. 카메라frame553 복사확인이215~219ms 늦었고 게임프레임은54개 진행했다.
당시 다음 작업은 capture/ack 지연 분리였고, 후속 t1495에서 저장 직렬화 사례를 확인했다.150ms/원래시각/실패를 유지하고 같은코드 재수집을 반복하지 않는다.
마지막 paused/AI-off/camera-off/probe-off/error없음,0.00983kmh. NAS t1477 조향terminal/t1483 고속partial.

[60km/h 교사 경로](ac-speed60-profile-preparation-20261029.md)를 CPU에서 구현/224검사 완료했다.
당시 **미배포** 상태여서 작업폴더 Lua와 게임 Lua가 달랐다. 현재 t1478 배포로 일치한다. t1473 조향 학습을 같은 PID/핸들로 끝내고
idle 상태를 다시 검증한 뒤 원자 배치/generation 확인이 필요하다. source 일치 검사를 우회하지 않는다.
새60은80km/h·80/720m 한계, 기존30 한계는 그대로다. 고속 학생/교사 실제 주행 성공 주장은 없다.

최신 [조향 우선 실험](flight-steering-reflection-and-motor-training-20261029.md): t1471 광학반사 TRAIN재생 완료,
원본 신경/scene 재현오차0·저자 가중치 불변. t1472 조향/앞길 사전학습 완료지만 반사 조건은 개발에서 악화했다.
원본만 조향RMSE0.055968/0.101012, 반사추가0.091628/0.126199; 평균/0조향 기준 미달이다.
**t1473 조향 학습/평가는 종료0이며 개발 미통과다. 과거 PID40084/session31336은 terminal이다.**
중간 확인 당시3/24승인/손실0.658804→0.655661. 새 시각 latent와 power/운동 그래프/몸 readout을 고정하고
DNa04/05 조향 출력층만 학습한다. 사전학습 출력→액추에이터 우회는 없다. 원래 시간/상태/64관측detach 유지.
출력 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1473-pretrained-real-flight-steering`.
당시에는 같은 PID/핸들의24회/개발평가 종료 확인이 다음 작업이었고, 이제 완료했다. timeout으로 재제출하지 않는다.
아직 학생 제어/고속 운전/레이싱 승격 없음. NAS terminal+진행snapshot: `runs/flight-steering-reflection-and-motor-training-20261029-t1474`.

사용자 최신 정정: **30km/h는 레이싱 학습 기준이 아니다. 일반 주행 고정30 기본값은 폐기한다.**
새 계획은 `configs/flight-racing-speed-curriculum-20261029.json`: 교사60→100km/h 단계 검증 뒤 직선가속/코너제동/랩타임 최적화다.
기존 goal0..30 입력 상한도 확인했다. 새 `racing_goal.py`는 외부 속도 상한과 레이싱 목적을 명시하고 구형 체크포인트 해석을 거부한다.
10검사/Ruff 통과. 새 고속 actor/실제 고속 주행은 아직 미검증이다. 교사 곡률 기반 속도는 표적용이며 actor goal로 우회 전달하지 않는다.

최신 [실제 제동/하중 반응](flight-actual-brake-feedback-20261029.md): 새 유한 AI 제동 개입 경로를 구현했다.
관련219검사/Ruff 통과 후 t1464 paused hot reload, 현재 generation은 `4525688ULL:0.0011165`다.
t1465 738관측/누락2, t1466 원시재생차0이나122.284ms 공백1회로 연속학습 자격false다.
t1467 독립 native 감사: 제동 전30.8319→1.5초내0.06290km/h, 최대brake1, 양성38표본,
앞바퀴 하중 비율 평균0.51164→0.62797. API time=.4를 실제 제동 지속시간으로 가정하지 않는다.
개발 진단/fit금지/강제 개입이며 기존 자연 AI 라벨러는 새 프로파일을 거부한다.
30km/h 목표에 강제 정지를 그대로 imitation 정답으로 묶지 않는다. 다음은 목적이 명시된 감속 과제와 몸 제어 식별이다.
파리 모델 가중치/학생 배치는 그대로다. 마지막 paused/AI-off/camera-off/probe-off,속도0.00923/error없음.
계측/감사 종료0. NAS: `runs/flight-actual-brake-feedback-20261029-t1468`.

당시 t1462에서 일반 주행30km/h 기준을 잡았으나, 이후 사용자 지적으로 폐기했다. 30km/h는 계측용이다.
t1462에서 30초 AI 교사 시험을 종료0으로 수행했고 실제 최고31.7119km/h를 확인했다.
735관측/카메라 대기 누락4회이며 자료/학생 정책 승격은 없다. 종료 후paused/AI-off/full brake다.
원본: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1462-user-requested-speed30`.

최신 [새 구간 감속 계측](flight-new-region-speed-transition-20261029.md): t1455~t1460 모두 종료0.
세션 재시작 없이 기존 v5 실패 개발 위치를 이어서 3초/30초 AI 진단을 했다. 원래 개발 역할 유지/fit 금지다.
30초 관측739/누락0/원시재생차0/최대간격56.238ms. 20→15km/h 지시 뒤 회복 프로토콜은 통과했지만
주행 후반 brake>0.01 표본0이라 실제 제동 자료가 아니다. 마지막 주행속도16.43km/h, 정확15유지도 미검증이다.
3초 준비 시험의83.64ms 공백1회와 raw audit의continuous_training=false를 보존한다.
t1460에서paused/AI-off/camera-off/probe-off,0.00955km/h를 확인했다. 학생 제어/UI 조작은0이다.
다음은 현재generation의 vJoy/페달/조향·제어권을 검증하고 실제 brake 요청·응답이 있는 교사 프로파일이다.
자료의 개발 역할을 바꾸지 않는다. NAS compact 근거: `runs/flight-new-region-speed-transition-20261029-t1461`.

최신 t1449 [이력 단위 비선형 검증/v4 범위 감사](flight-preview-history-generalization-20261029.md) 완료.
TRAIN만 사용한 6회 whole-history 검증 × 2구조 × 200epoch, GPU 2,400업데이트다.
학습 방향 오차는 작아졌으나 빼둔 이력의 전체/3초 후 집계는 두 구조·모든 거리에서 평균보다 나빴다.
업데이트 경로 부재보다는 과적합/구간 일반화 문제가 남는다. 이것은 비교용 probe이며 fly actor가 아니다.
v4 6완전이력/4,432관측 캐시(t1451,몸16)와 preview(t1452)를 만들었지만 새 actor 학습은 없다.
t1453에서 v4 TRAIN 이동 표본 전부가 기존 TRAIN 경로 0.283m 이내임을 확인했다.
재촬영을 새로운 경로 다양성으로 세지 않는다. 다음은 원래 구간 시작/선택을 감사하고
경로 묶음별 역할을 사전 고정한 새 좌/우 코너·다속도·실제 제동 자료를 준비한다.
모든 실행 terminal 0. NAS: `runs/flight-preview-history-generalization-20261029-t1454`.

후속 t1447 [겹눈 공간 표현 비교](flight-preview-spatial-comparison-20261029.md)도 완료했다.
원래 GPU 경로로 8이력의 특징 오차 0을 확인했으나, 8×8/원위치 강도 등 6개 표현 모두
개발 두 구간·모든 거리의 방향 예측이 평균/0방향보다 나빴다. 3초 후도 평균보다 나빴다.
t1446 NumPy 재현 실패는 보존했고 tolerance는 늘리지 않았다. 실행은 종료 0이며 actor 미배치다.
다음은 TRAIN 이력 단위 내부 검증으로 비선형 공간 표현과 구간 편향을 분리한다.
역할 고정 코너·속도·제동 자료 계획도 필요하다. 근거: `runs/flight-preview-spatial-comparison-20261029-t1448`.

추가 t1443/t1444 [앞길 방향 검사](flight-road-preview-diagnosis-20261029.md)도 완료했다.
기존 시각 특징만 받는 고정 선형 probe는 개발 두 구간의 5/10/20m 방향 모두 평균·0방향보다 나빴다.
3초 후에도 실패했으며, 개발 표적은 모두 ±5도 이내여서 코너 일반화 검증 범위도 부족하다.
원래 역할/정답은 유지했다. 다음은 공간 구조를 보존한 시각 표현 비교와 사전 역할을 정한 코너·속도·제동 자료다.
증거는 `runs/flight-road-preview-diagnosis-20261029-t1445`. 새 actor 배치는 없다.

**새v7의24회공동학습/개발평가는완료했지만gate=false이며게임미배치다.**
[현재모델변경·전체수치·실패·다음작업](flight-embodied-command-learning-20261029.md)를먼저읽는다.

- 기존v6는학습대상이조향4포트뿐이고power경로는고정이었다.
  몸16에는명시적속도/바퀴피드백이없었고기능적시각자기운동6채널은zero/invalid다.
- 새v7은원래모델을고정하고공학적상위시각경계에서DNa04/05조향과DNg02 power current를조절한다.
  기존9,728뉴런/1,311,145연결·몸경계·readout은변경하지않는다. 직접액추에이터잔차는없다.
- 몸21은기존16정확보존+speedKmh/3.6/바퀴각속도4다. 차량용몸감각대용값이며실제파리감각튜닝이아니다.
  unregistered 가속/roll·pitch각속도마스크와기능적시각6채널누락은그대로남는다.
- 새turn경계는영상/목표/속도크기만받고yaw/바퀴간차이는받지않는다.
  기존몸반사·power문맥에는몸신호가남아전체모델이그단서를쓸가능성까지배제한것은아니다.
- 11관련검사와실제그래프초기drive동일/두head기울기검증을통과했다(t1435).
  prototype1433은바퀴도turn입력에있던이전구조라최종초기값으로사용하지않았다.
  t1432 float64/float32검사실패는원래float32cache변환을명시한후exact비교로해결했고원본실패를보존했다.
- t1434는같은v2 train6/개발2의8이력5,932관측에몸21만추가했다.
  시각/정답/mask/목표/원래시각은동일하다. actor에교사조작·pose를넣지않았다.
- t1437 GPU24epoch/24승인업데이트,loss0.675491→0.665344,원래상태/이전입력hold보존,
  64관측마다미분만detach했다. 고정마지막24회checkpoint만평가했다.

## 개발 실패를 축소하지 말 것

- 조향RMSE기존0.070537/0.102438→새0.068259/0.093337. 둘다0조향기준을못이겼다.
- forward전체기존0.183794/0.190467→새0.133740/0.174545이나한개발구간은TRAIN평균보다나쁘다.
- 3초후forward는기존0.011026/0.016736→새0.105075/0.102474로회귀했다.
  개발07유효578중253,3초후유효517중223에서잘못된제동을예측했다. 실제제동정답0이다.
- 전체에서는운동정보제거가조향을악화시켰지만3초후에는두구간모두오히려개선했다.
  안정주행의운동시각기여통과가아니다. 치환대조를실제뉴런lesion으로부르지않는다.
- 알려진개발자료이며blind평가가아니다. 고정30km/h/제동0으로다른목표속도·제동·셋업을판정하지않는다.

## 실제 실행 상태와 위치

현재 알려진 새 학습/평가/수집 job은 모두 terminal이다. 재개 시에도 다시 확인한다.

- t1437학습:도구session12100/PID41436 종료0. 폴더
  `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261029-t1437-embodied-joint-training`.
- t1439개발평가:도구session19580 종료0. t1440 잘못된제동raw예측감사도완료했다.
- t1436훈련실행소스사본, t1438평가사본. 기존사본을수정하지않는다.
  다음평가가필요하면1438의`python -m scripts.evaluate_embodied_flight_navigation --training <t1437> --output <새폴더>`를사용한다.
- NAS전체모델증거보존:`runs/flight-embodied-command-learning-20261029-t1441`.
  로컬원본도유지한다. 소스사본은설정실행소스이며프로세스메모리소급검사를주장하지않는다.
- 모든이전수집handle도terminal이다: v4 session50310종료1/감사99226종료0,
  v5 session5965종료1/감사62380종료0. 경과시간만으로새실행을시작하지않는다.

## 수집/부품의 별도 근거

- v5 t1424는8lease5,911관측완료후index8개발이70관측/wait9회/카메라신선도로실패했다.
  완료train/개발7개만t1428에서감사했다. 평가5는의미적감사/fit에서제외했다. 나머지15개미실행이다.
  원본은로컬runtime-runs에유지하며대용량v4/v5raw전체가NAS패키지에다복사된것은아니다.
- 선택적비동기픽셀캐시는38검사/동일shotPNG8×4/3초74관측·원시배열차0을통과했으나
  후속v5도신선도로실패했다. 전체수집안정화성공으로부르지않는다. GPU재생은수집종료뒤시작했다.
- v4 t1412는7lease뒤index7카메라신선도로실패,완료6이력4,432관측의GPU/몸/전체시간감사는t1417에서완료했다.
  t1425 train5/개발1봉인도완료했고제동정답은0이다.
- 실제장면의알려진카메라회전4조건부호와첫영상유지대조는통과했지만크기/밝기오류가남는다.
  [감각반응범위](flight-spatial-and-actual-camera-motion-20261029.md).
- 별도빠른몸피드백은6km/h교란12회/6쌍모두yaw IAE감소·중앙값13.17%였다.
  [원본요약](../runs/flight-fast-physical-yaw-report-20261029-t1285/summary.md).
  v7은기존50ms모델이며이5ms경로자격을상속하지않는다.
- 마지막엔진기록은36644/640×360/비전면/CSP generation4525688ULL:0.0017325다.
  다음조작전에fresh heartbeat·native정지/full brake·소유권·실행Lua일치를다시확인한다.
  native AI표시와CSP AI-off 불일치는아직별도문제다. 실제새actor소유권자격을자동승격하지않는다.

## 다음 작업

1. 도로/차량기준앞길방향·곡률preview의중간시각목표를분리한다.
   camera pose/교사미래궤적은표적생성용으로만사용하며actor에는넣지않는다.
   4×4공간요약정보손실·광도초기과도를함께검사한다.
2. 가속/정속/실제감속·제동과여러속도목표로power경로를검증한다.
   잘못된제동예측을clipping하거나cleanup제동을주행정답으로바꾸지않는다.
3. 조향/power운동readout결합을검사하고전체/3초후성능을함께판정한다.
4. 그후에만빠른몸/시각통합·실제학생폐루프·타이어/셋업·유효완주·빠른라인/랩타임을검증한다.
   일반운전모델로대전제를대체하거나완료기준을줄이지않는다.

## 실행 관례

- cwd는원래NAS프로젝트다. CPU venv는`C:/Users/jeff4/AppData/Local/flybrain-review-20261027/venv`.
  GPU는BASE Python312에`C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-env/Lib/site-packages`를
  `site.addsitedir`로 추가하고 `sys.path` 맨 앞에 둔다. 고정한 실행 소스의 `src`와 루트를 `PYTHONPATH`로 지정한다.
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`을 사용한다. GPU를 쓰는 다른 작업은 종료하지 않는다.
- 원래시각/gap/실패/신경상태를보존한다. observer timeout만으로프로세스를죽이거나다시제출하지않는다.
- 과거NPZ중복214개정리는완료했다. 고유67종과`.episode-links.json`을보존하며
  `episode_storage.load_array/resolve_payload_path`와`graph.sha256`을사용한다. 옛NPZ를전부재복제하지않는다.
- 모든과거인계원문과당시live상태는[보존연혁](CONTEXT-history-through-v7-20261029.md)에남겼다.
  현재상태판정은이문서위쪽과실제원본으로한다.
