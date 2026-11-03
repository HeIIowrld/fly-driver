"""Preserve model/data evidence and replace stale stacked status with a current handoff."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

local = Path('C:/Users/jeff4/AppData/Local/flybrain')
base = local/'runtime-runs'
destination = Path('runs/flight-embodied-command-learning-20261029-t1441')
destination.mkdir(exist_ok=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


names = ['20261029-t1431-flight-command-capacity',
    '20261029-t1432-embodied-navigation-path-verification',
    '20261029-t1433-embodied-navigation-path-verification',
    '20261029-t1434-body21-flight-cache', '20261029-t1435-final-embodied-path-verification',
    '20261029-t1437-embodied-joint-training', '20261029-t1439-embodied-development-evaluation',
    '20261029-t1440-unsupported-braking-audit']
assert read(base/names[5]/'result.json')['status'] == 'completed_fixed_joint_command_training'
assert read(base/names[6]/'result.json')['complete_learning_gate_passed'] is False
rows = []
for name in names:
    source = base/name
    assert (source/'result.json').exists()
    for path in sorted(source.rglob('*')):
        if path.is_file():
            target = destination/name/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = digest(path)
            shutil.copy2(path, target)
            assert expected == digest(target) == digest(path)
            rows.append(dict(source=str(path), archived=str(target), sha256=expected, bytes=path.stat().st_size))
for name in ('20261029-t1424-teacher-corpus-v5', '20261029-t1428-v5-prefix-audits'):
    for file in ('manifest.json', 'result.json'):
        source = base/name/file
        target = destination/'terminal-data-job-metadata'/name/file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert digest(source) == digest(target)
snapshot = local/'execution-snapshots/20261029-t1436'
archive = destination/'configured-training-source-t1436.zip'
sources = {}
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(snapshot.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.json'):
            name = path.relative_to(snapshot).as_posix()
            sources[name] = digest(path)
            zipped.write(path, name)
with zipfile.ZipFile(archive) as zipped:
    for name, expected in sources.items():
        assert hashlib.sha256(zipped.read(name)).hexdigest() == expected
overlay = destination/'evaluation-source-t1438-overlay'
overlay.mkdir()
for relative in ('scripts/evaluate_embodied_flight_navigation.py',
                 'flight-embodied-navigation-evaluation-20261029.json'):
    source = local/'execution-snapshots/20261029-t1438'/relative
    target = overlay/Path(relative).name
    shutil.copy2(source, target)
    assert digest(source) == digest(target)
for relative in ('src/flybrain_drive/flight_embodied_navigation.py',
                 'src/flybrain_drive/virtual_navigation_body.py'):
    assert digest(snapshot/relative) == digest(local/'execution-snapshots/20261029-t1438'/relative)
for relative in ('configs/ac-virtual-navigation-body-v2.json',
                 'configs/flight-command-capacity-diagnosis-20261029.json',
                 'configs/flight-embodied-navigation-evaluation-20261029.json',
                 'tests/test_virtual_navigation_body.py', 'tests/test_flight_embodied_navigation.py'):
    target = destination/'contract-and-tests'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(relative, target)
    assert digest(Path(relative)) == digest(target)
histories = {}
for original, historical in (
    ('README.md', 'README-history-through-v7-20261029.md'),
    ('docs/CONTEXT.md', 'docs/CONTEXT-history-through-v7-20261029.md'),
    ('docs/research-plan.md', 'docs/research-plan-history-through-v7-20261029.md'),
    ('docs/implementation-status.md', 'docs/implementation-status-history-through-v7-20261029.md')):
    historical = Path(historical)
    assert not historical.exists()
    shutil.copy2(original, historical)
    assert digest(Path(original)) == digest(historical)
    histories[historical.as_posix()] = digest(historical)


def replace_section(path, heading, next_heading, text):
    path = Path(path)
    original = path.read_text(encoding='utf8')
    start = original.index(heading)
    stop = original.index(next_heading, start+len(heading))
    path.write_text(original[:start]+heading+'\n\n'+text.strip()+'\n\n'+original[stop:], encoding='utf8')


replace_section('README.md', '## 현재 상태 — 2026-10-29', '## 자료와 기록', '''
**시각→조향·가감속 학습은 아직 미통과다.**
새v7에서속도/바퀴되먹임과가감속학습경로를추가하고GPU24회공동학습·개발비교를완료했다.
조향은기존보다개선됐으나0조향기준을못이겼고, 정속구간에서잘못된제동을예측해미승격이다.
[수정·검증·수치·다음모델작업](docs/flight-embodied-command-learning-20261029.md).

- 원래 시각/운동 모델과 실측 연결망을 바탕으로 한 운동 회로는 고정했다.
  새 상위 시각 경계와 차량 속도/바퀴 입력은 공학적 근사이며 측정된 파리 감각 튜닝으로 부르지 않는다.
- 실제장면회전4조건의부호와첫영상유지대조를확인했지만도로판단·생리·학생주행합격은아니다.
- 별도빠른몸피드백은6km/h제한교란에서6/6효과·중앙값13.17%를확인했다.
  새v7은기존50ms회로를사용하며이빠른경로의자격을상속하지않는다.
- 최근v5교사수집은8구간후카메라신선도로실패했다. 완료자료·실패·평가역할을보존했다.
- 현재알려진학습/평가/수집작업은모두종료했다. 파리actor배치는0이다.

다음은앞길방향/곡률preview의중간시각표적과coast/제동·명령결합을검증한다.
타이어/셋업·하중·빠른몸통합·유효완주·레이싱라인/랩타임은미완료다.
상세연혁은 [당시상태원문](README-history-through-v7-20261029.md)에보존했다.
''')
replace_section('docs/research-plan.md', '## 현재 판단', '## 작업 순서와 통과 조건', '''
새v7은몸21(기존16+차량속도/바퀴5)과비선형시각→DNa04/05·DNg02명령학습경계를추가했다.
GPU24회공동학습·개발비교를완료했으나조향기준·정속제어·잘못된제동문제로미승격이다.
학습손실감소·회로실행·카메라회전반응을운전성공으로대체하지않는다.
[최신근거·결과](flight-embodied-command-learning-20261029.md).

빠른몸회로의6km/h교란효과는별도증거이며, 현재v7의50ms운동모델/시각/몸통합은아직실제주행자격이없다.
제동표적0·고정30km/h목표·같은트랙의알려진개발구간이라는한계를유지한다.
현재실행중인알려진학습/평가/수집job은없다.
과거판단·계획은 [보존원문](research-plan-history-through-v7-20261029.md)에남겼다.
''')
replace_section('docs/research-plan.md', '## 바로 다음 작업', '## 학습과 정보 경계', '''
1. 도로/차량기준앞길방향·곡률preview를읽는중간시각목표를분리한다.
   교사미래궤적/pose는표적생성에만허용하고actor는영상/허용된몸감각만받는다.
2. 제동과서로다른속도목표를포함하는가속/정속/감속자료로power경로를검증한다.
   현재잘못된제동예측을clipping해숨기지않으며cleanup제동을주행정답으로쓰지않는다.
3. 조향/power운동readout결합과coast회귀를검사하고전체/3초이후성능을함께판정한다.
4. 원래개발역할과실패를유지하며최종고정checkpoint를평가한다.
   그후빠른몸/시각통합·학생폐루프·유효완주·레이싱라인/랩타임을검증한다.
''')
implementation = Path('docs/implementation-status.md')
text = implementation.read_text(encoding='utf8')
start, stop = text.index('최신 [v7명령 공동학습]'), text.index('## 근거를 확인할 위치')
summary = '''
**현재 시각→조향·가감속 학습은 미통과이며 새 파리 actor 배치는0이다.**
[v7구현·24회학습·개발결과](flight-embodied-command-learning-20261029.md)를기준으로한다.

| 항목 | 구현/검증 범위 | 미완료 |
|---|---|---|
| 겹눈/저자시각회로 | 독립4시점45,669뉴런모델·원래시간재생·제한된실제회전반응 | 도로preview·안정주행기여·T4d contrast 등 |
| 몸감각 | 기존16보존+명시적차량속도/바퀴5의새21계약 | 실제파리감각튜닝·선형가속/roll/pitch각속도등록 |
| 명령/운동 | 고정9,728뉴런/1,311,145연결운동모델에학습경계추가·양head기울기확인 | 생리·실제폐루프·빠른몸경로통합 |
| 공동학습 | GPU24회·고정마지막checkpoint·개발2구간비교 | 조향기준·coast회귀·잘못된제동으로미승격 |
| 빠른몸피드백 | 별도6km/h제한교란6/6효과·중앙값13.17% | v7은50ms모델이며이자격을상속하지않음 |
| 최종주행 | 자료수집/인터페이스와일부몸반응검증 | 가감속/제동·하중/타이어/셋업·학생완주·빠른라인/랩타임 |

현재알려진학습/평가/수집job은모두terminal이다.
과거상태의원문은 [연혁](implementation-status-history-through-v7-20261029.md)에보존했다.

'''
text = text[:start]+summary+text[stop:]
text = text.replace('## 최신 ', '## 당시 ')
implementation.write_text(text, encoding='utf8')
Path('docs/CONTEXT.md').write_text('''# 다음 작업을 위한 현재 문맥

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
  site.addsitedir/sys.path先頭で追加し、凍結source/srcとsourceをPYTHONPATHに指定する。
  OMP_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1。外部GPU利用者は停止しない。
- 원래시각/gap/실패/신경상태를보존한다. observer timeout만으로프로세스를죽이거나다시제출하지않는다.
- 과거NPZ중복214개정리는완료했다. 고유67종과`.episode-links.json`을보존하며
  `episode_storage.load_array/resolve_payload_path`와`graph.sha256`을사용한다. 옛NPZ를전부재복제하지않는다.
- 모든과거인계원문과당시live상태는[보존연혁](CONTEXT-history-through-v7-20261029.md)에남겼다.
  현재상태판정은이문서위쪽과실제원본으로한다.
''', encoding='utf8')
plan_file = Path('configs/fly-flight-mechanism-plan-20261029.json')
plan = read(plan_file)
plan['updated_at_kst'] = datetime.now(timezone(timedelta(hours=9))).isoformat()
data = plan['next_visual_and_pedal_data']
data['currently_active_owned_model_jobs'] = []
data['currently_active_owned_data_jobs'] = []
data['embodied_navigation_v7'] = dict(report='docs/flight-embodied-command-learning-20261029.md',
    archive=destination.as_posix(), training_run=str(base/names[5]), evaluation_run=str(base/names[6]),
    learned_command_paths=['DNa04/05 steering', 'DNg02 common power modulation'],
    new_body_features=21, previous16_body_features_preserved=True,
    added_vehicle_feedback=['unsigned native speed magnitude m/s', 'four native wheel angular rates rad/s'],
    actual_fly_sensor_physiology_claimed=False, source_graph_body_readout_frozen=True,
    related_tests_passed=11, actual_graph_joint_gradients_verified=True,
    fixed_epochs=24, accepted_optimizer_updates=24, first_loss=.6754910230681282,
    final_loss=.6653443474397495, development_gate_passed=False,
    steering_development_rmse=[.06825875490903854, .09333650767803192],
    forward_development_rmse=[.1337403953075409, .17454473674297333],
    late_forward_rmse=[.10507484525442123, .10247436165809631],
    invalid_braking_valid_predictions_episode07=253, invalid_braking_late_valid_predictions_episode07=223,
    positive_braking_labels=0, actor_deployed=False, physiological_or_closed_loop_qualified=False)
data['next_action'] = ('Prioritize road/egocentric preview intermediate visual supervision and steering/power '
    'coordination/coast regression. Teacher pose/future trajectory may supply targets only, never actor inputs. '
    'Obtain real braking and variable speed tasks before braking qualification. Preserve failed v5; no collection '
    'restart in this model turn. Self-motion visual6, acceleration registration, fast-body fusion, tyre/setup, '
    'student circuit and racing-line/laptime optimization all remain outstanding.')
write(plan_file, plan)
write(destination/'transfer-receipt.json', dict(files=rows, copied_files=len(rows),
    copied_bytes=sum(r['bytes'] for r in rows), local_originals_retained=True,
    training_source_hashes=sources, training_source_zip_sha256=digest(archive),
    evaluation_overlay_parent='configured-training-source-t1436.zip',
    configured_source_not_retrospective_process_memory=True,
    documentation_history_sha256=histories,
    big_v4_v5_raw_payloads_retained_locally_not_fully_copied_here=True,
    tests=dict(passed=11, scope='new body21 and command boundary plus legacy body tests',
               real_measured_graph_initial_identity_and_joint_gradients_verified=True)))
print(json.dumps(dict(copied_files=len(rows), copied_bytes=sum(r['bytes'] for r in rows),
                      preserved_document_histories=len(histories))))
