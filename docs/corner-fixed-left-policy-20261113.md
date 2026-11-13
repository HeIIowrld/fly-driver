# 고정 좌우 코너 판독과 레이싱 라인 대조 · 2026-11-13

한 설정을 Silverstone 1967, Vallelunga Extended, Monza의 네 방향 저장 영상에 적용했다. 실제 코너 시작점이 4–20m 앞이고 방향과 시작 거리 ±5m가 함께 맞으면 적중이다. 렌더 전후 정답 거리가 둘 다 있으면 두 거리에서 이 기준을 통과해야 한다. 보류, 직선 출력, 미출력은 접근 분모에 포함한다. 정답 라벨은 저장 영상의 관측과 국소 몸 감각으로 판독을 실행한 뒤 채점에 사용했다.

## 문턱 고정과 재생 결과

기존 T4/T5 판독에서 확정한 우회전은 그대로 보존했다. 기존 좌회전은 복구 경계에서 고른 후보를 확인 점수 0.97339 이상일 때 유지했다. 회전 경보가 없던 프레임은 원래 관측에서 고른 좌회전 후보의 확인 점수가 0.87175 이상이면 좌회전을 출력했다. 두 조건은 동일한 오류 유형 분류 확인기의 점수를 사용한다. 후보를 고른 관측 경계나 복구 경계는 영상과 국소 속도·요율로 계산했다. T4/T5 결측과 코너 내부 차단 조건을 적용했다.

Imola·Laguna Seca·Zandvoort·Mugello에서 확인 점수, 후보 출처, 두 문턱을 비교했다. 기존 실험의 합산 상한을 가져오고 녹화별 상한을 이번 비교에 추가했다. 네 개발 녹화 합산으로 직선·범위 밖, 내부, 반대 방향, 거리 오류의 순증을 각각 8·5·2·2장 이내로 제한했고, 녹화별 상한은 각각 4·3·1·1장으로 정했다. 채택 조건의 개발 순증은 각각 −6·−4·0·−5장이며 좌 적중이 네 곳에서 2·1·2·5장 늘었다. 새로 적합한 비교 모형은 개발 서킷을 하나씩 제외했다. 선택된 오류 유형 확인기와 하위 관측기는 개발 네 곳을 사용한 기존 동결본이다. 따라서 개발 수치는 문턱 보정용이다. Silverstone·Vallelunga·Monza는 이 설정의 학습과 문턱 선택에서 제외했으며, 기존 실패 분석에 쓴 비교 영상이다. [동결 설정 및 비교 후보](../runs/vision-corner-left-risk-reallocation-20261113-01/frozen.json)

| 저장 영상 | 좌 방향·거리 동시 적중 | 우 방향·거리 동시 적중 | 직선·범위 밖 경보 | 내부 경보 | 접근 중 보류 / 미출력 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Silverstone 1967 | 0→1/11 | 2/13 유지 | 11→5 | 7→7 | 20 / 0 |
| Vallelunga Extended | 0→2/23 | 5/38 유지 | 6→4 | 10→10 | 46 / 0 |
| Monza | 0→1/24 | 4/42 유지 | 26→13 | — | 58 / 0 |

Silverstone은 746번에서 실제 13.19m를 15m로 읽었다. 방향만 맞고 시작 거리 오차가 5m를 넘긴 좌회전 경보도 한 장 추가됐다. Vallelunga는 같은 접근의 995·996번에서 실제 10.39m·8.11m를 모두 13m로 읽었다. Monza는 1163번에서 기존 15m 정답과 일치했다. 접근 구간 중 좌 적중을 포함한 구간은 Silverstone 1/3, Vallelunga 1/6, Monza 1/8이었고 Vallelunga에서 두 장이 연속 적중했다. 방향이 맞은 모든 출력의 시작 거리 평균 절대오차는 좌 Silverstone 5.26m·Vallelunga 3.75m·Monza 0.00m, 우 3.37m·2.33m·4.89m였다. Monza는 기존 989개 라벨로 비교했으며 코너 내부 경보의 분모가 없다. 세 녹화에서 모든 기존 우회전 상태와 거리가 프레임별로 일치했다. [전체 채점](../runs/vision-corner-left-risk-reallocation-20261113-01/comparison.json) · [검증](../runs/vision-corner-left-risk-reallocation-20261113-01/verification.json)

## 레이싱 라인

새 코너 출력을 이전의 연속 경로 생성기에 연결해 세 영상 5,552장을 다시 재생했다. Silverstone 421장, Vallelunga 735장, Monza 2,566장에서 경로가 나왔으며 기존 경로 좌표가 바뀐 프레임은 0장이었다. 이전 Monza 비교에서 연결된 코너는 22곳 중 8곳이었다. [새 코너 출력과 경로 재생](../runs/vision-corner-left-risk-reallocation-20261113-01/rolling-line-comparison.json) · [진입부터 탈출까지 연결 평가](../runs/vision-corner-onset-through-line-20261113-01/monza-through-corner.json)

Monza의 별도 교사 평가 랩 1,246프레임에서 교사 궤적의 전방 5·10·20m 호길이 지점을 차체 전방 단면으로 변환해 횡오차를 계산했다. 해당 단면을 관측 범위 안에서 내놓은 연속 경로는 각각 1,009·1,021·1,019장이다. 20m에서 동일 1,019장의 연속 경로는 평균 횡오차 1.97m, 차체 진행 방향 직선은 0.59m였다. 현재 좌회전인 동일 49장에서는 각각 3.93m와 2.92m, 우회전인 동일 135장에서는 2.55m와 1.94m였다. 기존 카메라·속도 학습 판독까지 모두 출력한 공통 103장의 20m 오차는 연속 경로 1.96m, 카메라·속도 판독 0.65m였다. 연속 경로의 커버리지와 교사 궤적 오차를 함께 기록했다. 이 비교로 빠른 경로 추종이나 랩타임은 측정하지 않았다. [교사 궤적 대조](../runs/vision-corner-fixed-left-20261113-01/monza-teacher-line.json)

## 독립 서킷 평가 상태

설정을 고정한 뒤 학습·문턱 선택·기존 비교에 쓰지 않은 **Highlands Short**의 저장 영상에 적용했다. BMW Z4 GT3 오프라인 내장 AI 주행으로 전·좌·우·후 영상 1,200묶음을 저장했다. 주행선 표시는 꺼져 있었고, 저장 간격 중앙값은 90.05ms였다. 전체 1,200장은 정지 663, 직선 46, 범위 밖 134, 코너 내부 304, 4–20m 접근 53장이다. 접근은 좌 21장(5곳), 우 32장(9곳)이다. 같은 레이아웃과 영상 묶음의 해시를 확인했다. [이전 원본 목록](../runs/vision-corner-fixed-left-20261113-01/independent-inventory.json) · [추가 원본 목록](../runs/vision-corner-frozen-transfer-20261113-01/additional-inventory.json) · [촬영 기록](../runs/vision-corner-frozen-transfer-20261113-01/highlands-capture/capture/result.json) · [입고와 정답 분모](../runs/vision-corner-frozen-transfer-20261113-01/highlands-admission.json)

| 지표 | 기존 T4/T5 | 같은 고정 설정 |
| --- | ---: | ---: |
| 좌 방향·시작 거리 ±5m 동시 적중 | 0/21 | 2/21 |
| 우 방향·시작 거리 ±5m 동시 적중 | 1/32 | 1/32 |
| 직선·범위 밖 경보 | 12/180 | 8/180 |
| 코너 내부 경보 | 5/304 | 9/304 |
| 반대 방향 / 방향만 맞고 거리 실패 | 2 / 3 | 2 / 1 |
| 접근 보류 / 미출력 / 전체 실패 | 42 / 0 / 52 | 42 / 0 / 50 |
| 전체 보류 / 미출력 | 1,140 / 0 | 1,140 / 0 |

좌회전 845번은 실제 11.09m를 9m, 다른 코너의 1014번은 실제 9.71m를 11m로 읽었다. 방향이 맞은 좌 출력의 평균 절대오차는 10.89→1.69m였다. 우 출력의 평균 절대오차는 6.22m였고 기존 우 예측은 프레임별로 그대로였다. 좌우 모두 두 프레임 연속 동시 적중은 0곳이었다. **독립 오경보 기준 불합격.** 직선·범위 밖 경보는 4장 줄었지만 코너 내부 경보가 4장 늘어 고정한 녹화별 순증 상한 3장을 1장 넘었다. 나머지 세 오류 유형은 상한을 충족했다. [독립 채점](../runs/vision-corner-frozen-transfer-20261113-01/highlands-replay/result.json) · [프레임별 예측](../runs/vision-corner-frozen-transfer-20261113-01/highlands-replay/forecasts.jsonl) · [정답](../runs/vision-corner-frozen-transfer-20261113-01/highlands-replay/labels.jsonl)

Silverstone·Vallelunga·Monza는 보관된 센서 기록으로 다시 실행했고, 전체 예측 파일과 채점의 SHA-256이 앞서 기록한 값과 일치했다. Highlands는 저장 원본에서 경계와 T4/T5를 새로 추출했다. FlyVis는 Vallelunga에서 그래프 정지 후 사용한 기존 GPU 적분 함수로 실행했다. 그 경로의 이전 두 프레임 대조에서 원본·반사 특징 차이는 최대 8.94×10⁻⁸이었고, 이번 원본·반사 특징은 각각 1,198/1,200장에 나왔다. 최종 판독·채점 문턱과 가중치는 같은 해시를 유지했다. [세 서킷 재현 검사](../runs/vision-corner-frozen-transfer-20261113-01/reproduction.json) · [영상 특징 추출](../runs/vision-corner-frozen-transfer-20261113-01/highlands-motion-provenance.json) · [전체 채점 검산](../runs/vision-corner-frozen-transfer-20261113-01/verification.json)

앞서 Silverstone 원본 영상으로 독립 평가와 같은 재생 경로를 점검했다. 기존 T4/T5 추출 기록을 재사용하고 영상 경계를 다시 계산해 고정 판독의 1,200프레임 출력을 모두 재현했다. [원본 재생 점검](../runs/vision-corner-left-risk-reallocation-20261113-01/silverstone-original-image-rehearsal/policy-reproduction.json)

Highlands Short의 독립 평가와 경로 채점 결과를 기록했다. [독립 원본과 채점](corner-highlands-frozen-transfer-20261113.md)
