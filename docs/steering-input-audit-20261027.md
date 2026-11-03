# 조향 실패 진단: 교사·영상 단서·다음 학습

기록: 2026-10-27, Asia/Seoul. 앞선 감각 기반 학생은 [개발 평가4조건 모두 이탈](steering-pilot-20261027.md)했다. 이번에는 교사 자체의 수행 가능성과 실제 정책 입력을 확인했다. **교사는 완료4/4지만 학생의 성공은 아직 입증되지 않았다.** AC·실제 초파리 회로의 결과가 아니다.

## 교사는 같은 물리 과제를 수행한다

`sensory_audit.evaluate_teacher`는 학생과 같은 차량·코스·고정 속도 의도·시작 조건·판정 예산을 사용한다. 교사는 원래부터 경로/자세 정답을 읽는 별도 제어기이며 actor가 아니다. 관측 전 시각·정답, 적용 명령, 적용 후 시각·정답을 한 행에 기록한다. 학생 입력에는 이 기록을 넣지 않는다.

| 개발 조건 | 완료 | 판단 단계 | 전진 거리 | 횡오차 RMS |
|---|---|---:|---:|---:|
| 기본 시작 | 예 | 6,700 | 96.72 m | 0.028 m |
| 중간 시작 | 예 | 4,488 | 64.49 m | 0.028 m |
| 왼쪽 자세 오차 | 예 | 6,708 | 96.73 m | 0.119 m |
| 오른쪽 자세 오차 | 예 | 6,704 | 96.72 m | 0.122 m |

네 조건의 전체 교사 진단은 렌더링을 생략하는 O 관측 모드에서 실행했다. 이는 정답 교사의 물리 진단을 빠르게 수행하기 위한 것이며 학생의 V 평가를 대체하지 않는다. 별도의 V 실행에서 기본 시작 조건 전체6,700단계를 기록했고 **O 실행의 시각·정답·명령 배열과 원소별로 정확히 일치**했다. 이 확인 뒤 나머지 V 중복 실행을 의도적으로 중단했다. 렌더링한 네 조건 모두가 완료됐다고 표시하지 않는다.

[교사 전체 결과](../runs/sensory-steering-audit-20261027/teacher-and-first-visual-tests/runs/teacher-no-render/evaluation.json), [전체 trace 일치 확인](../runs/sensory-steering-audit-20261027/teacher-and-first-visual-tests/render-parity.json), [V 중단 사유](../runs/sensory-steering-audit-20261027/rendered-teacher/runs/teacher-audit/supervisor-stop.json).

## 작은 경계 표식이 겹눈 입력에서 사라진다

기존 장면은 초록 지면과 지름14cm의 경계 표식을 사용한다. 실제 원영상과 눈당8×12 방향으로 적분한 값을 직접 확인했다. 이 시작 장면에서는 작은 표식이 적분 입력에 거의 남지 않는다. 기존 데이터에서 ray validity는1이며 영상은 존재하므로 카메라 누락과 구분해야 한다. 이 관찰만으로 기존 실패 원인을 확정하지 않는다.

기존 합성 겹눈 입력(왼눈, 최근접 확대):

![기존 작은 표식 장면의 8×12 방향 입력](../runs/sensory-steering-audit-20261027/observation-rays-t4.png)

새 시험에는 `road_visual="lane_surface"`를 추가했다. 도로면과 흰 경계를 정적 장면 도형으로 표시하며 접촉 형상·힘·actor의 숫자 채널을 추가하지 않는다. 평지에서는 접촉용 heightfield를 유지하고 같은 범위의 평면을 렌더링한다. 얇은 도형에 생기는 그림자 잡음을 피하려고 이 시각 조건에서만 광원 그림자를 끈다. 이는 통제된 합성 시각 기준선이며 실제 서킷의 포토리얼 재현이 아니다. 경사면에서는 접평면 조각으로 그리므로 정밀한 노면 시각 검증은 별도로 필요하다.

수정된 원영상(왼눈, 96×96 확대):

![연속된 도로면과 경계가 보이는 수정 원영상](../runs/sensory-steering-audit-20261027/observation-visible-fast-source-t4c.png)

이 조건의 눈당16×24 방향 입력:

![공간 배치를 유지하는 16×24 방향 입력](../runs/sensory-steering-audit-20261027/observation-visible-fast-rays-t4c.png)

`test_road_visual.py`에서 장면 변경 전후의 접촉 형상 수, 같은 명령에 대한 차량 위치·자세와 몸 센서 배열이 정확히 일치함을 확인했다. 영상의 변화와 지면의 실제 표시, actor 정답 채널 차단도 확인한다. 최종 수정 소스의 전체 테스트는 **120개 통과, 25.35초**다. [검증 로그](../runs/sensory-steering-audit-20261027/final-source-tests/pytest-validation.log), [실행 소스 해시](../runs/sensory-steering-audit-20261027/final-source-tests/source-provenance.json).

초기 평면 도형의 크기를0으로 지정해 지면이 보이지 않는 문제를 영상 확인으로 발견했다. 해당 학습 시도는 시범 수집 전에 중단했고, 중단 사유·로그·초기 모델을 보존했다. 이후 유한 크기로 수정하고 지면 표시 검증을 추가했다. [중단 기록](../runs/sensory-steering-audit-20261027/stopped-visual-preflight/supervisor-stop.json). 검증용 사본의 초기 Git/출처 파일 누락 때문에 생긴 패키지 테스트 실패와 수정 로그도 함께 보존했다.

## 다음 학생 실험

설정은 [sensory-visible-bc-pilot.json](../configs/sensory-visible-bc-pilot.json)이다. 연속 도로면, 눈당16×24 방향, 공간 격자 인코더, 코스 전체를 포함한 교사 시퀀스1개와 네 위치의 좌우 복귀 시퀀스8개, BC32epoch를 사용한다. 기존의 세 위치8초 표본만으로 남았던 코스 구간의 공백을 줄인다. 실제 표본 수·완료/절단 이유는 수집 결과로 판정한다. 이번 후보는 BC만 실행하며 DAgger 재시도는 이 평가 뒤 결정한다.

여러 조건을 함께 바꾸는 탐색 실험이다. 격자 인코더 하나의 인과적 효과를 판정하는 대조 실험이 아니다. 실행 루트는 WSL `/home/hj/flybrain-sensory-visible-fast-20261027-t4c`, 출력은 `runs/visible-bc-101101`, 외부 제한은1,800초다. 후속 확인에서 정상 종료와 전체 결과 수집을 완료했다. AC·실제 회로·속도/제동·독립 반복 검증 관문은 그대로 남는다.

### 후속 확인: 새 학생도 완주하지 못했다

전체 코스 시범6,700개와 복귀 시범3,200개, 합계9,900개 판단 표본으로32epoch를 실행했다. 학습 MSE는 약0.03538에서0.002816으로 줄었지만, 개발4조건 모두 이탈했다.

| 새 장면의 평가 조건 | 완료 | 평균 전진 거리 |
|---|---:|---:|
| 무학습 | 0/4 | 21.01 m |
| BC 학습 후 | 0/4 | 25.03 m |
| 같은 BC 모델, 영상 누락 | 0/4 | 32.89 m |

영상 누락이 평균에서 더 좋으며 특히 오른쪽 복귀 조건에서64.98m를 갔다. 이것은 영상 기여의 증거가 아니다. 단일 시드의 개발 조건에서 나타난 결과라 일반적인 영상의 해로움을 주장할 수도 없다. 현재 모델이 시각 단서를 운전에 유용하게 사용하고 있다는 검증에 실패한 상태다. 영상 단서를 늘리고 시범 구간을 채운 것만으로는 문제가 해결되지 않았다.

[최종 전체 요약](../runs/sensory-visible-bc-pilot-20261027/runs/visible-bc-101101/summary.json), [종료 코드0](../runs/sensory-visible-bc-pilot-20261027/supervisor-exit.json), [50개 파일 SHA-256 수집 영수증](../runs/sensory-visible-bc-pilot-20261027/collection-receipt.json). 실패 trace와 무학습·BC checkpoint를 모두 보존했다. 새 코스·다른 눈 해상도를 썼으므로 이전 장면의 수치와 엄밀한 성능 비교로 합치지 않는다.

다음 학습을 늘리기 전에 관측에서 횡방향·방향 단서를 읽을 수 있는지와 측정 조향값을 복사하는 의존성을 분리해 진단한다. 경로/자세 정답은 감독 label에만 쓰고 actor 입력에는 넣지 않는다. 같은 시각 표현으로 정답을 추정하는 보조 학습도 후보이며 현재 미구현이다. 실제 회로 후보의 학습이나 장기 PPO를 성공이 입증된 기준선처럼 바로 확대하지 않는다.

[46개 파일 SHA-256 수집 영수증](../runs/sensory-steering-audit-20261027/collection-receipt.json)은 완료된 진단과 중단된 시도를 함께 기록한다. `final-source-tests` 아래 학생 파일은 수집 시점의 초기 스냅샷이며 완료 결과가 아니다. 영상 원배열과 추가 이미지의 해시는 별도 검증 영수증에 남긴다.

## 실제 기관 입력 후보도 확보했다

이전의 `FeCO` 문자열 검색 결과0은 해당 기관이 없다는 뜻이 아니었다. 공식 주석의 `status=Traced`, `class=mechanosensory_proprioceptive`를 직접 조회하면 **1,454개**가 있고 모두 복구 그래프에 존재한다. 이 중 subclass가 `chordotonal organ`인425개, `haltere`인201개, `campaniform sensilla`인426개, `hair plate`인113개다. 나머지 주석도 삭제하지 않고 보존했다. 이는 주석별 개수이며 완전한 생물학적 기관 센서 수를 주장하는 수치가 아니다.

`scripts/inventory_body_ports.py`는 실제 body ID·유형·진입 신경·rootSide·somaSide·MANC 대응 필드를 유지한다. 쉼표로 연결된 애매한 유형 이름도 그대로 남긴다. [기계 판독 목록과 소스 해시](../assets/external/malecns-body-port-inventory-20261027/inventory.json). 추출 스크립트의 해시도 목록에 봉인했으며 주행 중인 소스 사본에는 이 후속 스크립트를 덧붙이지 않았다.

`chordotonal organ` 주석만으로 FeCO의 claw/hook/club을 판정하지 않는다. [Lee 등, 2025](https://www.nature.com/articles/s41467-025-59302-3)는 FANC/FlyWire 자료에서 위치·운동 입력의 국소 경로와 진동 입력의 상행 경로를 구분한다. 그 결과는 본 프로젝트의 기능 설계 근거지만 MaleCNS의 개별 body ID 대응표가 아니다. 따라서 다음 회로 작업은 유형·해부학 대응을 확인하고 실제 국소/상행 연결을 감사하는 것이다. 현재 기관 인코더 값을 임의의 후보 뉴런에 연결해 기능 재현이라고 표시하지 않는다.
