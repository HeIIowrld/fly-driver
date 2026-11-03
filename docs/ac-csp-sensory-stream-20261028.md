# 엔진4시점과 native 몸 감각의 입력 스트림

2026-10-28 KST. t152에서 백그라운드의 BMW Z4 GT3 / Monza 세션을 읽어 네 방향 영상과
native 몸 감각32개를 망막 입력으로 변환했다. t153에서 원본 PNG·physics prefix bytes로
전부 재현했고 입력7필드의 최대 절대 오차는 모두0이었다. 모델 추론·조향·페달 요청은0이다.
기존 가속 유지 실패나 자율 완주를 해결한 결과는 아니다.

## 시각 정렬의 범위

CSP 앱은 갱신할 때마다 고유 producer generation을 만든다. 창 핸들과 generation을
포함하는 nonce 응답으로 Python의 보낸 시각/받은 시각 사이에 CSP clock 응답이 있었다는
범위를 만든다. `host_time - CSP_time`은 `[sent-CSP, received-CSP]` 안에 있어야 한다.
여러 응답의 교집합이 없거나 generation이 바뀌면 원점을 재사용하지 않는다.

`CSPClockBounds`는2회 이상의 일관된 응답과 최대0.1초의 불확실성 예산을 요구한다.
t152는 시작4회로 약9.0655ms의 범위를 얻었다. 기록 끝의2회도 같은 범위와 양립했다.
온라인에서 사용한 초기 범위는 고정해 보존했으며, 이후 응답으로 과거 관측 시각을
유리하게 덮어쓰지 않는다. 제한된 구간에서 clock 단위 기울기1 가정이 일관된지 확인한
것이고 장기 clock drift나 실제 노출 동기화를 입증한 것은 아니다.

영상 나이는 `decision - (CSP_render_start + offset_lower)`로 계산해 불확실성을 포함한
상한을 사용한다. 실제 카메라 노출 시각이 아니라 엔진 렌더 호출 시각의 상한 평가다.
소스별 시각을 유지하며150ms를 넘는 소스만 결측 처리한다. 메타데이터 파일 커밋 뒤
native physics와 SDK를 읽고, 그 뒤의 host 결정 시각을 저장한다.

## 시점별 망막

`configs/ac-csp-retina.json`과 `CSPCameraRetina`는 front/left/right/rear 네 소스를 분리한다.
각각1709개 픽셀 수용기를 사용한다. 실제 반환 투영행렬의 X/Y 배율과 선언한 차체 방향을
이용해 단위 시선 방향을 구성한다. 차체 좌표는 공학적 x-전방/y-왼쪽/z-위쪽이고 실제 파리
수용장의 배치·수용각을 측정한 것으로 부르지 않는다. 공간 필터 폭1.5픽셀도 공학적 설정이다.
`body_ray_registration_verified=false`와 실제 노출 동기화 미검증 표시는 유지한다.

네 이미지를 파노라마로 이어 붙이거나 먼저 평균하지 않는다. 시점별 RGB·시간 대비·유효성·
나이를 따로 만든다. 같은 영상 재사용 시 대비를 새로 만들지 않고, 결측·stale·긴 시간 간격
후에는 이전 대비를 사용하지 않는다. 같은 ID의 픽셀·시각 변경, 시간 역행, producer 변경은
거부한다. 다른 기관 인코딩에서 오류가 나도 시각 이력을 중간 상태로 남기지 않는다.

## 몸 감각과 진단의 분리

몸 입력은 기존 인코딩의 `suspensionTravel`, `tyreCoreTemperature`, `wheelLoad`,
`wheelAngularSpeed`만 허용한다. 기관132개 중 최초98개, 이후126개가 유효했다. 바퀴 회전0은
유효한 정차 관측이며 센서 결측과 구별한다. 조향 roadwheel angle과 body angular velocity
인코딩은 여전히 지원하지 않는다.

actor 필드는 `view_rgb`, `view_contrast`, `view_valid`, `contrast_valid`, `view_age`,
`organs`, `organ_valid`뿐이다. 영상 sample 인터페이스는 픽셀·ID·generation·시각 외 필드를
거부한다. 월드 위치·방향·지도·SDK의 privileged teacher 값은 별도 원본 진단 기록으로
보존하지만 이7필드 계산에는 넘기지 않는다. 재현 감사는 복사된 body 필드 값을 믿는 대신
해시를 확인한 원본 native prefix를 다시 해독했다.

## 기록과 검증

실시간 기록은 로컬 spool에 먼저 쓰고, 제한된 producer가 종료된 뒤 NAS run 디렉터리에
복사해 해시를 검사한다. consumer 오류가 producer 재실행을 촉발하지 않는다. producer는
기존 최대100묶음·60초 제한을 유지한다. 원본 local 디렉터리도 삭제하지 않았다.

- t152: 요청 주기0.1초,32관측·128장, 네 시점 모두 유효한 관측32개. 전면 표본0.
- t153:32관측 전체,7필드 오차0 재현. 원본 픽셀·native/SDK wire·source 순서·clock 종료
  교집합을 검사했다. 독립 최종평가는 아니다.
- clock 범위·세대 변경, 소스별 결측·stale, 대비 재시작, ID 변조, actor 진실값 주입 거부,
  오류 시 이력 복구를 단위 검사했다.

데이터에는 `training_admissible=false`를 유지했다. 이는 교사 정답을 가진 학습 dataset이나
25Hz 모델 제어 기록이 아니다. 현재는 카메라 묶음마다 입력을 생성하는 관측 스트림이다.
producer 요청 주기와 실제 도착 간격은 같다고 가정하지 않는다. 다음은 고정 제어 결정마다
최신 카메라를 유효한 나이 내에서 재사용하는 비동기 수신 경로, 새 입력 계약으로 모델
등록, 현재 세션 조향 재보정, 새 교사 관측/정답 수집과 학습이다. 이 과정에서 과거 시점의
학습 자료를 새 카메라 자료로 가장하거나 학생 자신의 요청을 전문가 정답으로 쓰지 않는다.
