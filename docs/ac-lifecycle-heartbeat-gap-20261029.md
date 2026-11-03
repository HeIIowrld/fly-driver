# 수명주기 상태 파일 공백과 비교 재개

기록일: 2026-10-29, Asia/Seoul. [대전제](PROJECT_CHARTER.md)는 유지한다.
후속 [원래 이력 분해·주파수 반응·빠른 경로 검토](flight-yaw-bandwidth-and-fast-path-20261029.md)를 완료했다.
아래의 ‘다음 작업’은 당시 기록이며 실제 안정화 미통과는 유지한다.

## 실패 원본과 확인한 범위

`t1240/block-1/trial-05`는 원래257관측을 기록한 뒤 종료 처리에서 실패했다.
`api-pause/result.json`은 `FileNotFoundError`, `lifecycle_requests_sent=0`이다.
즉 이번 오류는 pause API가 거부하거나 응답하지 않은 경우가 아니라 **요청 전 상태 파일 읽기 실패**다.
실패 시험은 성공으로 바꾸지 않고 원본과267개 축 요청 기록을 보존한다.

상태 게시자는250ms마다 `io.save`로 상태를 쓴다. 이후 실제 상태에서
`heartbeat_write_failures=3`을 확인했고 상태 파일도 다시 존재했다.
파일 교체 공백과 일치하지만 특정 Windows 내부 호출이나 정확한 공백 길이를 실측한 것은 아니다.
당시10ms 읽기 재시도만으로 회복하지 못했다는 사실을 넘어서 원인을 단정하지 않는다.

## 보완과 실제 확인

- `scripts/control_ac_session.py`의 수명주기 메타데이터만 최대500ms 기다린다.
  일시적 파일 부재/공유 충돌만 재시도한다. 잘못된 JSON, 오래된 heartbeat, API 오류를 성공으로 바꾸지 않는다.
- 원래 heartbeat 신선도2초는 유지한다. 내용과 수정 시각은 같은 열린 파일에서 얻는다.
  별도 `stat()`으로 다른 세대의 시각을 섞는 경합도 제거했다.
- 주행 중 진단50ms·명령80ms·영상150ms 가드와 뉴런 시간/상태는 바꾸지 않는다.
  이 대기는 actor 추론 루프에 들어가지 않는다. 이미 보낸 API 요청은 재전송하지 않는다.
- 메타데이터250ms 공백 회복, 영구 부재, 잘못된 내용, 기존 명령 계약을 포함해20검사와 Ruff를 통과했다.
- [t1241 읽기 전용 확인](../runs/flight-lifecycle-gap-status-20261029-t1241/result.json):
  실제 엔진50656, AI off, camera off, gas0/brake1, 정차에 가까운 속도, 창 비전면.
- [t1242 실제 pause](../runs/flight-lifecycle-gap-pause-20261029-t1242/result.json):
  새 명령1회, terminal 성공과 paused 상태 확인, 창 입력/활성화0.
  이 실제 요청에서는 추가 메타데이터 재시도가0회였으므로 실제 공백 회복을 실측했다고 주장하지 않는다.

## 비교를 이어가는 규칙

[t1243 재개 선언](../runs/flight-lifecycle-gap-continuation-amendment-20261029-t1243/result.json)을
다음 실행 전에 기록했다. 계획·모델·교란·평가 기준은 그대로다.
완료된 독립 trial00~04는 보존하고, 실패한 trial05는 집계에서 제외한 채 원본을 유지한다.
trial05~11은 각자 새 API 에피소드에서 시작한다. 실패 관측을 삭제하거나 다음 이력에 이어 붙이지 않는다.

`runs/flight-reclaimed-gpu-study-continued-20261029-t1244`가 이 재개 실행이다.
도구 세션39189는 종료0이다. 전체3블록·12회·3,084관측의 독립 감사가 완료됐고,
짝 조건6/6은 유효했다. IAE 감소4/6·개선율 중앙값1.043%로 사전 안정화 기준은 **미통과**다.
[최종 표와 원본 응답](../runs/flight-physical-yaw-study-report-20261029-t1245/summary.md)을 읽는다.
원본 관측80ms 초과0, 의도 재생 최대 오차2.20e-7이다. 실패 trial-05는 여전히 실패로 보존한다.
[t1246 최종 상태](../runs/flight-physical-yaw-study-final-status-20261029-t1246/result.json)는
paused/AIoff/cameraoff/gas0/brake1/주행선OFF/창 비전면이다.
[t1247 독립 native·장치 확인](../runs/flight-physical-yaw-study-idle-proof-20261029-t1247/result.json)에서도
정차·중앙 조향·제동·AIoff·vJoy FREE를 확인했다. 장치를 획득하거나 축을 전송하지 않았다.
최종 관련27검사와 Ruff도 통과했다.
다음은 원래 전체 이력의 동일 입력 반사실 재생으로 몸 구동·지연·시각 명령을 분리하는 것이다.

이 시험은6km/h 공학적 페달과 작은 양방향 조향 교란의 범위다. 영상 경로 판단·학습한 가감속·
비행 생리 재현·학생 단독 유효 완주·타이어/셋업 대응·레이싱 라인/랩타임은 별도 미완료다.
