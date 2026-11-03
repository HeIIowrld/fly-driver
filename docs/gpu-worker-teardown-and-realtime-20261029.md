# GPU 작업의 종료 대기와 실시간 비교 재개

기록일:2026-10-29,Asia/Seoul. [프로젝트 대전제](PROJECT_CHARTER.md)는 유지한다.
후속 [상태 파일 공백 보완과 전체 비교](ac-lifecycle-heartbeat-gap-20261029.md)에서12회/3블록 감사를 완료했다.
오차 감소4/6·중앙값1.043%로 안정화 사전 기준은 미통과다. 아래 중단/다음 작업은 당시 기록이다.
이 작업의 목적은 원래 초파리 시각/비행 회로의 관측·시간·상태를 보존하면서 실제 비교를 끝까지 수행하는 것이다.
일반 운전 모델, 축소한 회로, 느슨한 신선도 기준으로 대체하지 않았다.

## 실제로 발견하고 수정한 문제

예전 실패 검증PID26764(t1159)와27384(t1232)는 GetExitCodeProcess에서1을 반환했지만,
같은 프로세스 핸들의 WaitForSingleObject(0)는258, 즉 아직 종료 신호가 오지 않은 상태였다.
기존 `ProcessReceipt`는 종료 코드만 보고 `active=false`로 분류했다. 이 해석은 잘못됐으며 수정했다.
두 작업은 원래 결과가 명시적 예외/실패이고 출력0인 오프라인 검증이었다.

- [t1235](../runs/flight-failed-gpu-worker-cleanup-20261029-t1235/result.json)는 UTF8 설정을 기본cp949로 읽다가
  사전 점검에서 실패했다. 프로세스 종료 요청0이었다.
- [t1236](../runs/flight-failed-gpu-worker-cleanup-20261029-t1236/result.json)은 실패 원본·해시·PID·생성 시각·
  실행 모듈/결과 경로를 확인하고 두 작업에만 각각 종료 요청1회를 보냈다.
  그 뒤 같은 핸들에서 종료 신호0과 종료 코드1을 확인했다. 다른 작업은 종료하지 않았다.
- Windows GPU 메모리 표시가 정리 직전12,926.7MiB에서 직후6,993.5MiB로 줄었다.
  사라진 두 PID의 표시는 각각2,643.0/3,290.1MiB였다. 이 수치는 당시 도구 출력의 관측값이며
  정확한 드라이버 내부 원인은 조사하지 않았다.
- [t1237 동일 원본 재생](../runs/flight-after-gpu-cleanup-zero-prefix-20261029-t1237/result.json)은 원래8관측 모두와
  마지막80ms 가드 실패를 그대로 재현했다. 앞선7추론은25.125~33.985ms였다.
  정리 전 t1226의 같은7추론은41.466~66.239ms였다. 수치 출력은 같고 관측을 삭제/재시각화하지 않았다.

[Microsoft 프로세스 종료 문서](https://learn.microsoft.com/en-us/windows/win32/procthread/terminating-a-process)와
[대기 API 문서](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject)를 기준으로,
이제 실제 프로세스 종료 신호를 확인한다. 관측된 종료 코드/대기 결과의 불일치를 DLL이나 특정 드라이버
버그 하나로 단정하지 않는다. 이전의 ‘종료 코드1 확인’은 실제 자원 회수 완료 증거로 쓰지 않는다.

앞서 엔진 재시작 t1213의 원래PID46748은 t1236에서 종료 신호와 코드0을 추가 확인했다.
따라서 이전 엔진 재시작의 실제 완료 근거는 유지한다.

## 재발 방지

`win_process_receipt.py`는 SYNCHRONIZE 권한으로 같은 핸들의 신호를 검사한다.
`owned_worker_wait.py`는 시간 초과 때문에 작업을 중단/대체하지 않는다.
명시적 실패 결과와 종료 진행 상태가 함께 확인된 소유 작업만 한 번 정리할 수 있다.
실제 주행 작업은 제동·중립·pause·스레드 종료·오류 없는 장치 해제가 기록돼야 하며,
상태가 불명확한 수명주기 작업이나 다른 작업은 대상이 아니다.

현재 검증/감사/실제 시험 코디네이터에 이를 연결했다. 관련185검사와 Ruff를 통과했다.
t1239의 성공 작업은 각 `trial-XX-process.json`에서 자연 종료0·실제 종료 신호를 확인하며,
정상 작업을 강제 종료하지 않았다.

## 원본 대조한 연구용 대안 — 실제 운전에 미적용

| 기록 | 결과와 판단 |
|---|---|
| [t1225 자원 창](../runs/flight-overlapping-resource-window-20261029-t1225/result.json), [t1226 원본 재생](../runs/flight-overlapping-zero-prefix-profile-20261029-t1226/result.json) | 43표본 중1개 보수적 query 구간이 실제 계산과 겹침. 별도LLM의 높은compute는 이 표본에 없었음. 관측된 python17588을 해당 worker로 고유 귀속할 PID 영수증은 당시 없으므로 과도한 원인 주장은 하지 않음 |
| [t1227](../runs/flyvis-fullbatch-stage-20261029-t1227/result.json), [t1228](../runs/flyvis-fullbatch-versus-binary-20261029-t1228/result.json) | 1~16단계 각각의 단일 그래프는225관측/448묶음 전체 뉴런 대조 통과. 기존 binary 경로와 순서를 번갈아 비교한 합계3.760→5.656초로 더 느려 미승격 |
| [t1229](../runs/flyvis-fractional-stage-20261029-t1229/result.json), [t1230](../runs/flyvis-fractional-versus-binary-20261029-t1230/result.json) | 소수 dt를 같은float32 동적 입력으로 넘기는 경로도 전체 대조 통과. 기존 binary와의 합계4.995→5.338초로 개선되지 않아 미승격 |
| [t1231](../runs/flight-linear-yaw-coupled-verification-20261029-t1231/result.json), [t1232/full](../runs/flight-linear-yaw-drive-verification-20261029-t1232/full/result.json) | 일반Linear로 잘못 가정한 초기 시도는 보호된FrozenLinear 형식에서 거부됐고, 이어 CSR deepcopy도 지원되지 않아 실패. 둘 다 운전에 적용하지 않았음 |
| [t1233](../runs/flight-linear-yaw-drive-verification-20261029-t1233/result.json) | 실제FrozenLinear 버퍼를 보존하고 별도 복사본의 yaw 열만0으로 만드는 동등 개입을 검증. full/yaw0 각225관측에서 원래 모델의 모든 뉴런 구동값 차이0, 최종 의도 오차 최대1.83e-7 |
| [t1234](../runs/flight-linear-yaw-zero-prefix-profile-20261029-t1234/result.json) | 원래8관측 재생 통과. 당시 시각 계산이 더 느렸으나 두 번째 실패 GPU 작업의 메모리가 남은 상태였으므로 개입 구현의 고유 비용으로 단정하지 않음. 실제 운전에 미적용 |

이 비교들은 자원 회수 전의 조건에서 수행했다. 모든 부하에서의 일반적인 속도 순위로 해석하지 않는다.
실제 시험에는 기존 인증된 binary 시각 경로와 원래 입력 복사형 yaw0 개입을 그대로 사용한다.
모든 연구용 대안과 실패 원본은 보존했다. 유리한 결과만 골라 배포하지 않았다.

## 실제 비교 재개

새 계획은 `configs/flight-physical-yaw-reclaimed-gpu-v2-20261029.json`이다.
원래12회/3블록·257관측·양방향280ms 교란과50/80/150ms 제한을 유지한다.
정리 전 성공 시험을 새 비교에 포함하지 않고 처음부터 진행했다.

- [t1238 점검](../runs/flight-reclaimed-gpu-preflight-20261029-t1238/result.json) 통과.
- [t1239 첫 블록](../runs/flight-reclaimed-gpu-block0-20261029-t1239/result.json): full/yaw0 양방향4회 모두257관측 완료.
- [t1240/audit-0](../runs/flight-reclaimed-gpu-study-20261029-t1240/audit-0/result.json):4회 전체 독립 감사와
  두 방향의 모든 짝 조건 통과. 첫 블록의 회전 오차 적분값 개선은3.408%/0.432%였다.
  이는 예비 결과이며 전체 판정이 아니다.
- [t1240/block-1](../runs/flight-reclaimed-gpu-study-20261029-t1240/block-1/result.json)은 실패로 종료했다.
  trial-04는 완료했고 trial-05도257관측을 기록했지만 최종 pause API 실패가 `cleanup_errors`에 남았다.
  trial-05는 성공 시험으로 세지 않는다. 해당 worker의 종료 신호와 코드1은
  [프로세스 영수증](../runs/flight-reclaimed-gpu-study-20261029-t1240/block-1/trial-05-process.json)으로 확인했다.
  코디네이터 도구 세션42670도 종료1이다. 전체12회/3블록 감사·집계는 미완료다.
  `progress.json`의 block-1/PID는 마지막 기록이며 현재 실행 중이라는 증거가 아니다.
- 다음 작업은 최종 pause 실패 원인과 현재 게임·장치 상태를 확인하는 것이다.
  제동 기록만으로 pause 완료를 주장하거나 실패 시험을 다음 시험에 이어 붙이지 않는다.
  첫 블록의 예비 수치로 모델·기준을 바꾸지 않는다.

학생 단독 완주·반복 안정성·학습한 가감속·타이어/셋업·레이싱 라인/랩타임과 생리학적 재현은 여전히 별도 미완료다.
