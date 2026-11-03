# 120FPS 실측과 내부 센서 지연 계측

갱신: 2026-10-29, Asia/Seoul. 아래 경로의 원본은
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/`에 있다.
이전 무제한 FPS 실패는 `flight-exterior-stream-qualification-and-corpus-20261029.md`에 보존한다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다.
차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 실제 변경과 결과

- t1577에서 원본 video.ini를 백업하고 FPS_CAP_MS만 8.333333으로 적용했다.
  원본 엔진36644의 종료 신호/종료0과 새48364를 확인했다. 새 generation은
  `26544658ULL:0.0027389`, 창640×360이며 전면 활성화하지 않았다.
- 복구 전 native AI 플래그는 true였다. 별도 recovery-only preflight는 진행 중인
  native graphics 패킷의 AC_PAUSE와 정차·브레이크·CSP idle·해제된 vJoy를 확인했다.
  이것은 운전 소유권이 아니다. 기존 학생/일반 restart의 native AI-off 가드는 유지했다.
  실행 뒤 preflight 필드 검사를 더 엄격하게 고쳤으며, 실행본과 수정본을 구분한다.
- t1579 첫 pause는 브레이크가 해제돼 요청0으로 거부됐다. t1580 후속 자격 시도는
  begin_run 이전 거부여서 결과 디렉터리가 없다. t1581 정차 브레이크 초기화7회 뒤
  t1582 pause, t1583 정차8묶음 검증을 완료했다.
- t1585 3초:75관측, 대기누락0, 최대56.6401km/h. t1586 원시 감사 최대 간격41.5772ms.
- t1587 30초:737관측, 대기누락0, 최대61.0938km/h. t1588 원시 감사 최대 간격42.8705ms.
  두 감사 모두 원시 배열 차0, 80ms 초과0, baseline packet advance 확인이다.
- t1590/1591의 몸16→21 확장에서 기존16이 정확히 보존됐다. 무효 채널2,3,5,6,7은
  무효로 남겼다. t1592는 실제 저자 시각망 네 독립 상태로737관측을28.929초 GPU 재생했다.
  원본 시간30.00083초, 내부6624 tick, peak5,515,289,088B, 저자 가중치 변경0이다.
  이것은 학생 운전이나 운동 회로 통합 검증이 아니다.

## 후속 수집은 실패

t1593 geographic corpus는 첫 transit에서 실패했다. 부모 수집기 종료1, active=false,
완료 TRAIN0/개발0이다. 자식272관측/대기누락3회,414영상 묶음이며 종료 사유는
`owned camera ended or became stale`이다. 120FPS만으로 중단이 해결되지 않았다.
실패 이력은 fitting 불가이며 t1594 무결성 감사 완료를 수집 성공으로 세지 않는다.

t1595 원본 공백은 최대214.2095ms이고 그 사이 render counter는2 증가했다.
동시에 비동기 저장 그룹355에서198.7862ms 쓰기 구간이 보인다.
기록된 개별 GeometryShot 호출 최대는 첫 front30.2165ms이며, 공백 뒤 rear는2.0623ms다.
동시 발생만으로 디스크나 GPU를 원인으로 확정하지 않는다.

t1596은414묶음 전체/1656개 view 복사 기록을 감사했다.
GPU복사→ACK 최대6.2815ms, metadata MMF 인지까지 상한38.8654ms,
인지→metadata 검증 최대0.8118ms다. 기록된 복사 단계로214ms 공백을 설명하지 못한다.

t1584는 3초 교사 전환과 실제로 겹친 수동 관찰이다. CSP AI-off/비정지이고 native
packet도 진행하는 동안 native AI=true가106행 관측됐다. 일시정지된 낡은 플래그만으로
모든 불일치를 설명할 수 없다. 의미는 미확정이며 학생 소유권 가드는 완화하지 않는다.

## 내부 Python 센서 계측 수정 — 아직 미배포

`integrations/ac/flybrain_telemetry/flybrain_telemetry.py`에 일정 크기의 누적 host timing을 추가했다.
SDK body/teacher 읽기, publish와 파일 serialize/open-write-close/replace-retry를 구분한다.
각 단계의 마지막/최대 시작·종료·소요 시간, 횟수와 실패 횟수를 다음 패킷에 보낸다.
실패한 최대 지연도 보존한다. 추가 로그 파일/추가 쓰기/새 스레드는 만들지 않는다.
센서 타임스탬프, 50Hz 발행 간격, 동일 payload 재시도, 3ms replace 재시도 예산을 유지한다.
이 필드는 actor_input=false인 진단이며 simulation clock/노출 시각이 아니다.
패킷 전체를 발행하지 못한 마지막 호출은 다음 성공 전에는 외부에서 볼 수 없다.

214ms 실패와198ms open 지연을 가상 시계로 주입해, 성공 뒤 최대 지연 보존과 원래
측정 시각 보존을 검증했다. 관련 file/TCP/teacher SDK/recovery preflight44검사 통과.
실제 게임 원인 확인이나 지연 해결로 주장하지 않는다. 설치된 publisher는 여전히 이전 코드다.

## 현재 상태와 다음 실행

20:59:03 KST의 신선한 CSP heartbeat(파일 age4.9ms)는 paused=true, AI-off,
camera-off/probe-off, gas0/brake1/0.00969km/h다. 현재 수집/학습 프로세스는 없다.
오래된 소유 프로세스17212/2328의 종료 미확정은 별도 cleanup 문제로 남는다.

다음은 새 계측 소스/설정/설치본의 해시와 백업을 묶고, 검증된 복구 절차로 게임에 반영한 뒤
동일 판정 기준의3초→30초에서 원본 host timing을 수집하는 것이다. 현재 FPS 설정은 이미
120이므로 옛 0→120 적용 계획을 그대로 재사용하지 않는다. native 플래그 불일치 상태에서
학생 입력을 허용하지 않는다. 원인이 드러난 단계에 맞춰 수정하고, 사전 분리한 새 코너
TRAIN/개발 자료로 시각→조향 및 가감속 경계를 다시 학습·검증한다.

학생 배포0, 이번 optimizer update0, 유효 학생 랩0이다. 고속60→100/다속도·실제 제동·
하중·차체 기울기·타이어/셋업 적응·비행 운동 회로 통합·레이싱 라인/랩타임은 모두 남아 있다.
