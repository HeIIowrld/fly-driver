# 공유메모리 metadata·ACK 정차 검증

2026-10-29, Asia/Seoul. **초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는
가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며
조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.**

**실제 CSP와 Python 사이의 공유메모리 통신으로 8묶음/32개 화면을 수신했고,
같은 순간의 producer PNG와 픽셀 차0을 확인했다. 아직 정차 검증이며 주행 수집기에 연결하지 않았다.**

## 구현

`ac_metadata_mmf.py`는 GUID별 별도 metadata/ACK map을 사용한다.
Python은 자신의 새 ACK map만 만들고 기존 이름이 있으면 거부한다.
CSP는 metadata map에 쓰고 ACK map은 읽는다. Python은 metadata map을 읽기 전용으로 연다.
native AC mapping이나 producer map에 쓰지 않는다.

packet은 little endian `magic/version/sequence/length` 32비트 정수4개,
SHA256 hex64바이트, payload32768바이트로 총32848바이트다.
쓰기 중에는 odd sequence, 완료 뒤에는 even sequence를 공개한다.
수신 측은 반복 복사·sequence·길이·SHA256·캡처 번호·request/generation을 검증한다.
checksum 손상과 캡처 건너뛰기는 오류이며 이전/미완료 packet과 구분한다.
네 GPU 영상 모두 복사된 후에만 ACK를 한 번 공개한다. 실패한 GPU 그룹은 ACK를 보내지 않는다.

기존 `copy_shared_group`의 파일 ACK는 기본값으로 유지했고, 명시적 `ack_writer`에서 새 경로를 쓴다.
실행 중인 Lua의 새 경로는 `mmf-probe-<GUID>`/최대8묶음/정차·full brake·CSP AI-off로 제한했다.
주행·9개 이상·공유 GPU transport 없는 요청은 map을 열기 전에 거부한다.
기존 교사/학생 guard를 완화하지 않았다.

## 검사와 실제 게임 결과

관련230검사와 Ruff를 통과했다. 실제 Windows map 이름 충돌에서도 기존 내용을 보존했고,
LuaJIT FFI와 Python의 packet을 양방향으로 교환했다. odd/torn/old/skipped packet과 checksum 손상,
실패 GPU 그룹의 ACK 금지, Lua 정차 제한을 검사했다.

t1515는 기존 설치 SHA256을 확인하고 paused/offline/정차/full brake/장치 해제 상태에서 Lua를 한 번 원자 배치했다.
파일만 교체했으며 engine 재시작·desktop 활성화·클릭·키 입력·AI setter·vJoy 입력은0이다.
새 generation은 `4525688ULL:0.001221`, 배포 SHA256은
`4baf7f0b6293cca9e5b24a3ed941e6ac76cab8904a630ae6799d72ed9886c342`다.

t1516의 원래 request는 `mmf-probe-148fcb71ef6a4594b74f7f2dc53f5c09`다.

- 8묶음·네 시점 총32개 RGB 화면이 같은-shot producer PNG와 정확히 일치했다.
- producer frame metadata 파일0개, consumer ACK 파일0개였다.
- GPU 복사 후 ACK를 읽은 producer의 PNG reference identity도 일치했다.
- 원본 수신 JSON 바이트·ACK JSON 바이트·수신 SHA receipt·GPU receipt·RGB NPY·PNG를 보존했다.
- producer는8묶음 `completed`, host probe 종료0. game window는 foreground=false였다.

t1517은 별도 프로그램으로 보존된 JSON/SHA/캡처 identity/실제 host 수신→GPU copy→ACK 순서와
32개의 원시 RGB/PNG를 다시 비교했고 오차0으로 종료0이다.
원시 map 전체 dump를 보존한 시험은 아니다. 수신 payload와 SHA/sequence receipt를 보존했으며,
packet 경계 동작은 양방향 LuaJIT/Windows 검사와 함께 판단한다.

## 검증 범위와 다음 작업

이 시험에서는 metadata·ACK의 프레임별 파일만 제거했다. PNG 비교 파일과 request/terminal/상태·clock 파일은 남는다.
635ms 원본 공백의 원인이 해결됐거나 주행 중 지연이 줄었다는 결과는 아직 없다.
기본 실시간 recorder는 여전히 파일 metadata/ACK를 사용한다. 새 코너 train/dev 자료0개와
파리 시각 조향 개발 실패·학생 미배치는 그대로다.

다음은 별도 v3 실시간 stream에 이 경로를 연결하는 것이다. 원본 metadata/ACK payload를
bounded RAM 또는 별도 writer로 모두 보존하고, 실제 준비시각에 receipt를 기록해야 한다.
파일이 아직 없다는 이유로 기존 receipt를 다시 매기지 않는다. 종료/파일 hash/원시 재생 검증 전 fit은 금지한다.
정차 자격과 주행 수집 자격을 구분하고, 새 주행 계약/소스를 먼저 고정한 뒤3초→30초 시험으로 진행한다.
그 후에만 원래 역할을 보존한 geographic 코너 수집과 파리 시각→운동 회로 학습을 이어간다.

## 보존

모든 소유 배치/시험/감사는 terminal이다. 마지막 상태는 paused/AI-off/camera-off/probe-off/full brake다.
native/CSP AI flag 불일치는 별도 미해결이며, 이번 읽기 전용 픽셀 검증을 학생 제어권 인증으로 쓰지 않는다.

- 수정 전 Lua: `C:/Users/jeff4/AppData/Local/flybrain/execution-snapshots/t1514-before-mmf`.
- 실행: local `runtime-runs/20261029-t1515-*`, `t1516-*`, `t1517-*`.
- NAS: `runs/ac-paused-mmf-qualification-20261029-t1518`. 정차 시험의 작은 원시 payload까지 복사/해시 검증한다.
- 이전 실패: t1508~t1513을 보존한다. 기존 geographic collection plan의 source hash는 이제 달라졌으므로
  같은 실패 batch/plan을 재사용하지 않는다.
