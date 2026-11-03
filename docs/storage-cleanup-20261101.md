# 저장 공간 정리 — 2026-11-01

C의 여유 공간은 시작 시 32.11GiB, 작업 직후 53.49GiB였다. 다른 프로세스의 사용량도 변할 수 있어 그 차이 전체를 이 작업의 정확한 물리 회수량이라고 단정하지 않는다.

## 종료된 연구 기록 D 이동

`C:/Users/jeff4/.cache/ecl/terra866-teleport-development-r1`의 종료된 개발 기록 4,235파일, 22,705,621,339바이트(약 21.15GiB)를 `D:/CodexResearchRuns/ecl-relocated-20261101/terra866-teleport-development-r1`로 복사했다. 원본 보유 상태에서 각 파일의 SHA-256·이름·크기를 비교하고 원본 파일 목록과 수정 시각이 변하지 않았음을 재확인했다. 원래 C 경로는 D를 가리키는 Windows junction으로 보존했고 경로를 통한 원본 ledger 접근도 확인한 후 C의 옛 사본을 지웠다. 이동 전 ledger의 active/queued=0, 관련 slot closed와 명시적 실행 중 프로세스 부재를 확인했다. D의 새 디렉터리는 원본과 같은 SYSTEM·Administrators·사용자 full control, CodexSandboxUsers read 접근만 갖도록 기본 상속을 차단했다.

- [이동 결과](D:/CodexResearchRuns/ecl-relocated-20261101/terra866-result.json)
- [파일별 검증 영수증](D:/CodexResearchRuns/ecl-relocated-20261101/terra866-verified-files.json)
- [이동 스크립트](../scripts/relocate_closed_ecl_terra866.ps1)

원본 경로는 junction이므로 D 드라이브가 연결돼 있어야 접근된다. 다른 ecl 실행 기록은 삭제하거나 이동하지 않았다.

## 최근 카메라 원본의 중복 이미지

현재 프로젝트의 종료된 최근 카메라 실행 6개에서 이미지 1,072개, 110,280,669바이트를 삭제했다. 각 원본 픽셀의 크기와 SHA-256이 해당 실행의 보관 ZIP member와 일치하고 ZIP 자체의 해시가 원래 결과와 같음을 확인했다. ZIP 6개와 metadata·원본 실행 결과는 유지했다. ZIP을 통해 읽는 CameraArchive 경로는 유지되지만, 옛 camera spool의 BMP/PNG를 직접 여는 코드는 보관 ZIP에서 해당 파일을 복원해야 한다. 복원에 필요한 원래 경로·member·SHA는 아래 기록에 있다.

첫 실행은 status.json 갱신과 충돌해 571개 삭제 후 중단됐으며, 계획과 삭제 영수증을 대조해 누락된 기록이 없음을 확인했다. 그다음 안정적인 상태 읽기로 남은 501개를 검증하고 삭제했다. 첫 실패를 성공으로 바꾸지 않았다.

- [첫 실행의 검증된 복원 목록](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-recent-camera-duplicate-cleanup/verified-pixels.json)
- [첫 실행의 삭제 기록](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-recent-camera-duplicate-cleanup/deleted.jsonl)
- [남은 501개 정리 결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-recent-camera-duplicate-cleanup-remainder/result.json)

학습 데이터의 역할을 변경하거나 기존 실패 기록을 삭제하지 않았다.

## 동영상 중복 검사 색인 D 이동 — 2026-11-01 추가

`C:/Users/jeff4/AppData/Local/VideoDuplicateTool/cache`의 SQLite 기록 4개, 총 65,669,898,240바이트(61.16GiB)를 `D:/CodexStorage/video-duplicate-cache-20261101/cache`로 보존 이동했다. 복사 중 계산한 원본 SHA-256을 D 사본 전체 재독으로 비교하고, 원본 파일의 목록·크기·수정 시각과 사용 중인 핸들 부재를 확인했다. C의 기존 경로를 D 대상 junction으로 연결해 SQLite 스키마와 page count가 읽힘을 확인한 뒤 확인된 C 사본만 제거했다. 데이터베이스의 의미적 완전성 전체 검사나 해당 앱의 전체 실행 시험은 하지 않았다. D가 연결돼 있어야 원래 경로가 동작한다. [파일별 영수증](D:/CodexStorage/video-duplicate-cache-20261101/receipt.json). 복구용 이동 스크립트는 `C:/Users/jeff4/AppData/Local/Temp/flybrain-relocate-video-cache-20261101.py`에 있다. 이동 완료 직후 C 여유는 약 62.0→123.16GiB였다. 별도로 현재 Python의 pip 다운로드 캐시 14.8MB를 purge했다. 이 정리는 flybrain 학습 원본이나 진행 중인 실행을 삭제하지 않았다.
