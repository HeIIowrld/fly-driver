# 저장 공간 정리 — 2026-11-02

사용자 요청에 따라 지금 사용하지 않는 종료된 자료를 C에서 D로 옮겼다. 파일별 SHA-256·크기·원본 변경 여부와 사용 중인 프로세스 부재를 확인한 뒤에만 C 사본을 제거했다. 원래 C 경로는 D를 가리키는 junction으로 보존했다. D가 연결되지 않으면 해당 경로에 접근할 수 없다.

- 종료된 flybrain 준비 자료 `C:/Users/jeff4/.codex/flybrain-stage-20261027`: 654파일, 710,939,173바이트를 `D:/CodexStorage/flybrain-stage-archive-20261102/flybrain-stage-20261027`로 이동했다. 원본과 사본 및 junction을 통한 해시를 확인했다. 대상의 상속 권한을 차단하고 기존 `.codex`의 사용자·SYSTEM·관리자·CodexSandboxUsers 접근 규칙으로 제한했다. [파일별 영수증](D:/CodexStorage/flybrain-stage-archive-20261102/receipt.json).
- 종료된 개발 기록 `C:/Users/jeff4/.cache/ecl/terra924-openlibrary-development-r1`: 5,870파일, 8,611,492,592바이트를 `D:/CodexResearchRuns/ecl-relocated-20261102/terra924-openlibrary-development-r1`로 이동했다. ledger에서 active/queued/new unresolved 0, 6개 slot closed 및 각 모델 실행 종료, 잠금 미사용, 해당 경로 프로세스 부재를 확인했다. 후속 grading은 미시작이며 파일과 원래 경로를 보존했다. D 대상 접근 규칙의 상속을 차단하고 원래 접근 규칙을 보존했다. [결과](D:/CodexResearchRuns/ecl-relocated-20261102/terra924-result.json), [파일별 영수증](D:/CodexResearchRuns/ecl-relocated-20261102/terra924-verified-files.json).

두 항목 총 9,322,431,765바이트(약 8.68GiB)를 이동했다. 이 작업 전후 C 여유는 126,017,404,928바이트(약 117.36GiB)에서 135,241,924,608바이트(약 125.95GiB)로 바뀌었다. 동시 시스템 사용량 때문에 차이를 정리 파일의 정확한 물리 회수량으로 단정하지 않는다. 사용 중인 환경, 학습 원본과 미확인 세션은 이동하거나 삭제하지 않았다.
