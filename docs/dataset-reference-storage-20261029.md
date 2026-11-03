# 학습 데이터 참조 저장과 과거 중복 정리

갱신: 2026-10-29 11:28 KST

새 병합은 원본을 해시 검증하여 참조하는 방식이 기본이다. 과거 중복 사본도 후속 정리에서 214개, 약 10.34GB를 실제 삭제했고, 고유 원본 67종과 과거 manifest·실행 기록은 유지했다. 기존 경로는 별도 참조표와 수정된 로더로 연결한다.

[실제 회수량·경로 호환·실행 검증](historical-episode-compaction-20261029.md)을 확인한다. 앞선 11:13 검토 당시의 "기존 사본 유지·회수 0GB" 상태는 후속 삭제 완료로 대체되었다.

새 병합은 `scripts/merge_multiview_teacher.py --input runs/source --output runs/new-union`을 사용한다. 독립 NPZ 사본이 필요한 export만 `--storage copy`를 지정한다. 참조가 남은 원본과 `.episode-links.json`은 보존한다.
