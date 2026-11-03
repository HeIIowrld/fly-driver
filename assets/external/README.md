# External data

이 폴더의 실제 자산은 Git에서 제외한다. `configs/assets.json`에서 출처·준비 상태를
확인한다. 등록된 URL이 다운로드 완료나 사용 권한 확보를 뜻하지 않는다.

측정 커넥톰은 정규화한 `nodes.csv`와 `edges.csv`를 가져와 `flybrain import-graph`로
변환한다. 노드는 `id,type,role`이며 role은 `input`, `internal`, `output` 중 하나다.
연결은 `pre_id,post_id,synapse_count`다. input/output 선택은 유형·좌우·시각 위치를
검토한 뒤 명시해야 한다. 모든 감각 또는 운동 뉴런을 자동 포트로 지정하지 않는다.

실측 코스에는 XY뿐 아니라 Z, 좌우 경계, 법선·bank와 원본 hash/라이선스/좌표계를
남긴다. DEM 높이를 붙인 근사 코스, 시각 mesh, 물리 접촉 surface를 구분한다.
