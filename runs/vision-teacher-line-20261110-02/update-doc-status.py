from pathlib import Path
entry = '**몬자 영상 기반 교사 경로점 — 2026-11-10.** 5·10·20m 교사 라벨과 다음 굽이 예고를 학습했다. 별도 몬자 랩에서 영상만/영상+속도 20m 횡오차는 둘 다 약 0.50m였고 두 미사용 코너는 영상만 5.10m/2.15m였다. 학습·저장 평가 완료. 다른 서킷, FlyVis 대조, AC 폐루프와 더 빠른 라인은 미검증이다. [실행·실패·다음 검증](REPORT_LINK).'
for name in ('README.md','docs/CONTEXT.md','docs/research-plan.md','docs/implementation-status.md'):
 p=Path(name)
 raw=p.read_bytes()
 i=raw.index(b'\n')
 ending=b'\r\n' if raw[:i].endswith(b'\r') else b'\n'
 link='docs/vision-teacher-line-20261110.md' if name=='README.md' else 'vision-teacher-line-20261110.md'
 body=entry.replace('REPORT_LINK',link)
 p.write_bytes(raw[:i+1]+ending+body.encode()+ending+raw[i+1:])
