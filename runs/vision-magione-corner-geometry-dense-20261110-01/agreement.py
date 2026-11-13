import json
from collections import Counter
from pathlib import Path

from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
source = out / "rows.jsonl"
assert digest(source) == read(out / "result.json")["rows_sha256"]
rows = [json.loads(line) for line in source.read_text(encoding="utf8").splitlines()]
summary = {}
for threshold in (.025, .05, .10):
    counts = Counter()
    for row in rows:
        turns = row["turns"]
        hit = next(((x,a) for (x,a),(xx,b) in zip(turns,turns[1:])
                    if xx==x+2 and a*b>0 and min(abs(a),abs(b))>=threshold and x<=20), None)
        if hit is None:
            continue
        if not row["near"]:
            counts["outside"] += 1
            continue
        direction = 1 if hit[1]>0 else 2
        if direction == row["label"]["direction"]:
            counts[f"direction_{direction}"] += 1
            counts["distance_hit"] += abs(hit[0]-row["label"]["distance"])<=5
    summary[str(threshold)] = dict(counts)
write(out/"spatial-agreement.json",dict(summary=summary,source_sha256=digest(source),
    seed=None,termination="archive_end",program_sha256=digest(Path(__file__))))
print(summary)
