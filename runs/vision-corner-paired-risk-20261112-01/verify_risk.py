"""Verify the shared candidate and paired risk changes on all three replay timelines."""
from collections import Counter
from pathlib import Path

from paired_risk import OUT, DEVELOP, SILVER, VAL, MONZA, ARCHIVE, METHODS, old, frames


def main():
    training_path, result_path = OUT/"training.json", OUT/"result.json"
    training, result = old.read(training_path), old.read(result_path)
    assert old.digest(training_path)==result["training_sha256"]
    assert training["development"]==list(DEVELOP) and training["excluded"]==[SILVER,VAL,MONZA]
    for name, sha in training["source_sha256"].items():
        assert all(track not in name for track in training["excluded"])
        assert old.digest(Path(name))==sha
    for variant in training["variants"].values():
        assert variant["outside"]<=training["caps"]["outside"] and variant["inside"]<=training["caps"]["inside"]
    winner = "blend_"+str(training["winner"]["weight"])
    replay = {}
    for track,folder in ((SILVER,"silverstone"),(VAL,"vallelunga"),(MONZA,"monza")):
        report = result["tracks"][track]
        for name,sha in report["source_sha256"].items():
            assert old.digest(Path(name))==sha
        evaluated = frames(OUT/f"{track}-frames.jsonl",report["rows_sha256"])
        archived = ARCHIVE/folder/"predictions/forecasts.jsonl"
        expected = frames(archived,old.read(archived.with_name("result.json"))["forecasts_sha256"])
        indexed = {(r["run"],r["frame"]):r for r in expected}
        changes = {}
        for base in METHODS:
            grouped = Counter()
            for row in evaluated:
                match = indexed[row["run"],row["frame"]]
                assert row["candidate"]==match["candidate"]
                a,b = row["outcomes"][base],row["outcomes"][winner]
                if a!=b:
                    grouped[row["truth"],row["label"],a,b]+=1
            changes[base] = [dict(truth=k[0],label=k[1],before=k[2],after=k[3],frames=n)
                             for k,n in sorted(grouped.items())]
        replay[track]=dict(frames=len(evaluated), changes=changes)
    old.write(OUT/"verification.json",dict(status="verified",winner=winner,tracks=replay,
             source_sha256={str(p):old.digest(p) for p in (Path(__file__),training_path,result_path)},
             termination="shared_candidates_and_threshold_changes_verified"))


if __name__=="__main__":
    main()
