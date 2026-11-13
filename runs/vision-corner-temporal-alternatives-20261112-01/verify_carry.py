"""Check that exploratory carry adds candidates only to previously empty pools."""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def digest(path):
    with path.open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def main():
    base, changed = (read(folder/"result.json") for folder in (OUT,OUT/"carry"))
    training = read(OUT/"carry/training.json")
    assert digest(OUT/"carry/training.json")==changed["training_sha256"]
    assert digest(OUT/"carry/rankers.joblib")==training["checkpoint_sha256"]
    assert training["carry"]["required_prior_frames"]==2 and not training["carry"]["recursive"]
    for name,sha in training["source_sha256"].items():
        assert digest(ROOT/name)==sha,name
    results = {}
    for track,r in changed["tracks"].items():
        for name,sha in r["source_sha256"].items():
            assert digest(ROOT/name)==sha,name
        a,b = (folder/f"{track}-frames.jsonl" for folder in (OUT,OUT/"carry"))
        assert digest(a)==base["tracks"][track]["rows_sha256"] and digest(b)==r["rows_sha256"]
        original,trial = rows(a),rows(b)
        labels, actual = Counter(),[]
        for i,(x,y) in enumerate(zip(original,trial,strict=True)):
            assert (x["run"],x["frame"],x["truth"],x["label"])==(y["run"],y["frame"],y["truth"],y["label"])
            assert y["scores"]["temporal"]["candidate"]==y["scores"]["t4t5_confirm"]["candidate"]
            if x["pool"]==y["pool"]:
                continue
            assert not x["pool"] and i>=2 and original[i-1]["pool"] and original[i-2]["pool"]
            assert original[i-1]["frame"]+1==original[i]["frame"] and original[i-2]["frame"]+2==original[i]["frame"]
            assert original[i-1]["run"]==original[i-2]["run"]==x["run"]
            labels[x["truth"]] += 1
            actual.append(dict(frame=x["frame"],truth=x["truth"],label=x["label"],pool=y["pool"],
                               final={k:v for k,v in y["final"].items() if k.endswith("candidate_only")}))
        results[track] = dict(new_pool_frames_by_truth=dict(labels),frames=actual)
    data = dict(status="verified_exploratory_only",tracks=results,seed=training["seed"],
                source_revision=training["source_revision"],
                original_result_sha256=digest(OUT/"result.json"),carry_result_sha256=digest(OUT/"carry/result.json"),
                verifier_sha256=digest(Path(__file__)))
    (OUT/"carry-verification.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
