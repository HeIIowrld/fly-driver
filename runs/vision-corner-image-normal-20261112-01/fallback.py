"""Re-score image-normal proposals as a fallback for previously abstained frames."""
from pathlib import Path

from baseline import SILVER,fit_boundaries,publish,summarize_approaches
from evaluate_fused_corner import frame_key
from heads import review
from recover import baseline_data
from score_corner_captures import summarize
from train_vision_teacher_line import digest,jsonl,read,write
from evaluate_fused_corner import read_rows

OUT=Path(__file__).parent


def main():
    hashes={Path(__file__),Path(baseline_data.__code__.co_filename),Path(fit_boundaries.__code__.co_filename)}
    source=OUT/"result.json"
    result=read(source)
    hashes.add(source)
    details=[]
    output={}
    for track in (SILVER,"spa"):
        group,edges,_,_=baseline_data(track,hashes)
        evidence=OUT/f"{track}-frames.jsonl"
        assert digest(evidence)==result["tracks"][track]["rows_sha256"]
        hashes.add(evidence)
        saved={frame_key(row):row["methods"] for row in read_rows(evidence)}
        baseline=[]
        rescued=[]
        for row in group["rows"]:
            key=frame_key(row)
            supported,proposal=fit_boundaries(edges[key],.002)["hinge"]
            old=publish(row,proposal,supported)
            known=saved.get(key)
            if known is not None:
                a=known["baseline"]
                proposal = None
                if a["state"] in ("left","right"):
                    proposal=(1 if a["state"]=="left" else 2,a["distance_m"])
                old=publish(row,proposal,a["observable"])
                assert (old["state"],old["predicted_distance_m"])==(a["state"],a["distance_m"]),key
            new=old
            if old["state"]=="abstain" and known is not None:
                candidate=known["normal_fill"]
                if candidate["state"] in ("left","right"):
                    new=publish(row,(1 if candidate["state"]=="left" else 2,candidate["distance_m"]),True)
                    details.append(dict(track=track,run=key[0],frame=key[1],truth=row["truth"],
                        label=row["label"],distance_m=new["predicted_distance_m"],
                        direction_hit=new["direction_hit"],joint_hit=new["joint_hit"]))
            baseline.append(old)
            rescued.append(new)
        old_score=review(group,baseline)
        assert old_score["metrics"]==result["tracks"][track]["methods"]["baseline"]["score"]["metrics"]
        score=review(group,rescued)
        output[track]=dict(original=old_score,fallback=score,approach=summarize_approaches(rescued),
            distance_change=summarize(rescued)["distance_change"])
        print(track,score["metrics"]["candidate"],flush=True)
    path=OUT/"fallback-alerts.jsonl"
    jsonl(path,details)
    write(OUT/"fallback-result.json",dict(status="evaluated",seed=None,tracks=output,rows_sha256=digest(path),
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
