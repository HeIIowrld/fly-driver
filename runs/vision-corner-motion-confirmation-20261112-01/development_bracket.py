"""Check a Silverstone-derived support rule without changing it on development videos."""
from pathlib import Path

from evaluate_fused_corner import frame_key,read_rows
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent
RECEIPT=OUT/"development-receipts.json"


def main():
    tracks={}
    details=[]
    hashes={Path(__file__),RECEIPT}
    for track,expected in read(RECEIPT)["tracks"].items():
        result_path=OUT/f"{track}-development.json"
        assert digest(result_path)==expected
        result=read(result_path)
        source=OUT/f"{track}-frames.jsonl"
        assert digest(source)==result["rows_sha256"]
        hashes.update((source,result_path))
        base=result["methods"]["baseline"]["score"]["metrics"]["candidate"]
        modes={mode:dict(base) for mode in ("ungated_fallback","bracketed_fallback")}
        for row in read_rows(source):
            original=row["methods"]["baseline"]
            proposed=row["methods"]["normal_fill"]
            if original["state"]!="abstain" or proposed["state"] not in ("left","right"):
                continue
            onset=proposed["distance_m"]
            sections=proposed["added"]
            before=sum(s<=onset for s in sections)
            after=sum(s>onset for s in sections)
            bracketed=before>=2 and after>=2
            outside=row["truth"] in ("straight","out_of_range")
            for name,metric in modes.items():
                if name=="bracketed_fallback" and not bracketed:
                    continue
                if proposed["direction_hit"]:
                    metric[proposed["state"]]+=1
                metric["joint"]+=int(proposed["joint_hit"])
                metric["outside"]+=int(outside)
            details.append(dict(track=track,run=row["run"],frame=row["frame"],truth=row["truth"],
                         label=row["label"],side=proposed["state"],distance_m=onset,
                         direction_hit=proposed["direction_hit"],joint_hit=proposed["joint_hit"],
                         new_at_or_before=before,new_after=after,accepted=bracketed))
        tracks[track]=dict(baseline=base,**modes)
        print(track,tracks[track],flush=True)
    rows=OUT/"development-bracket-proposals.jsonl"
    jsonl(rows,details)
    write(OUT/"development-bracket.json",dict(status="evaluated",seed=None,tracks=tracks,
          rows_sha256=digest(rows),source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
