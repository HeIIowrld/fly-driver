"""Repeat the held-out protocol with the neighboring-section geometry proposals."""
from pathlib import Path

import transfer_eval as protocol
from evaluate_fused_corner import read_rows
from train_vision_teacher_line import digest,read

OUT=Path(__file__).parent/"neighbor-supported"


def proposals(hashes):
    hashes.update((Path(__file__),Path(protocol.__file__)))
    result={}
    for track in protocol.TRACKS:
        source=protocol.DEVELOPMENT/f"{track}-frames.jsonl"
        receipt=protocol.DEVELOPMENT/f"{track}-development.json"
        assert digest(source)==read(receipt)["rows_sha256"]
        hashes.update((source,receipt))
        chosen=[]
        for row in read_rows(source):
            old=row["methods"]["baseline"]
            new=row["methods"]["normal_supported"]
            if old["state"]=="abstain" and new["state"] in ("left","right"):
                chosen.append(dict(track=track,run=row["run"],frame=row["frame"],truth=row["truth"],
                    label=row["label"],side=new["state"],distance_m=new["distance_m"],
                    direction_hit=new["direction_hit"],joint_hit=new["joint_hit"]))
        result[track]=chosen
    return result


if __name__=="__main__":
    OUT.mkdir(exist_ok=False)
    protocol.OUT=OUT
    protocol.load_proposals=proposals
    protocol.main()
