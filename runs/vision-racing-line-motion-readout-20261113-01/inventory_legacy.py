"""Read track identifiers inside legacy camera archives omitted by the four-view filter."""
from collections import Counter
from pathlib import Path
import json
from zipfile import ZipFile

OUT=Path(__file__).resolve().parent
INVENTORY=OUT/"d-drive-archive-inventory.json"


def main():
    raw=json.loads(INVENTORY.read_bytes())
    result=[]
    for record in raw["camera_folders"]:
        if record["status"]=="screened":
            continue
        folder=Path(record["path"])
        bundle=folder/"camera.zip"
        if not bundle.is_file():
            continue
        tracks=Counter()
        with ZipFile(bundle) as archive:
            for file in archive.infolist():
                if not file.filename.endswith(".json") or file.file_size>1000000:
                    continue
                info=json.loads(archive.read(file))
                if not isinstance(info,dict):
                    continue
                track=info.get("track_id")
                if isinstance(track,str) and ("index" in info or "generation" in info):
                    tracks[track]+=1
        result.append(dict(folder=str(folder),previous_status=record["status"],tracks=dict(tracks)))
    counts=Counter()
    for r in result:
        counts.update(r["tracks"])
    (OUT/"d-drive-legacy-track-identities.json").write_text(json.dumps(
        dict(archives=result,metadata_counts=dict(counts),termination="metadata_complete"),indent=2)+"\n",encoding="utf-8")
    print(len(result),dict(counts),flush=True)


if __name__=="__main__":
    main()
