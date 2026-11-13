"""Inventory local camera metadata without reading any image payload."""
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from zipfile import BadZipFile,ZipFile

from train_vision_teacher_line import digest,write

OUT=Path(__file__).parent
ROOTS=("runs","assets","D:/CodexResearchRuns")
SEEN={"magione","ks_laguna_seca","ks_zandvoort","mugello","imola","spa","monza"}
VIEWS={"front","left","right","rear"}


def directories():
    for root in ROOTS:
        result=subprocess.run(["rg","--files","--hidden","--no-ignore",root,
            "-g","stream.jsonl","-g","camera.zip"],capture_output=True,text=True,encoding="utf8")
        if result.returncode not in (0,1):
            raise RuntimeError(result.stderr)
        for line in result.stdout.splitlines():
            yield root,Path(line).parent


@contextmanager
def reader(folder):
    bundle=folder/"camera.zip"
    if bundle.is_file():
        with ZipFile(bundle) as zip_file:
            yield zip_file.read
    else:
        yield lambda name:(folder/name).read_bytes()


def capture_name(row):
    name=str(row.get("file","")).replace("\\","/").split("/")[-1]
    if not name or not name.endswith(".json") or name in (".",".."):
        raise ValueError("Missing capture metadata filename")
    return name


def screen(folder):
    manifest=folder/"archive.json"
    if not manifest.is_file():
        return dict(status="no_manifest")
    entries=json.loads(manifest.read_text(encoding="utf-8-sig"))
    members={r["file"]:r["sha256"] for r in entries}
    if len(members)!=len(entries) or "stream.jsonl" not in members:
        return dict(status="no_verified_stream")

    with reader(folder) as raw:
        def checked(name):
            contents=raw(name)
            if sha256(contents).hexdigest()!=members[name]:
                raise ValueError(f"Metadata digest differs for {name}")
            return contents

        stream=checked("stream.jsonl")
        captures=[json.loads(line) for line in stream.splitlines() if line.strip()]
        identity_rows={}
        for row in captures:
            if len(row.get("views",[]))!=4 or {v.get("view_id") for v in row["views"]}!=VIEWS:
                continue
            if row.get("ideal_line_shown") is not False or row.get("aid_probe_only") is not False:
                continue
            identity_rows.setdefault(str(row.get("generation","")),[]).append(row)
        tracks=set()
        for group in identity_rows.values():
            for i in sorted({0,len(group)//2,len(group)-1}):
                row=group[i]
                meta=json.loads(checked(capture_name(row)))
                if meta.get("index")!=row.get("capture_id") or meta.get("generation")!=row.get("generation"):
                    raise ValueError("Capture identity differs")
                if meta.get("track_id"):
                    tracks.add(meta["track_id"])
        eligible=sum(map(len,identity_rows.values()))
        return dict(status="screened" if eligible else "no_four_view_aid_free_frames",
                    captures=len(captures),eligible=eligible,tracks=sorted(tracks),
                    manifest_sha256=digest(manifest),stream_sha256=sha256(stream).hexdigest())


def run():
    records=[]
    for root,folder in sorted(set(directories())):
        try:
            record=screen(folder)
        except (BadZipFile,OSError,ValueError,KeyError,TypeError) as error:
            record=dict(status="rejected",reason=str(error))
        records.append(dict(root=root,directory=str(folder),**record))
    counts=Counter(t for r in records for t in r.get("tracks",[]))
    possible=[r for r in records if set(r.get("tracks",[]))-SEEN]
    write(OUT/"result.json",dict(roots=ROOTS,records=records,counts=counts,
          status=Counter(r["status"] for r in records),possible=possible,
          images_read=0,seed=None,program_sha256=digest(Path(__file__)),termination="independent_metadata_inventory_complete"))
    print(len(records),counts,[(r["directory"],r["tracks"],r["eligible"]) for r in possible],flush=True)


if __name__=="__main__":
    run()
