from collections import Counter
from pathlib import Path
import hashlib,json,subprocess,zipfile
from train_vision_teacher_line import digest,write

out=Path(__file__).parent
files=subprocess.run(["rg","--files","--hidden","--no-ignore","runs","-g","bundle.json"],
                     capture_output=True,text=True,encoding="utf8",check=True).stdout.splitlines()
records=[]
for filename in sorted(files):
    descriptor=Path(filename)
    info=json.loads(descriptor.read_bytes())
    if info.get("schema")!="flybrain.csp.camera_bundle.v1" or info.get("file")!="camera.zip":
        continue
    archive=descriptor.parent/"archive.json"
    row=dict(directory=str(descriptor.parent),descriptor_sha256=digest(descriptor),bundle_sha256=info["sha256"])
    try:
        assert digest(archive)==info["archive_manifest_sha256"]
        entries={r["file"]:r for r in json.loads(archive.read_bytes())}
        def verified(z,name):
            raw=z.read(name)
            assert hashlib.sha256(raw).hexdigest()==entries[name]["sha256"]
            return raw
        if "stream.jsonl" not in entries:
            row["status"]="no_stream"
        else:
            with zipfile.ZipFile(descriptor.parent/"camera.zip") as z:
                frames=[json.loads(line) for line in verified(z,"stream.jsonl").splitlines()]
                views=[f for f in frames if {v["view_id"] for v in f["views"]}=={"front","left","right","rear"}
                       and not f["ideal_line_shown"] and not f["aid_probe_only"]]
                ids=set()
                for frame in (views[i] for i in sorted({0,len(views)//2,len(views)-1}) if views):
                    meta=json.loads(verified(z,Path(frame["file"]).name))
                    assert meta["index"]==frame["capture_id"]
                    ids.add(meta["track_id"])
                row.update(status="screened",frames=len(frames),four_view_aid_off_frames=len(views),
                           track_ids=sorted(ids))
    except (OSError,ValueError,KeyError,AssertionError,zipfile.BadZipFile) as err:
        row.update(status="rejected",reason=str(err),error=type(err).__name__)
    records.append(row)
counts=Counter(track for row in records for track in row.get("track_ids",[]))
frames=Counter()
for row in records:
    if len(row.get("track_ids",[]))==1:
        frames[row["track_ids"][0]]+=row["four_view_aid_off_frames"]
engine_names=("acs.exe","acs_x86.exe")
inv=subprocess.run(["tasklist","/FO","CSV","/NH"],capture_output=True,text=True,errors="replace",check=True).stdout
active=[name for name in engine_names if f'"{name}"' in inv.lower()]
write(out/"result.json",dict(revision=subprocess.check_output(
    ["git","-c","safe.directory=*","rev-parse","HEAD"],text=True).strip(),seed=None,
    program_sha256=digest(Path(__file__)),bundle_counts=dict(counts),frame_counts=dict(frames),
    record_status=dict(Counter(r["status"] for r in records)),records=records,active_engine_images=active,
    scope="repository ZIP archive inventory; stream and representative metadata hashes only, no unseen pixel decode",
    termination="completed repository camera bundle source inventory"))
print(dict(counts),dict(frames),dict(Counter(r["status"] for r in records)),active,flush=True)
