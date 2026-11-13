"""Screen additional D-drive research archives without reading the image payloads."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import BadZipFile, ZipFile

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0, str(RUNS/"vision-corner-independent-source-rescreen-20261111-01"))
sys.path.insert(0, str(RUNS.parent/"scripts"))
from rescreen import capture_name, screen
from train_vision_teacher_line import digest, read, write

ROOTS = tuple(Path(x) for x in ("D:/CodexStorage","D:/CodexResearchFixtures","D:/02_Personal","D:/03_SW"))
PREVIOUS = RUNS/"vision-corner-fixed-left-20261113-01/independent-inventory.json"
KNOWN_TRACKS = {"ks_silverstone1967","ks_vallelunga","monza","spa","magione","imola",
                "mugello","ks_laguna_seca","ks_zandvoort"}
TRACK_FILE = re.compile(r"(^|[/\\])(stream\.jsonl|camera[^/\\]*\.zip|[^/\\]+\.acreplay)$",re.I)


def inventory(root):
    result = subprocess.run(["rg","--files","--hidden","--no-ignore",str(root),"-g","*.acreplay",
                             "-g","*camera*.zip","-g","stream.jsonl","-g","archive-map.json","-g","bundle.json",
                             "-g","archive.json","-g","*Highland*","-g","*highland*","-g","*flybrain*.zip",
                             "-g","*camera*.mp4","-g","*vision*.mp4"],
                            capture_output=True,text=True,encoding="utf-8")
    if result.returncode not in (0,1,2):
        raise RuntimeError(result.stderr)
    return [Path(x) for x in result.stdout.splitlines()],result.stderr.strip().splitlines()


def zip_tracks(path):
    result = dict(path=str(path), frame_count=0, tracks=[], status="no_capture_stream")
    with ZipFile(path) as z:
        names = z.namelist()
        stream = next((x for x in names if Path(x).name=="stream.jsonl"),None)
        if not stream:
            return result
        prefix = str(Path(stream).parent).replace("\\","/").rstrip("/")
        prefix = "" if prefix=="." else prefix+"/"
        rows = [json.loads(r) for r in z.read(stream).splitlines() if r.strip()]
        result["frame_count"] = len(rows)
        track = Counter()
        for row in rows:
            name = prefix+capture_name(row)
            metadata = json.loads(z.read(name))
            track[metadata.get("track_id","unknown")] += 1
        result.update(tracks=dict(track),status="track_identity_from_internal_metadata")
    return result


def main():
    roots, archives, standalones, other = {}, [], [], []
    for root in ROOTS:
        paths, errors = inventory(root)
        folders = {p.parent for p in paths if p.name in {"stream.jsonl","camera.zip","archive.json","bundle.json"}}
        for folder in sorted(folders):
            try:
                result = screen(folder)
            except (OSError,BadZipFile,ValueError,KeyError,TypeError) as exc:
                result = dict(status="rejected",reason=str(exc))
            archives.append(dict(path=str(folder),**result))
        for p in paths:
            if p.suffix.lower()==".zip" and (p.name!="camera.zip" or not (p.parent/"archive.json").is_file()):
                try:
                    standalones.append(zip_tracks(p))
                except (OSError,BadZipFile,ValueError,KeyError,TypeError) as exc:
                    standalones.append(dict(path=str(p),status="rejected",reason=str(exc)))
            if p.suffix.lower() in (".acreplay",".mp4") or "highland" in p.name.lower():
                other.append(str(p))
        roots[str(root)] = dict(filenames_matched=len(paths),camera_folders=len(folders),enumeration_errors=errors)
    tracks = Counter(t for r in archives for t in r.get("tracks",[]))
    candidates = [r for r in archives if set(r.get("tracks",[]))-KNOWN_TRACKS]
    zip_candidates = [r for r in standalones if set(r.get("tracks",[]))-KNOWN_TRACKS-{"unknown"}]
    file = OUT/"d-drive-archive-inventory.json"
    write(file,dict(roots=roots,camera_folders=archives,camera_archive_counts=dict(tracks),
                    standalone_zips=standalones,other_replay_or_named_candidates=other,
                    eligible_new_track_folders=candidates,unseen_track_zip_candidates=zip_candidates,
                    prior_inventory_sha256=digest(PREVIOUS),source_sha256=digest(Path(__file__)),
                    seed=None,termination="metadata_inventory_complete",images_decoded=0))
    print(dict(tracks),len(archives),len(standalones),len(other),candidates,zip_candidates,flush=True)


if __name__=="__main__":
    main()
