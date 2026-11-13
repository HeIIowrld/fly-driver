"""Recover a completed Highlands camera bundle from its immutable producer and consumer files."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from flybrain_drive.ac_camera_archive import bundle_camera_files, CameraArchive
from flybrain_drive.ac_metadata_mmf import verify_mmf_archive_row
from flybrain_drive.graph import sha256 as digest


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("request","producer","consumer","layout_before","race_ini","output"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    request=read(args.request)
    terminal=args.producer/"result.json"
    proof=read(terminal)
    required=request["frames"]
    if proof.get("status")!="completed" or proof["request_id"]!=request["id"] or proof["frames"]!=required:
        raise ValueError("Only terminal, complete original camera data can be recovered")
    import configparser
    config=configparser.ConfigParser(interpolation=None,strict=False)
    config.read(args.race_ini,encoding="utf-8-sig")
    if (args.race_ini.read_bytes()!=args.layout_before.read_bytes()
            or config.get("RACE","TRACK")!="ks_highlands" or config.get("RACE","CONFIG_TRACK")!="layout_short"):
        raise ValueError("Active Highlands Short layout differs from the verified witness")
    stream=args.consumer/"stream.jsonl"
    rows=[json.loads(line) for line in stream.read_bytes().splitlines()]
    if [r["capture_id"] for r in rows]!=list(range(required)):
        raise ValueError("Consumer stream is incomplete")
    if any(r["generation"]!=request["expected_generation"] or r["ideal_line_shown"] or r["aid_probe_only"] for r in rows):
        raise ValueError("Unexpected generation or driver aid in the camera stream")
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"result.json").exists():
        raise FileExistsError(args.output/"result.json")
    destination=args.output/"vision"
    archive_proof = bundle_camera_files((args.producer,args.consumer),destination) if not destination.exists() else {
        "bundle_sha256":read(destination/"bundle.json")["sha256"], "all_hashes_verified":True,
        "source_retained":True,"format":"zip_stored"}
    with CameraArchive(destination) as archive:
        consumer=archive.json("consumer-result.json")
        journal=archive.json("frame-persistence-journal.json")
        stream_lines=archive.read("stream.jsonl").splitlines(keepends=True)
        if (consumer["frames"]!=required or consumer["error"] is not None or not journal["closed"]
                or journal["submitted"]!=required or journal["completed"]!=required or journal["failed"]!=0
                or journal["error"] is not None or len(journal["records"])!=required
                or journal["schema"]!="flybrain.immutable_frame_persistence.v2"):
            raise ValueError("Default MMF writer did not finish every group")
        for row, line, record in zip(rows,stream_lines,journal["records"],strict=True):
            if (record["group_id"]!=row["capture_id"] or record["status"]!="persisted"
                    or record["stream_line_sha256"]!=sha256(line).hexdigest() or len(record["files"])!=7
                    or len(row["views"])!=4 or {v["view_id"] for v in row["views"]}!={"front","left","rear","right"}):
                raise ValueError("Immutable frame receipt differs")
            if len({f["file"] for f in record["files"]})!=7:
                raise ValueError("Immutable frame receipt repeats a file")
            for f in record["files"]:
                payload=archive.read(f["file"])
                if len(payload)!=f["bytes"] or sha256(payload).hexdigest()!=f["sha256"]:
                    raise ValueError("Original persistence file differs")
            verify_mmf_archive_row(archive,row)
            name=Path(row["file"]).name
            meta=archive.json(name)
            if (meta["track_id"],meta["car_id"],meta["index"],meta["generation"])!=(
                    "ks_highlands","bmw_z4_gt3",row["capture_id"],row["generation"]):
                raise ValueError("Camera frame identity differs")
    mapping=args.output/"archive-map.json"
    write(mapping,{"ks_highlands":str(destination.resolve())})
    write(args.output/"result.json",dict(status="captured",track="ks_highlands",layout="layout_short",
        layout_source="unchanged_active_race_ini",frames=required,actor_control_sent=False,
        archive=archive_proof,producer_terminal=proof,archives_sha256=digest(mapping),
        race_ini_sha256=digest(args.race_ini),consumer_stream_sha256=digest(stream),
        program_sha256=digest(__file__),camera_request_sha256=digest(args.request),seed=None,
        termination="recovered_complete_original_capture"))
    print("Archived",required,"frames",archive_proof["bundle_sha256"],flush=True)


if __name__=="__main__":
    main()
