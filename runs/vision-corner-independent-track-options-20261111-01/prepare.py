import argparse
from collections import Counter
import configparser
from pathlib import Path

from evaluate_independent_corner import run
from evaluate_live_corner_geometry import load_track_truth
from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
source = out.parent/"vision-corner-independent-protocol-20261111-02/protocol.json"
tracks = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks")
write(out/"empty-archive-map.json", {})
for track_id in ("ks_laguna_seca", "ks_zandvoort"):
    directory = tracks/track_id
    models = configparser.ConfigParser(interpolation=None)
    assert models.read(directory/"models.ini", encoding="utf-8-sig")
    listed = [Path(models[section]["file"]) for section in models if "file" in models[section]]
    assert listed and all((directory/model).is_file() for model in listed)
    ui = directory/"ui/ui_track.json"
    assert ui.is_file()
    geometry = (directory/"ai/fast_lane.ai").resolve()
    _, n, spans = load_track_truth(geometry)
    sides = Counter("left" if sign>0 else "right" for _,_,sign in spans)
    protocol = read(source)
    minimum = protocol["data_admission"]["minimum_distinct_physical_corners_per_side"]
    assert all(sides[side]>=minimum for side in ("left","right"))
    hashes = protocol["hashes"]
    del hashes[str((tracks/"mugello/ai/fast_lane.ai").resolve())]
    hashes.update({str(path):digest(path) for path in (geometry, ui, directory/"models.ini", source, Path(__file__))})
    protocol.update(proposed_track=track_id,
        track_status="Model files and geometry checked; user menu availability and camera capture pending",
        geometry_preflight=dict(installed_spline_points=n, physical_spans_by_side=sides, actor_input=False),
        supersedes_protocol_sha256=digest(source))
    option = out/(track_id+"-protocol.json")
    write(option,protocol)
    denied = out/(track_id+"-empty-admission")
    run(argparse.Namespace(protocol=option,track_dir=directory,archives=out/"empty-archive-map.json",
                           cache=out/"unused-cache",output=denied))
    assert read(denied/"result.json")["status"]=="not_admitted"
    print(track_id,sides)
