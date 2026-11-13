"""Audit paired map and image labels without exposing the installed mesh."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from kn5_surfaces import OUT, digest

RUNS = OUT.parent
ROOT = RUNS.parent
MANUAL = RUNS / "vision-corner-visible-line-labels-20261112-01" / "labels.json"
FRAMES = (488, 489, 490, 620, 621, 622, 623, 807, 808, 809, 1048, 1049, 1050)
MODES = ("image", "mesh", "adjacent_invalid_only")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def distance(point, line):
    start, end = line[:-1], line[1:]
    delta = end - start
    fraction = np.clip(((point - start) * delta).sum(axis=1) / (delta * delta).sum(axis=1), 0., 1.)
    return float(np.min(np.linalg.norm(point - start - fraction[:, None] * delta, axis=1)))


def image_report(labels, annotation):
    stats = dict(common_mesh_to_visible_px=[], common_ai_to_visible_px=[],
                 common_mesh_to_competitor_px=[], common_ai_to_competitor_px=[])
    if annotation["target"] is None:
        return stats
    target = np.asarray(annotation["target"], dtype=float)
    alternate = np.asarray(annotation.get("competitor", []), dtype=float)
    for label in labels:
        if label["station_m"] < 16. or label["side"] != "right" or not label["witnessed"]:
            continue
        a, b = label["pixel"], label["ai_pixel"]
        if any(p is None or not target[0, 0] <= p[0] <= target[-1, 0] for p in (a, b)):
            continue
        stats["common_mesh_to_visible_px"].append(distance(a, target))
        stats["common_ai_to_visible_px"].append(distance(b, target))
        if len(alternate):
            stats["common_mesh_to_competitor_px"].append(distance(a, alternate))
            stats["common_ai_to_competitor_px"].append(distance(b, alternate))
    return stats


def plot(frames, manual):
    out = Image.new("RGB", (1188, 4 * 163), "#181818")
    draw = ImageDraw.Draw(out)
    box = (192, 83, 320, 130)
    scale = 3
    for j, number in enumerate((620, 621, 622, 623)):
        key = str(number)
        label = manual[key]
        crop = Image.open(ROOT / label["source_path"]).convert("RGB").crop(box)
        crop = crop.resize(((box[2]-box[0])*scale, (box[3]-box[1])*scale), Image.Resampling.NEAREST)
        for i, kind in enumerate(("Road-facing strip", "Configured surface transition", "AI corridor")):
            x, y = 6 + i*396, 16 + j*163
            out.paste(crop, (x,y))
            draw.text((x,y-14), f"{key}  {kind}", fill="white")

            def pixel(p):
                return (round(x + (p[0]-box[0])*scale), round(y + (p[1]-box[1])*scale))

            if i == 0:
                draw.line([pixel(p) for p in label["target"]], fill="#50E3AD", width=2)
                draw.line([pixel(p) for p in label["competitor"]], fill="#FFB85E", width=2)
                continue
            for r in frames[key]["labels"]:
                p = r["pixel" if i == 1 else "ai_pixel"]
                if (r["side"] != "right" or r["station_m"] < 16 or p is None or
                        not (box[0] <= p[0] < box[2] and box[1] <= p[1] < box[3])):
                    continue
                px, py = pixel(p)
                color = "#50E3AD" if i == 1 and r["witnessed"] else "#FFB85E"
                draw.ellipse((px-2,py-2,px+2,py+2), fill=color)
                draw.text((px-3,py+4+(r["station_m"]%4)*2), str(int(r["station_m"])), fill="white")
    path = OUT / "surface-vs-strip.png"
    out.save(path)
    return digest(path)


def scores(rows, mode):
    count = Counter()
    for r in rows:
        offer = r[mode]
        if r["truth"] == "approach":
            side = "right" if r["label"] == "2" else "left"
            count[f"{side}_frames"] += 1
            if offer and int(r["label"]) == offer[0]:
                count[f"{side}_direction"] += 1
                lo, hi = r["truth_endpoints_m"]
                if lo - 5. <= offer[1] <= hi + 5.:
                    count[f"{side}_joint"] += 1
        if offer and r["truth"] in ("straight", "out_of_range"):
            count["outside_alerts"] += 1
    return dict(count)


def main():
    result_path = OUT / "surface-labels.json"
    result = read(result_path)
    assert result["frames"] == result["scored_mesh_frames"] == 1200
    assert not result["actor_input"] and result["source_asset_copies"] == 0
    for filename, sha in result["source_sha256"].items():
        assert digest(ROOT / filename) == sha, filename
    rows_path = OUT / "mesh-oracle.jsonl"
    assert digest(rows_path) == result["mesh_oracle_sha256"]
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf8").splitlines()]
    assert len(rows) == len({r["frame"] for r in rows}) == 1200
    manual = read(MANUAL)["frames"]
    assert set(result["labels"]) == {str(f) for f in FRAMES}
    transition = Counter()
    for frame, info in result["labels"].items():
        assert digest(ROOT / manual[frame]["source_path"]) == manual[frame]["sha256"]
        for r in info["labels"]:
            if r["side"] == "right" and r["station_m"] >= 16.:
                transition[r["inside"], "+".join(r["outside"]) or "unconfirmed"] += 1
    comparisons = {frame: image_report(info["labels"], manual[frame])
                   for frame, info in result["labels"].items()}
    errors = {}
    for key in next(iter(comparisons.values())):
        values = [v for frame in comparisons.values() for v in frame[key]]
        errors[key] = dict(points=len(values), median_px=float(np.median(values)) if values else None)
    data = dict(status="verified_diagnostic", source_revision=result["source_revision"], seed=None,
                source_result_sha256=digest(result_path), verifier_sha256=digest(Path(__file__)),
                overlay_sha256=plot(result["labels"], manual), comparisons=comparisons,
                common_pixel_summary=errors, transition_counts={f"{a} to {b}": n for (a,b), n in transition.items()},
                far_comparison={m: scores(rows, m) for m in MODES},
                termination="hashes_image_alignment_and_1200_diagnostic_frames_verified")
    (OUT / "verification.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k:v for k,v in data.items() if k in ("common_pixel_summary","transition_counts","far_comparison")},
                     indent=2))


if __name__ == "__main__":
    main()
