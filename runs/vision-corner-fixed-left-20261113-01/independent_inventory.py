"""Recheck camera archive coverage and pin a separate installed circuit for transfer."""
from collections import Counter
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
INVENTORY = RUNS / "vision-corner-independent-source-rescreen-20261111-01"
sys.path.insert(0, str(INVENTORY))
sys.path.insert(0, str(RUNS.parent / "scripts"))
from rescreen import screen
from train_vision_teacher_line import digest, read, write

TRACK = "ks_highlands"
LAYOUT = "layout_short"
ROOTS = (RUNS, RUNS.parent / "assets", Path("D:/CodexResearchRuns"))
AC_TRACK = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks") / TRACK
TRAINING = {
    "surface_shape": RUNS / "vision-corner-surface-shape-20261112-01/training.json",
    "rank": RUNS / "vision-corner-listwise-20261112-01/training.json",
    "phase": RUNS / "vision-corner-phase-confirmation-20261112-01/training.json",
    "straight": RUNS / "vision-corner-recovery-routing-20261112-01/straight-training.json",
    "width": RUNS / "vision-corner-width-conditioned-boundary-20261112-01/training.json",
    "local_image": RUNS / "vision-corner-local-pixel-recovery-20261112-01/training.json",
    "selected_confirmation": RUNS / "vision-corner-selected-confirmation-20261112-01/training.json",
}


def folders(root):
    result = subprocess.run(["rg", "--files", "--hidden", "--no-ignore", str(root),
                             "-g", "stream.jsonl", "-g", "camera.zip", "-g", "archive.json"],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return {Path(line).parent for line in result.stdout.splitlines()}


def main():
    inventories = []
    tracks = Counter()
    admitted = {}
    for root in ROOTS:
        status = Counter()
        for folder in sorted(folders(root)):
            try:
                info = screen(folder)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                info = dict(status="rejected", reason=str(exc))
            status[info["status"]] += 1
            ids = info.get("tracks", [])
            tracks.update(ids)
            for name in ids:
                admitted.setdefault(name, []).append(dict(folder=str(folder), eligible=info["eligible"],
                                                          manifest_sha256=info["manifest_sha256"]))
        inventories.append(dict(root=str(root), status=dict(status)))
        print(root, dict(status), flush=True)
    roles = {}
    for name, path in TRAINING.items():
        receipt = read(path)
        circuits = receipt.get("development", receipt.get("training_circuits"))
        if not circuits:
            raise ValueError(f"No development circuit provenance for {name}")
        roles[name] = dict(circuits=circuits, sha256=digest(path), file=str(path))
    trained = set().union(*(v["circuits"] for v in roles.values()))
    comparisons = {"ks_silverstone1967", "ks_vallelunga", "monza"}
    eligible = set(admitted) - trained - comparisons
    installed = (AC_TRACK / f"models_{LAYOUT}.ini", AC_TRACK / LAYOUT / "ai/fast_lane.ai",
                 AC_TRACK / LAYOUT / "ai/pit_lane.ai", AC_TRACK / "ui" / LAYOUT / "ui_track.json")
    installed_files = {str(p): digest(p) if p.is_file() else None for p in installed}
    write(OUT / "independent-inventory.json", dict(
        sources=inventories, camera_archive_counts=dict(tracks), new_track_archives={t: admitted[t] for t in eligible},
        recorded_training=roles, compared_tracks=sorted(comparisons),
        reserved_circuit=TRACK, reserved_layout=LAYOUT, installed_files=installed_files,
        training_overlap=TRACK in trained, exact_archive_candidates=admitted.get(TRACK, []),
        prior_inventory_sha256=digest(INVENTORY / "result.json"),
        source_sha256={str(p):digest(p) for p in (Path(__file__), INVENTORY / "rescreen.py")},
        images_read=0, seed=None, termination="metadata_inventory_complete"))
    print(dict(tracks), "eligible:", sorted(eligible), flush=True)


if __name__ == "__main__":
    main()
