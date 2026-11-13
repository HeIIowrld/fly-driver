"""Save the two observed road edges from the frozen causal camera replay."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import relative_corner_boundaries as relative
from corner_fusion_boundaries import stations, whole_shape_min
from evaluate_fused_corner import frame_key, load_frozen, load_score
from profile_replay import TRACKS, archive_path, sources
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent


def run(track):
    final, base, protocol_path = sources(track)
    protocol = load_frozen(protocol_path)
    config = Path(protocol["camera_config"])
    settings_path = Path(protocol["boundary_fusion_pins"])
    final_path, base_path = final / "result.json", base / "result.json"
    proof = read(final_path)
    mapping = archive_path(track)
    if (digest(mapping) != read(base_path)["archives_sha256"]
            or digest(protocol_path) != proof["protocol_sha256"]
            or digest(base_path) != proof["base_receipt_sha256"]):
        raise ValueError(f"Changed boundary inputs for {track}")
    rows = load_score(final / "score", proof["score_receipt_sha256"])
    snapshot = None

    def observe(fused, aligned):
        nonlocal snapshot
        snapshot = np.stack((aligned[-1], fused))
        return whole_shape_min(fused, aligned)

    observations = []
    with patch.object(relative, "whole_shape_min", observe):
        replay = relative.read_boundaries(mapping, proof["camera_bundle_sha256_by_run"], track,
                                          config, load_frozen(settings_path))
        for row, (key, _, _) in zip(rows, replay, strict=True):
            if frame_key(row) != key:
                raise ValueError(f"Frame order mismatch: {key}")
            observations.append(snapshot if snapshot is not None else np.full((2, len(stations), 2), np.nan))
            snapshot = None

    path = OUT / f"{track}-sections.npz"
    np.savez_compressed(path, values=np.stack(observations), stations=stations,
                        frame=np.array([r["frame"] for r in rows]),
                        run=np.array([r["run"] for r in rows]))
    sources_used = (Path(__file__), mapping, protocol_path, config, settings_path, final_path, base_path,
                    Path("scripts/relative_corner_boundaries.py"), Path("scripts/corner_fusion_boundaries.py"))
    write(OUT / f"{track}-sections.json", dict(track=track, frames=len(rows), revision=proof["revision"],
          sections_sha256=digest(path), input_sha256={str(p):digest(p) for p in sources_used},
          camera_bundle_sha256_by_run=proof["camera_bundle_sha256_by_run"],
          termination="full_camera_boundary_replay_complete"))
    print(track, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", nargs="+", choices=TRACKS)
    for name in parser.parse_args().tracks:
        run(name)
