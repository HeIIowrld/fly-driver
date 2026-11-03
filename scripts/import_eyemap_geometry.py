"""Convert pinned authors' reconstructed microCT axes; never execute their R code."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np

from flybrain_drive.eye_anatomy import split_axes_by_lens
from flybrain_drive.graph import sha256

PINNED_COMMIT = "99d2a43123db636cedb55af9ff31a59657e7d17e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import rdata

    source = Path(args.source)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                      text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Upstream revision differs from the reviewed source")
    raw_names = ["LICENSE", "README.md", "proc_uCT.R", "Fig_0.R", "data/microCT/20240701.RData",
                 "data/microCT/20240701_position/lens.csv",
                 "data/microCT/20240701_position/cone.csv"]
    hashes, upstream_hashes = {}, {}
    for name in raw_names:
        path = source / name
        actual = path.read_bytes()
        expected = subprocess.check_output(["git", "-C", str(source), "show", commit + ":" + name])
        if not name.endswith(".RData"):
            actual, expected = actual.replace(b"\r\n", b"\n"), expected.replace(b"\r\n", b"\n")
        if actual != expected:
            raise ValueError("Local source file differs from pinned commit: " + name)
        hashes[name] = sha256(path)
        upstream_hashes[name] = hashlib.sha256(expected).hexdigest()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        data = rdata.read_rda(source / "data/microCT/20240701.RData")
    basis = np.column_stack([data["vf"], data["vleft"], data["vn"]])
    match = np.asarray(data["i_match"])
    if not np.isfinite(match).all() or not np.array_equal(match, np.rint(match)):
        raise ValueError("R lens indices are not exact integers")
    arrays = split_axes_by_lens(np.asarray(data["ucl_rot_sm"]), match.astype(np.int64),
                                data["ind_left_lens"], data["ind_left_cone"], basis)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name in raw_names:
        destination = out / "upstream" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    np.savez_compressed(out / "eye-axes.npz", **arrays)
    import flybrain_drive.eye_anatomy as implementation
    metadata = {
        "source_kind": "authors_processed_microCT_reconstructed_smoothed_optical_axes",
        "upstream_repository": "https://github.com/reiserlab/eyemap_T4",
        "upstream_commit": commit, "paper": "https://doi.org/10.1038/s41586-025-09276-5",
        "upstream_file_sha256": hashes, "repository_license": "GPL-3.0; original LICENSE retained",
        "upstream_LF_or_binary_sha256": upstream_hashes,
        "source_validation": "exact RData; text normalized only from CRLF to LF for Git comparison",
        "geometry_sha256": sha256(out / "eye-axes.npz"),
        "script_sha256": sha256(__file__), "implementation_sha256": sha256(implementation.__file__),
        "conversion": "R order(i_match), split ind_left_lens; axes already canonical",
        "head_axes": "X forward, Y left, Z up, following proc_uCT.R vf/vleft/vn basis",
        "facet_row_namespace": "one-based processed lens rows, not neuron body IDs",
        "left_facets": len(arrays["left_lens_rows"]),
        "right_facets": len(arrays["right_lens_rows"]),
        "max_unit_vector_error": float(max(np.max(abs(np.linalg.norm(
            arrays[side + "_directions_head"], axis=1) - 1)) for side in ("left", "right"))),
        "basis_determinant": float(np.linalg.det(basis)),
        "parser_versions": {name: importlib.metadata.version(name) for name in ("rdata", "numpy")},
        "parser_warnings": [str(w.message) for w in captured],
        "R_class_handling": "solve_LSAP values checked as integer permutation; PCA wrapper unused",
        "limitations": ["acceptance angles and photoreceptor temporal responses not imported",
                        "no registration into individual MaleCNS retinal neurons",
                        "virtual head pose/origins must be declared by the consuming backend",
                        "927-node driving checkpoint still uses the original synthetic grid"],
    }
    (out / "eye-axes.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: metadata[key] for key in ("left_facets", "right_facets",
                                                    "max_unit_vector_error", "geometry_sha256")}))


if __name__ == "__main__":
    main()
