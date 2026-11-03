"""Import reconstructed eye axes without inventing neural IDs or optical acceptance."""

import json
from pathlib import Path

import numpy as np

from .compound_eye import EyeGeometry
from .episode_storage import load_array
from .graph import sha256


def split_axes_by_lens(canonical_directions, cone_to_lens, left_lens, left_cone, anatomical_basis):
    axes = np.asarray(canonical_directions, dtype=float)
    match, left, cone_left = map(np.asarray, (cone_to_lens, left_lens, left_cone))
    basis = np.asarray(anatomical_basis, dtype=float)
    if axes.ndim != 2 or axes.shape[1] != 3 or not len(axes):
        raise ValueError("Canonical axes must be a nonempty N x 3 array")
    n = len(axes)
    if (match.shape != (n,) or match.dtype.kind not in "iu"
            or not np.array_equal(np.sort(match), np.arange(1, n + 1))):
        raise ValueError("Cone-to-lens map must be a one-based permutation")
    if (left.shape != (n,) or cone_left.shape != (n,)
            or not np.isin(left, [0, 1]).all() or not np.isin(cone_left, [0, 1]).all()):
        raise ValueError("Left-eye masks must be explicit boolean or zero-one arrays")
    if (not np.isfinite(axes).all() or not np.allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-7)
            or basis.shape != (3, 3) or not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(basis), 1, atol=1e-7)):
        raise ValueError("Unit directions and a proper anatomical basis are required")
    order = np.argsort(match)
    left, cone_left = left.astype(bool), cone_left.astype(bool)
    if not np.array_equal(cone_left[order], left) or not left.any() or left.all():
        raise ValueError("Lens/cone sides disagree or an eye is empty")
    # ucl_rot_sm is already canonical; do not apply anatomical_basis a second time.
    arrays = {}
    for side, keep in (("left", left), ("right", ~left)):
        arrays[side + "_directions_head"] = axes[order][keep]
        arrays[side + "_lens_rows"] = np.arange(1, n + 1)[keep]
        arrays[side + "_cone_rows"] = (order + 1)[keep]
    return arrays


def load_anatomical_eyes(path, origins_body_m, *, sigma_degrees, head_to_body_rotation):
    """Explicit virtual mount and assumed acceptance; no mapping into MaleCNS neurons."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
    if metadata["geometry_sha256"] != sha256(path):
        raise ValueError("Anatomical geometry hash differs from provenance")
    rotation = np.asarray(head_to_body_rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3))
            or not np.isclose(np.linalg.det(rotation), 1)):
        raise ValueError("Head-to-body transform must be a proper declared rotation")
    if set(origins_body_m) != {"left", "right"}:
        raise ValueError("Both virtual eye origins must be declared")
    eyes = []
    with load_array(path, allow_pickle=False) as arrays:
        for side in ("left", "right"):
            axes = arrays[side + "_directions_head"]
            provenance = json.dumps({"geometry_sha256": metadata["geometry_sha256"],
                "upstream_commit": metadata["upstream_commit"],
                "source_kind": metadata["source_kind"],
                "head_to_body_rotation": rotation.tolist(),
                "acceptance_sigma_degrees_engineering_assumption": sigma_degrees,
                "virtual_origin_body_m": np.asarray(origins_body_m[side]).tolist(),
                "MaleCNS_retinal_neurons_registered": 0}, sort_keys=True)
            eyes.append(EyeGeometry(side, axes @ rotation.T, np.full(len(axes), sigma_degrees),
                                    origins_body_m[side], provenance))
    return tuple(eyes)
