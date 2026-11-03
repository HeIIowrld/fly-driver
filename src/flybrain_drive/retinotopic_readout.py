"""Exact modeled cell-type/column lookup; no guessed symmetry or missing-cell fills."""

import numpy as np


def column_indices(types, u, v, retinal_u, retinal_v, names):
    types, u, v = np.asarray(types), np.asarray(u), np.asarray(v)
    columns = list(zip(retinal_u, retinal_v, strict=True))
    if (len(names) != len(set(names)) or not names or types.shape != u.shape
            or u.shape != v.shape or len(columns) != len(set(columns))):
        raise ValueError('Unique explicit types and one-to-one original retinal columns required')
    result = []
    for name in names:
        selected = np.flatnonzero(types == name)
        mapping = {(int(u[i]), int(v[i])): int(i) for i in selected}
        if len(mapping) != len(selected) or any(c not in mapping for c in columns):
            raise ValueError('Selected modeled cell type has duplicate or missing retinal columns')
        result.append([mapping[c] for c in columns])
    return np.asarray(result, dtype=np.int64)
