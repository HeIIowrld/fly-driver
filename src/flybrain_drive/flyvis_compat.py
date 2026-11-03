"""Explicit cache-I/O compatibility for pinned datamate, never model dynamics changes."""

import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np

DATAMATE_WRITER_SHA256 = '5791c681fb63b6ed805a47f4e4f64ee4aec22a2b78ce37c9939ca95711c3d93a'


def write_closed_h5(path, value, *, cache_root):
    """Preserve array bytes/dtype and SWMR; close the owned cache file on every path."""
    import h5py

    path, root = Path(path).resolve(), Path(cache_root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('HDF5 compatibility writer is confined to its declared cache root')
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, libver='latest', mode='w') as stream:
        stream.create_dataset('data', data=np.asarray(value))
        stream.swmr_mode = True
        if not stream.swmr_mode:
            raise RuntimeError('SWMR cache publication failed')


def install_datamate_cache_compat(cache_root):
    """Patch only the exact reviewed writer in this process and only for owned cache paths."""
    io = importlib.import_module('datamate.io')
    directory = importlib.import_module('datamate.directory')
    digest = hashlib.sha256(inspect.getsource(io._write_h5).encode()).hexdigest()
    if digest != DATAMATE_WRITER_SHA256 or directory._write_h5 is not io._write_h5:
        raise ValueError('Unreviewed datamate cache writer; refuse compatibility replacement')
    root = Path(cache_root).resolve()

    def writer(path, value):
        return write_closed_h5(path, value, cache_root=root)

    io._write_h5 = directory._write_h5 = writer
    return dict(original_writer_sha256=digest, cache_root=str(root),
                change='closed context-managed HDF5 array write; no unlink of open file',
                model_equations_modified=False, upstream_files_modified=False)
