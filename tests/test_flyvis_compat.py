import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

from flybrain_drive.flyvis_compat import write_closed_h5


@pytest.mark.parametrize('value', [np.array([1., -2.5], dtype=np.float32),
                                 np.array([0, 8], dtype=np.int64),
                                 np.array([b'T4a', b'T5a'], dtype='S3')])
def test_cache_values_dtype_and_file_release_are_preserved(tmp_path, value):
    root = tmp_path/'owned'
    path = root/'array.h5'
    write_closed_h5(path, value, cache_root=root)
    with h5py.File(path, 'r', swmr=True) as stream:
        assert stream['data'].dtype == value.dtype
        assert np.array_equal(stream['data'][:], value)
    # Windows rejects this rename if the writer still owns an open file handle.
    path.replace(root/'closed.h5')


def test_cache_writer_refuses_paths_outside_the_owned_root(tmp_path):
    with pytest.raises(ValueError, match='confined'):
        write_closed_h5(tmp_path/'other.h5', np.array([1]), cache_root=tmp_path/'owned')
    assert not (tmp_path/'other.h5').exists()
