"""Optional fixed-weight float32 CSR inference; not a new biological circuit."""
import numpy as np
import numba


@numba.njit(parallel=True, nogil=True, fastmath=False)
def _matmul(indptr, indices, data, states):
    batch, width = states.shape
    result = np.empty((batch, width), dtype=np.float32)
    for item in numba.prange(batch * width):
        b, row = item // width, item % width
        total = np.float32(0.)
        for edge in range(indptr[row], indptr[row+1]):
            total += data[edge] * states[b, indices[edge]]
        result[b, row] = total
    return result


def fixed_csr_matmul(matrix, states, *, threads=8):
    """Use a bounded thread count for this call and restore the caller's mask."""
    if (states.ndim != 2 or states.dtype != np.float32 or
            states.shape[1] != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]
            or matrix.dtype != np.float32 or type(threads) is not int
            or not 1 <= threads <= numba.config.NUMBA_NUM_THREADS):
        raise ValueError('Matching float32 fixed CSR and supported thread count required')
    original = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        return _matmul(matrix.indptr, matrix.indices, matrix.data, states)
    finally:
        numba.set_num_threads(original)
