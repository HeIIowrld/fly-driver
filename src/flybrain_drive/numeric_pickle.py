"""Bounded import of legacy numeric AC records, never unrestricted pickle globals.

Only primitive containers and validated NumPy numeric reconstruction are supported.
The special constructors below build data; file-selected Python functions/classes
are never returned by find_class. This is intentionally not a general pickle loader.
"""

import io
import math
import pickle
import pickletools

import numpy as np

_ARRAY_TYPE = object()
_CODES = {"b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"}


class _NumericDtype:
    def __init__(self, code, *flags):
        if code not in _CODES or flags not in ((), (False, True)):
            raise ValueError("Only primitive numeric dtypes are accepted")
        self.code, self.order = code, "="

    def __setstate__(self, state):
        if (not isinstance(state, tuple) or len(state) != 8 or state[0] != 3
                or state[1] not in ("<", ">", "=", "|")
                or state[2:6] != (None, None, None, -1) or state[6:] != (-1, 0)):
            raise ValueError("Unsupported numeric dtype metadata")
        self.order = state[1]

    def numpy(self):
        return np.dtype(self.code).newbyteorder(self.order)


class _NumericArray:
    def __init__(self, budget):
        self.budget, self.value = budget, None

    def __setstate__(self, state):
        if (not isinstance(state, tuple) or len(state) != 5 or state[0] != 1
                or type(state[3]) is not bool):
            raise ValueError("Unsupported numeric array state")
        _, shape, dtype, fortran, raw = state
        self.assign(raw, dtype, shape, "F" if fortran else "C")

    def assign(self, raw, dtype, shape, order):
        if (self.value is not None or not isinstance(raw, (bytes, bytearray))
                or not isinstance(dtype, _NumericDtype) or not isinstance(shape, tuple)
                or len(shape) > 5 or any(type(n) is not int or n < 0 for n in shape)
                or order not in ("C", "F")):
            raise ValueError("Invalid bounded numeric array")
        size = math.prod(shape)
        dt = dtype.numpy()
        if size > 1_000_000 or size * dt.itemsize != len(raw):
            raise ValueError("Numeric array shape/payload exceeds bounds")
        self.budget[0] -= len(raw)
        if self.budget[0] < 0:
            raise ValueError("Total numeric allocation budget exhausted")
        self.value = np.frombuffer(raw, dtype=dt).copy().reshape(shape, order=order)
        self.value.setflags(write=False)
        return self


def load_numeric_pickle(data, *, max_bytes=16 * 1024**2, max_operations=2_000_000):
    if (not isinstance(data, bytes) or not 0 < len(data) <= max_bytes
            or not 0 < max_bytes <= 64 * 1024**2):
        raise ValueError("Explicit bounded numeric record bytes required")
    prohibited = {"PERSID", "BINPERSID", "EXT1", "EXT2", "EXT4", "INST", "OBJ",
                  "NEWOBJ", "NEWOBJ_EX", "NEXT_BUFFER", "READONLY_BUFFER"}
    count, _position, last = 0, -1, None
    for opcode, _, _position in pickletools.genops(data):
        count += 1
        if count > max_operations or opcode.name in prohibited:
            raise ValueError("Unsupported or oversized numeric pickle program")
        last = opcode.name
    if last != "STOP" or _position + 1 != len(data):
        raise ValueError("Incomplete or trailing numeric record data")
    budget = [64 * 1024**2]

    def reconstruct(kind, shape, code):
        if kind is not _ARRAY_TYPE or shape != (0,) or code != b"b":
            raise ValueError("Unsupported array reconstruction")
        return _NumericArray(budget)

    def frombuffer(raw, dtype, shape, order):
        return _NumericArray(budget).assign(raw, dtype, shape, order)

    def scalar(dtype, raw):
        if (not isinstance(dtype, _NumericDtype) or not isinstance(raw, bytes)
                or len(raw) != dtype.numpy().itemsize):
            raise ValueError("Invalid numeric scalar")
        return np.frombuffer(raw, dtype=dtype.numpy())[0].item()

    class NumericUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy" and name == "dtype":
                return _NumericDtype
            if module == "numpy" and name == "ndarray":
                return _ARRAY_TYPE
            if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
                if name == "_reconstruct":
                    return reconstruct
                if name == "scalar":
                    return scalar
            if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
                return frombuffer
            raise ValueError("Unsupported pickle global: " + module + "." + name)

        def persistent_load(self, pid):
            raise ValueError("External pickle references are forbidden")

    decoded = NumericUnpickler(io.BytesIO(data)).load()
    active, cache = set(), {}
    visited = 0

    def unwrap(value, depth=0):
        nonlocal visited
        visited += 1
        if visited > max_operations or depth > 100:
            raise ValueError("Decoded numeric data exceeds traversal budget")
        if type(value) in (str, bytes, int, float, bool, type(None)):
            if type(value) is int and value.bit_length() > 64:
                raise ValueError("Oversized scalar integer")
            return value
        identity = id(value)
        if identity in active:
            raise ValueError("Cyclic numeric records are unsupported")
        if identity in cache:
            return cache[identity]
        active.add(identity)
        if isinstance(value, _NumericArray) and value.value is not None:
            result = value.value
        elif type(value) in (list, tuple):
            result = type(value)(unwrap(v, depth+1) for v in value)
        elif type(value) is dict:
            if any(type(k) not in (str, int) for k in value):
                raise ValueError("Numeric records require simple dictionary keys")
            result = {k: unwrap(v, depth+1) for k, v in value.items()}
        else:
            raise ValueError("Unsupported decoded numeric object")
        active.remove(identity)
        cache[identity] = result
        return result

    return unwrap(decoded)
