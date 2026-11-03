import pickle

import numpy as np
import pytest

from flybrain_drive.numeric_pickle import load_numeric_pickle


@pytest.mark.parametrize("protocol", [4, 5])
def test_numeric_records_arrays_scalars_and_aliases_roundtrip(protocol):
    vector = np.arange(6, dtype="f4").reshape(2, 3)
    original = {"states": [{"sensors": vector, "speed": np.float64(2.5),
                            "load": [0, 1., None], "same_array": vector}],
                "metadata": ("monza", True)}
    result = load_numeric_pickle(pickle.dumps(original, protocol=protocol))
    row = result["states"][0]
    np.testing.assert_array_equal(row["sensors"], vector)
    assert row["sensors"] is row["same_array"] and not row["sensors"].flags.writeable
    assert row["speed"] == 2.5 and result["metadata"] == ("monza", True)


def test_arbitrary_globals_and_object_arrays_are_rejected():
    class Expression:
        def __reduce__(self):
            return eval, ("1 + 1",)

    with pytest.raises(ValueError, match="global"):
        load_numeric_pickle(pickle.dumps(Expression()))
    with pytest.raises(ValueError, match="dtype"):
        load_numeric_pickle(pickle.dumps(np.array([{"object": True}], dtype=object)))


def test_cycles_trailing_bytes_and_limits_are_rejected():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="Cyclic"):
        load_numeric_pickle(pickle.dumps(value))
    with pytest.raises(ValueError, match="trailing"):
        load_numeric_pickle(pickle.dumps(1) + b"junk")
    with pytest.raises(ValueError, match="bounded"):
        load_numeric_pickle(pickle.dumps([1, 2]), max_bytes=2)
    with pytest.raises(ValueError, match="oversized"):
        load_numeric_pickle(pickle.dumps([1, 2]), max_operations=2)
