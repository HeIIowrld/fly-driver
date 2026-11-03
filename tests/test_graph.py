import numpy as np
import pytest

from flybrain_drive.graph import demo_graph, load_graph, rewire, save_graph, validate_graph


def test_rewire_preserves_each_nodes_degrees_weights_and_ports(tmp_path):
    graph = demo_graph()
    changed, metadata = rewire(graph, seed=12, swaps=500)
    assert metadata["achieved_swaps"] == 500
    for key in ("pre", "post"):
        np.testing.assert_array_equal(
            np.bincount(graph[key], minlength=56), np.bincount(changed[key], minlength=56)
        )
    for key in ("ids", "types", "roles", "weights"):
        np.testing.assert_array_equal(changed[key], graph[key])
    assert not np.array_equal(changed["post"], graph["post"])
    validate_graph(changed)
    path = tmp_path / "graph.npz"
    save_graph(path, changed, metadata)
    np.testing.assert_array_equal(load_graph(path)["pre"], changed["pre"])
    with pytest.raises(FileExistsError):
        save_graph(path, changed, metadata)


def test_duplicate_pairs_rejected():
    graph = demo_graph()
    graph["pre"][1], graph["post"][1] = graph["pre"][0], graph["post"][0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_graph(graph)
