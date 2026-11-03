import numpy as np
import pytest

from flybrain_drive.connectome_selection import extract_id_subgraph, extract_type_subgraph


def data():
    # Incoming CSR: 10->20 weight2; 20->30 weight3; 40->20 weight7 (outside boundary).
    return {"body_ids": np.array([10, 20, 30, 40]), "crow": np.array([0, 0, 2, 3, 3]),
            "col": np.array([0, 3, 1]), "counts": np.array([2, 7, 3], np.float32)}


def annotations():
    return [{"bodyId": i, "type": t, "status": "Traced", "somaSide": "L"}
            for i, t in [(10, "input"), (20, "relay"), (30, "output"), (40, "other")]]


def test_incoming_matrix_orientation_and_boundary_loss_are_not_reversed():
    graph, audit = extract_type_subgraph(data(), annotations(),
        {"input": "input", "relay": "internal", "output": "output"})
    np.testing.assert_array_equal(graph["pre"], [0, 1])
    np.testing.assert_array_equal(graph["post"], [1, 2])
    np.testing.assert_array_equal(graph["weights"], [2, 3])
    assert audit["output_minimum_hops"] == {"30": 2}
    assert audit["boundary"][1]["incoming_fraction_retained"] == pytest.approx(2/9)
    assert audit["type_pair_connections"]["input -> relay"]["synapses"] == 2


def test_empty_type_or_unreachable_output_cannot_be_silently_called_a_circuit():
    with pytest.raises(ValueError, match="no selected"):
        extract_type_subgraph(data(), annotations(), {"missing": "input", "output": "output"})
    changed = data()
    changed["col"] = np.array([0, 3, 3])
    with pytest.raises(ValueError, match="no input path"):
        extract_type_subgraph(changed, annotations(),
                              {"input": "input", "relay": "internal", "output": "output"})


def test_source_indices_must_be_a_valid_directed_csr():
    changed = data()
    changed["crow"] = np.array([0, 0, 3, 2, 3])
    with pytest.raises(ValueError, match="CSR"):
        extract_type_subgraph(changed, annotations(), {"input": "input", "output": "output"})


def test_exact_id_selection_does_not_expand_shared_type_or_lose_real_type_labels():
    rows = annotations()
    rows[0]["type"] = rows[3]["type"] = "shared_sensory_type"
    graph, audit = extract_id_subgraph(data(), rows,
                                      {10: "input", 20: "internal", 30: "output"})
    np.testing.assert_array_equal(graph["ids"], ["10", "20", "30"])
    assert graph["types"][0] == "shared_sensory_type"
    assert audit["boundary"][1]["incoming_fraction_retained"] == pytest.approx(2/9)
    assert audit["id_roles"]["10"] == "input"
    with pytest.raises(ValueError, match="Duplicate normalized"):
        extract_id_subgraph(data(), rows, {10: "input", "10": "internal", 30: "output"})


def test_missing_type_annotation_is_explicit_and_does_not_merge_unknown_cell_gains():
    rows = annotations()
    rows[1]["type"] = None
    graph, audit = extract_id_subgraph(data(), rows, {10: "input", 20: "internal", 30: "output"})
    assert graph["types"][1] == "untyped:20"
    assert graph["types"].dtype.kind == "U"
    assert audit["untyped_body_ids"] == ["20"]
