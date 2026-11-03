import numpy as np
import pytest

from flybrain_drive.connectome_routes import audit_direct_routes


def inputs():
    raw = {"body_ids": np.array([10, 11, 20, 21, 22]),
           "crow": np.array([0, 0, 0, 2, 3, 4]), "col": np.array([0, 1, 0, 2]),
           "counts": np.array([4., 6., 5., 7.])}
    rows = [{"bodyId": body, "status": "Traced", "rootSide": side, "superclass": cls}
            for body, side, cls in [(10, "L", "sensory"), (11, "R", "sensory"),
                                   (20, None, "ascending_neuron"),
                                   (21, None, "vnc_motor"), (22, None, "descending_neuron")]]
    selectors = {side: {"status": "Traced", "rootSide": side} for side in ["L", "R"]}
    return raw, rows, selectors


def test_incoming_csr_exports_actual_sensory_direction_and_full_target_denominators():
    raw, rows, selectors = inputs()
    report, edges = audit_direct_routes(raw, rows, selectors)
    assert report["L"]["synapses"] == 9
    assert report["R"]["synapses"] == 6
    assert report["L"]["target_superclass_synapses"] == {"ascending_neuron": 4, "vnc_motor": 5}
    target = next(r for r in report["L"]["targets"] if r["body_id"] == "20")
    assert target["fraction_of_target_input"] == .4
    assert target["total_incoming_in_recovered_graph"] == 10
    assert set(zip(edges["pre_body_ids"], edges["post_body_ids"], strict=True)) == {
        (10, 20), (11, 20), (10, 21)}
    assert 22 not in edges["post_body_ids"]  # two-hop DN is not falsely called a direct target


def test_overlapping_or_absent_source_selectors_are_rejected():
    raw, rows, selectors = inputs()
    with pytest.raises(ValueError, match="overlap"):
        audit_direct_routes(raw, rows, {**selectors, "any": {"status": "Traced"}})
    with pytest.raises(ValueError, match="no graph neurons"):
        audit_direct_routes(raw, rows, {"missing": {"status": "Traced", "rootSide": "X"}})
    with pytest.raises(ValueError, match="Duplicate annotation"):
        audit_direct_routes(raw, rows + [rows[0]], selectors)
    with pytest.raises(ValueError, match="Invalid incoming"):
        audit_direct_routes({**raw, "col": np.array([0, 1, 0, 99])}, rows, selectors)
    with pytest.raises(ValueError, match="absent from recovered graph"):
        audit_direct_routes(raw, rows + [{**rows[0], "bodyId": 999}], selectors)
