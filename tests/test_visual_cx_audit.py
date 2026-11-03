import json
import sys

import numpy as np
import pytest

feather = pytest.importorskip("pyarrow.feather")
arrow = pytest.importorskip("pyarrow")

from scripts.audit_visual_cx_candidate import main


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    rows = [{"bodyId": i, "type": kind, "status": "Traced", "somaSide": "L"}
            for i, kind in enumerate(("MeTu1", "ER2_a", "PFL3", "DNa02", "organ"), 1)]
    feather.write_feather(arrow.Table.from_pylist(rows), tmp_path / "annotations.feather")
    np.savez(tmp_path / "raw.npz", body_ids=np.arange(1, 6),
             crow=np.array([0, 0, 1, 2, 4, 4]), col=np.array([0, 1, 2, 4]),
             counts=np.array([10, 20, 30, 40]))
    np.savez(tmp_path / "base.npz", ids=np.array(["3", "4", "5"]),
             types=np.array(["PFL3", "DNa02", "organ"]),
             roles=np.array(["input", "output", "input"]),
             pre=np.array([0, 2]), post=np.array([1, 1]), weights=np.array([30., 40.]))
    groups = {"groups": {"MeTu": {"types": ["MeTu1"], "role": "input"},
                         "ER": {"types": ["ER2_a"], "role": "internal"},
                         "PFL": {"types": ["PFL3"], "role": "internal"}},
              "replace_boundary_types": ["PFL3"],
              "ordered_path_stages": [{"name": kind, "types": [kind]}
                                      for kind in ("MeTu1", "ER2_a", "PFL3", "DNa02")]}
    (tmp_path / "groups.json").write_text(json.dumps(groups), encoding="utf8")
    monkeypatch.setattr(sys, "argv", ["audit", "--graph", str(tmp_path / "raw.npz"),
        "--annotations", str(tmp_path / "annotations.feather"),
        "--base", str(tmp_path / "base.npz"), "--groups", str(tmp_path / "groups.json"),
        "--output", str(tmp_path / "output")])
    return tmp_path, groups


def test_visual_extension_preserves_real_paths_and_removes_old_boundary_injection(candidate):
    path, _ = candidate
    main()
    result = json.loads((path / "output/summary.json").read_text())
    assert result["nodes"] == 5 and result["edge_records"] == 4
    assert result["groups"]["MeTu"]["path_lengths_to_output"] == [3]
    assert result["groups"]["MeTu"]["reaches_output"] == 1
    assert result["group_pairs"]["ER -> PFL"]["synapses"] == 20
    assert all(s["reaches_output_via_every_later_stage"] == 1
               for s in result["ordered_direct_chain"])
    with np.load(path / "output/visual-cx-body.npz", allow_pickle=False) as graph:
        assert graph["roles"][np.flatnonzero(graph["ids"] == "3")[0]] == "internal"
        assert set(graph["ids"][graph["roles"] == "input"]) == {"1", "5"}


def test_a_short_bypass_is_not_counted_as_a_complete_visual_cx_chain(candidate):
    path, _ = candidate
    np.savez(path / "raw.npz", body_ids=np.arange(1, 6),
             crow=np.array([0, 0, 1, 1, 4, 4]), col=np.array([0, 0, 2, 4]),
             counts=np.array([10, 20, 30, 40]))
    main()
    result = json.loads((path / "output/summary.json").read_text())
    assert result["groups"]["MeTu"]["path_lengths_to_output"] == [1]
    assert result["ordered_direct_chain"][0]["reaches_output_via_every_later_stage"] == 0


@pytest.mark.parametrize("fault", ["overlap", "missing", "old_boundary"])
def test_invalid_type_inventory_cannot_silently_select_another_circuit(candidate, fault):
    path, groups = candidate
    if fault == "overlap":
        groups["groups"]["ER"]["types"].append("MeTu1")
    elif fault == "missing":
        groups["groups"]["ER"]["types"].append("not_present")
    else:
        groups["replace_boundary_types"].append("not_present")
    (path / "groups.json").write_text(json.dumps(groups), encoding="utf8")
    with pytest.raises(ValueError, match="Overlapping|no Traced|must be replaced"):
        main()
    assert not (path / "output").exists()
