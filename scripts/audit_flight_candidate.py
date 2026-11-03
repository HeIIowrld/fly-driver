"""Independently compare an exported flight candidate with measured source edges and labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy.sparse import csr_matrix

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.candidate)
    source = json.loads(Path("configs/malecns-recovered-source.json").read_text(encoding="utf8"))
    nt_dir = Path("assets/external/malecns-neurotransmitters-20261029")
    nt_registry = json.loads((nt_dir / "source.json").read_text(encoding="utf8"))
    nt_path = nt_dir / "body-neurotransmitters.feather"
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=None,
            candidate_result_sha256=sha256(directory / "result.json"),
            graph_source_sha256=source["graph_sha256"],
            nt_registry_sha256=sha256(nt_dir / "source.json"),
            actuator_messages_sent=0,
        ),
    )
    result = dict(
        status="failed",
        actuator_messages_sent=0,
        functional_validation=False,
        vehicle_mapping_validated=False,
        goal_achieved=False,
    )
    try:
        candidate = json.loads((directory / "result.json").read_text(encoding="utf8"))
        if sha256(directory / "graph.npz") != candidate["graph_sha256"]:
            raise ValueError("Candidate graph hash mismatch")
        if sha256(source["graph_path"]) != source["graph_sha256"]:
            raise ValueError("Measured graph hash mismatch")
        if sha256(nt_path) != nt_registry["sha256"]:
            raise ValueError("Transmitter table hash mismatch")
        with load_array(directory / "graph.npz", allow_pickle=False) as archive:
            graph = {key: archive[key] for key in archive.files}
        with load_array(source["graph_path"], allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ("crow", "col", "counts", "body_ids")}
        bodies = graph["ids"].astype(np.int64)
        lookup = {int(body): index for index, body in enumerate(raw["body_ids"])}
        global_indices = np.array([lookup[int(body)] for body in bodies])
        incoming = csr_matrix(
            (raw["counts"], raw["col"], raw["crow"]), shape=(len(lookup), len(lookup))
        )
        expected = incoming[global_indices][:, global_indices].tocsr()
        actual = csr_matrix(
            (graph["weights"], (graph["post"], graph["pre"])), shape=(len(bodies), len(bodies))
        )
        delta = expected - actual
        if delta.nnz or expected.nnz != len(graph["pre"]):
            raise ValueError("Missing, duplicated or altered induced measured edges")
        annotations = json.loads(
            (directory / "selected-annotations.json").read_text(encoding="utf8")
        )
        annotation_by_id = {int(row["bodyId"]): row for row in annotations}
        if set(annotation_by_id) != set(bodies):
            raise ValueError("Selected annotation IDs differ")
        boundary = json.loads((directory / "boundary.json").read_text(encoding="utf8"))["boundary"]
        incoming_total = np.asarray(incoming.sum(axis=1)).ravel()
        outgoing_total = np.asarray(incoming.sum(axis=0)).ravel()
        retained_in = np.asarray(expected.sum(axis=1)).ravel()
        retained_out = np.asarray(expected.sum(axis=0)).ravel()
        for index, row in enumerate(boundary):
            if int(row["body_id"]) != bodies[index]:
                raise ValueError("Boundary rows not aligned to graph IDs")
            g = global_indices[index]
            measured = [
                incoming_total[g],
                outgoing_total[g],
                retained_in[index],
                retained_out[index],
            ]
            declared = [
                row["incoming_synapses_total"],
                row["outgoing_synapses_total"],
                row["incoming_synapses_retained"],
                row["outgoing_synapses_retained"],
            ]
            if not np.array_equal(measured, declared):
                raise ValueError("Cut-boundary synapse totals differ from source")
        table = feather.read_table(nt_path)
        selected_nt = table.filter(pc.is_in(table["body"], value_set=pa.array(bodies)))
        nt_rows = selected_nt.to_pylist()
        if len({row["body"] for row in nt_rows}) != len(nt_rows):
            raise ValueError("Duplicate transmitter body IDs")
        missing_nt = sorted(set(map(int, bodies)) - {row["body"] for row in nt_rows})
        write_json(run / "neurotransmitter-annotations.json", nt_rows)
        low_confidence = [
            str(row["body"])
            for row in nt_rows
            if row["predicted_nt_confidence"] is None or row["predicted_nt_confidence"] < 0.5
        ]
        roles = {}
        for role in ("input", "internal", "output"):
            selected = np.flatnonzero(graph["roles"] == role)
            fractions = [
                boundary[i]["incoming_fraction_retained"]
                for i in selected
                if boundary[i]["incoming_fraction_retained"] is not None
            ]
            roles[role] = dict(
                nodes=len(selected),
                incoming_retention_min=float(min(fractions)),
                incoming_retention_median=float(np.median(fractions)),
                incoming_retention_max=float(max(fractions)),
            )
        paths = json.loads((directory / "paths.json").read_text(encoding="utf8"))
        missing_sources = sorted(
            {pair["source_body_id"] for path in paths for pair in path["missing_pairs"]}
        )
        result.update(
            status="completed_structural_audit",
            termination_reason=("source edge and boundary equality verified; "
                                "transmitter labels joined"),
            nodes=len(bodies),
            edges=actual.nnz,
            exact_induced_edges=True,
            exact_boundary_totals=True,
            roles=roles,
            missing_path_source_annotations=[
                annotation_by_id[int(body)] for body in missing_sources
            ],
            neurotransmitter_annotation_count=len(nt_rows),
            missing_neurotransmitter_ids=missing_nt,
            consensus_nt_counts=dict(Counter(str(row["consensus_nt"]) for row in nt_rows)),
            predicted_nt_confidence_below_half_ids=low_confidence,
            nt_sha256=nt_registry["sha256"],
            signs_assigned=False,
            statement="Transmitter predictions retained as annotations, "
            "not converted automatically to excitatory/inhibitory functional signs",
        )
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run / "result.json", result)
    if result["status"] == "failed":
        raise RuntimeError(result["reason"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
