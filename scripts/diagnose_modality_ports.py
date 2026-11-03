"""Separated modality stimuli through measured graph ports; not autonomous driving."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "ports", "config", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    import torch

    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.evidence import begin_run, write_json
    from flybrain_drive.graph import load_graph, sha256
    from flybrain_drive.modality_ports import ModalityPortAdapter
    from flybrain_drive.policy import SparseCircuit
    from flybrain_drive.sensory_runtime import make_sensory_runtime

    torch.set_num_threads(1)
    torch.manual_seed(610101)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    graph = load_graph(args.graph)
    ports = json.loads(Path(args.ports).read_text(encoding="utf8"))
    env = FlyBrainEnv(**config["environment"])
    try:
        _, packer, sealed = make_sensory_runtime(env, config)
        position_size = sum(p.channels * (p.populations + 2) for p in packer.organs.positions)
        organ_size = packer.organ_size
        if organ_size - position_size != 6:
            raise ValueError("Diagnostic needs six declared signed rotation features")
    finally:
        env.close()
    groups = {}
    for name, bodies in ports["organ_input_groups"].items():
        if name.startswith("haltere_"):
            indices = list(range(position_size, organ_size))
        elif name.startswith("chordotonal_"):
            indices = list(range(position_size))
        else:
            raise ValueError("No declared modality hypothesis for this source group")
        groups[name] = {"body_ids": bodies, "feature_indices": indices}
    adapter = ModalityPortAdapter(graph, ports["visual_CX_cut_boundary_ids"], groups,
                                   organ_size=organ_size, visual_size=1)
    circuit = SparseCircuit(graph, config["environment"]["control_dt"])
    with torch.no_grad():
        adapter.visual_adapter.weight.fill_(.1)
        for layer in adapter.organ_adapters:
            layer.weight.fill_(.1 / layer.in_features)
    run = begin_run(args.output, {
        "graph_sha256": sha256(args.graph), "ports_sha256": sha256(args.ports),
        "script_sha256": sha256(__file__), "configuration": sealed,
        "organ_groups": groups, "seed": 610101,
        "scope": "unit stimuli and engineered adapter gains; not vision or driving evaluation",
        "functional_tuning_verified": False,
    })
    records, arrays = {}, {}
    for name in ("visual_only", "rotation_only", "position_only", "all_missing"):
        visual, visual_valid = torch.ones(1, 1), torch.zeros(1, 1)
        organs, organ_valid = torch.ones(1, organ_size), torch.zeros(1, organ_size)
        if name == "visual_only":
            visual_valid[:] = 1
        elif name == "rotation_only":
            organ_valid[:, position_size:] = 1
        elif name == "position_only":
            organ_valid[:, :position_size] = 1
        drive = adapter(visual, visual_valid, organs, organ_valid)
        driven = {str(body) for i, body in enumerate(graph["ids"]) if drive[0, i] != 0}
        if name == "visual_only":
            expected = set(ports["visual_CX_cut_boundary_ids"])
        else:
            prefix = "haltere_" if name == "rotation_only" else "chordotonal_"
            expected = ({body for g, rows in ports["organ_input_groups"].items()
                         if g.startswith(prefix) for body in rows}
                        if name != "all_missing" else set())
        assert driven == expected
        state = torch.zeros(1, len(graph["ids"]))
        outputs = []
        with torch.no_grad():
            for _ in range(100):
                output, state = circuit.step_with_drive(drive, state)
                outputs.append(output.numpy().copy())
        response = np.concatenate(outputs)
        if name == "all_missing":
            assert not response.any()
        else:
            assert np.any(response != 0)
        records[name] = {"driven_neurons": len(driven), "final_outputs": response[-1].tolist(),
                         "first_nonzero_output_tick": (int(np.flatnonzero(
                             np.any(response != 0, axis=1))[0]) if response.any() else None)}
        arrays[name + "_drive"] = drive.detach().numpy()
        arrays[name + "_outputs"] = response
    np.savez_compressed(run / "responses.npz", **arrays)
    write_json(run / "summary.json", {"stimuli": records, "measured_graph_nodes": len(graph["ids"]),
                                       "claim": "modality isolation and structural response only"})
    write_json(run / "result.json", {"status": "completed", "goal_achieved": False,
                                     "responses_sha256": sha256(run / "responses.npz")})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
