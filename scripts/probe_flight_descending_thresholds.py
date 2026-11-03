"""Resolve local-gradient versus finite-current differences at model steering DN gates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_commands import SeparatedFlightCircuit
from flybrain_drive.graph import sha256
from scripts.audit_flight_feedback_paths import encode, read, settle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, data_root = Path(args.candidate), Path(args.data)
    receipt, config = read(root / "result.json"), read(root / "manifest.json")["config"]
    checkpoint, spec = receipt["checkpoint"], config["arguments"]["spec"]
    if (sha256(checkpoint) != receipt["checkpoint_sha256"]
            or sha256(spec) != config["spec_sha256"]
            or sha256(data_root / "dataset.json") != config["dataset_manifest_sha256"]):
        raise ValueError("Unchanged candidate/spec/data required")
    contexts = [(1, 100), (4, 400)]
    levels = [1e-5, 1e-4, .0003, .001, .003, .01, .03, .1]
    run = begin_run(args.output, dict(arguments=vars(args), seed=863141,
        checkpoint_sha256=sha256(checkpoint), contexts=contexts, positive_current_levels=levels,
        model_steps=200, dt=.01, rate_change_threshold=1e-5, intent_change_threshold=1e-5,
        scope="artificial added neuron current, rate-model gate only; not measured DN excitability",
        actuator_messages_sent=0, optimizer_updates=0))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SeparatedFlightCircuit(spec, glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    reports = []
    for number, index in contexts:
        entry = read(data_root / "dataset.json")["episodes"][number]
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original episode changed")
        cases = [(None, 0.)] + [(port, level) for port in range(4) for level in levels]
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            inputs = {key: torch.tensor(np.repeat(archive[key][index:index + 1],
                                                  len(cases), axis=0), device="cuda")
                      for key in ("visual", "visual_valid", "body", "body_valid", "goals",
                                  "scene_features")}
        with torch.no_grad():
            drive = encode(model, inputs)
            for row, (port, current) in enumerate(cases):
                if port is not None:
                    drive[row, model.steering_ports[port]] += current
            state = settle(model, drive)
            intent = model._read_state_tensor(state)[0].cpu().numpy()
            rate = state[:, model.steering_ports].cpu().numpy()
            net = (torch.sparse.mm(model.circuit.matrix, state.T).T + drive)
            net = net[:, model.steering_ports].cpu().numpy()
        rows = []
        for row, (port, current) in enumerate(cases[1:], start=1):
            rows.append(dict(id=model.spec["steering_input_ids"][port], current=current,
                delta_rate=float(rate[row, port] - rate[0, port]),
                delta_turn=float(intent[row, 0] - intent[0, 0]),
                net_current=float(net[row, port])))
        thresholds = {}
        for bid in model.spec["steering_input_ids"]:
            selected = [row for row in rows if row["id"] == bid]
            thresholds[bid] = dict(
                first_rate_level=next((r["current"] for r in selected
                                      if r["delta_rate"] > 1e-5), None),
                first_turn_level=next((r["current"] for r in selected
                                      if abs(r["delta_turn"]) > 1e-5), None))
        reports.append(dict(file=entry["file"], index=index,
                            baseline_dn_rates=rate[0].tolist(), baseline_dn_net=net[0].tolist(),
                            thresholds=thresholds, added_current_responses=rows))
    write_json(run / "result.json", dict(status="completed", contexts=reports,
        physical_neuron_thresholds_measured=False, recognition_verified=False,
        actor_qualified=False, actuator_messages_sent=0, goal_achieved=False,
        termination_reason="declared model current levels and activation gates measured"))
    print(json.dumps([{k: r[k] for k in ("file", "baseline_dn_net", "thresholds")}
                      for r in reports]))


if __name__ == "__main__":
    main()
