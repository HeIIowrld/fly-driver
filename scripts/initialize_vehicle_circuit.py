"""Initialize the measured-structure multimodal controller, without driving or optimization."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.body_organs import BodyOrganEncoder, PositionOrganSpec, ScalarOrganSpec
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import load_graph, sha256
from flybrain_drive.local_retinal_circuit import LocalRetinalController
from flybrain_drive.neurotransmitters import join_transmitters
from flybrain_drive.sensor_contract import ActorObservation, BodySample, SampleTime
from flybrain_drive.sensory_policy import SensoryFeatureEncoder
from flybrain_drive.vehicle_circuit import VehicleCircuitController
from flybrain_drive.vehicle_ports import extend_organ_ports


def organs(settings, scalars=()):
    defaults = [asdict(PositionOrganSpec("steering_angle", 2, -.5, .5, .15, 1)),
                asdict(PositionOrganSpec("suspension_position", 4, -.07, .07, .02, 1))]
    positions = [PositionOrganSpec(**s) for s in settings.get("position_organs", defaults)]
    return BodyOrganEncoder(positions,
                            rotation_scale=settings.get("rotation_scale", 5),
                            max_sample_age=settings.get("max_sample_age", .15),
                            scalars=[ScalarOrganSpec(**s) for s in scalars])


def scalar_fixture(settings, scalars, temperature=353.15, loads=(2000,) * 4, missing=False):
    """Declared synthetic scalar stimulus; never a native or calibrated AC observation."""
    enc = organs(settings, scalars)
    samples = [BodySample("steering_angle", [0, 0], [1, 1], SampleTime(0, 0), "rad", "roadwheel",
                          "declared_sensor_model"),
               BodySample("suspension_position", [0] * 4, [1] * 4, SampleTime(0, 0), "m",
                          "compression_positive", "declared_sensor_model"),
               BodySample("angular_velocity", [0] * 3, [1] * 3, SampleTime(0, 0), "rad/s", "body",
                          "declared_sensor_model")]
    if not missing:
        samples.extend([
            BodySample("tyre_core_temperature", [temperature] * 4, [1] * 4, SampleTime(0, 0),
                       "K", "tyre_core", "declared_sensor_model"),
            BodySample("wheel_normal_load", loads, [1] * 4, SampleTime(0, 0), "N",
                       "contact_normal_compression_positive", "declared_sensor_model")])
    return enc.encode(ActorObservation("synthetic-initialization-stimulus", 0, 0, 0,
                                       (), tuple(samples)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vehicle-circuit-model.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    run = begin_run(args.output, {"configuration": config, "optimizer_updates": 0,
                                 "scope": "initialization and synthetic signal diagnostics only"})
    try:
        import pyarrow.feather as feather

        root = Path(config["candidate_root"])
        registry = json.loads((root / "source.json").read_text(encoding="utf8"))
        for name, expected in registry["files_sha256"].items():
            if sha256(root / name) != expected:
                raise ValueError("Registered candidate hash differs: " + name)
        if (sha256(config["source_checkpoint"]) != config["source_checkpoint_sha256"]
                or sha256(config["source_eye_calibration"])
                != config["source_eye_calibration_sha256"]):
            raise ValueError("Source checkpoint or eye calibration hash differs")
        nt_registry = json.loads(Path(config["neurotransmitter_registry"]).read_text(
            encoding="utf8"))
        if sha256(nt_registry["path"]) != nt_registry["sha256"]:
            raise ValueError("Transmitter source hash differs")
        graph = load_graph(root / "vehicle-candidate.npz")
        annotations = json.loads((root / "annotations.json").read_text(encoding="utf8"))
        ports = json.loads((root / "ports.json").read_text(encoding="utf8"))
        signs, joined, nt_audit = join_transmitters(graph["ids"],
            feather.read_table(nt_registry["path"]).to_pylist(), config["sign_hypothesis"])
        nt_audit["source_sha256"] = nt_registry["sha256"]
        write_json(run / "neurotransmitters.json", joined)
        write_json(run / "transmitter-audit.json", nt_audit)
        source = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=True)
        old_graph = {k: np.asarray(v, dtype={"pre": np.int64, "post": np.int64,
                                            "weights": np.float32}.get(k))
                     for k, v in source["graph"].items()}
        old = LocalRetinalController(old_graph, **source["constructor"])
        old.load_state_dict(source["model"])
        source_config = json.loads(Path(config["source_config"]).read_text(encoding="utf8"))
        settings = source_config["sensory_preprocessing"]
        scalars = json.loads(Path(config["scalar_calibration"]).read_text(
            encoding="utf8"))["scalar_organs"]
        old_organs, expanded = organs(settings), organs(settings, scalars)
        if old_organs.feature_count != source["constructor"]["organ_size"]:
            raise ValueError("Old feature calibration does not match source adapter")
        groups, feature_map = extend_organ_ports(graph, annotations, source["constructor"],
            old_organs.feature_names, expanded.feature_names, ports)
        eye_calibration = json.loads(Path(config["source_eye_calibration"]).read_text(
            encoding="utf8"))
        old_outputs = old_graph["ids"][old_graph["roles"] == "output"].tolist()
        if old_outputs != ports["steering"]["body_ids"]:
            raise ValueError("Steering output orientation/order differs from source declaration")
        constructor = {"visual_ids": source["constructor"]["visual_ids"], "organ_groups": groups,
            "directions": source["constructor"]["directions"],
            "available": source["constructor"]["available"], "organ_size": expanded.feature_count,
            "signs": signs.tolist(),
            "incoming_totals": load_array(root / "full-incoming-counts.npy",
                                       allow_pickle=False).tolist(),
            "output_groups": {"steering": ports["steering"]["body_ids"],
                              "forward": ports["forward_candidate"]["body_ids"],
                              "halt": ports["halt_partial_candidate"]["body_ids"]},
            "steering_signs": source["constructor"]["output_signs"],
            "organ_calibration_sha256": expanded.calibration_sha256,
            "eye_map_hashes": eye_calibration["eye_map_hashes"],
            "neighbors": source["constructor"]["neighbors"], "seed": source["constructor"]["seed"],
            "dt": config["dt_seconds"], "tau": config["tau_seconds"]}
        model = VehicleCircuitController(graph, **constructor)
        with torch.no_grad():
            model.adapter.visual_adapter.load_state_dict(old.adapter.visual_adapter.state_dict())
            model.feature_mean.copy_(old.feature_mean)
            model.feature_scale.copy_(old.feature_scale)
            for i, name in enumerate(old.adapter.group_names):
                j = model.adapter.group_names.index(name)
                model.adapter.organ_adapters[j].load_state_dict(old.adapter.organ_adapters[i].state_dict())
        batch, rays = 1, model.directions.shape[1]
        packed = {key: torch.zeros(batch, width) for key, width in (
            ("eye_age", 2), ("auxiliary", 7), ("auxiliary_valid", 7),
            ("applied_control", 3), ("applied_valid", 1))}
        packed.update(ray_values=torch.rand(batch, 2, rays, 3),
                      ray_directions=model.directions[None].clone(),
                      ray_valid=torch.tensor(constructor["available"], dtype=torch.float32)[None],
                      organs=torch.rand(batch, expanded.feature_count),
                      organ_valid=torch.ones(batch, expanded.feature_count))
        if set(packed) != SensoryFeatureEncoder.fields:
            raise ValueError("Synthetic stimulus actor schema differs")
        old_packed = {**packed, "organs": packed["organs"][:, feature_map],
                      "organ_valid": packed["organ_valid"][:, feature_map]}
        for key in ("organs", "organ_valid"):
            packed[key][:] = 0
            packed[key][:, feature_map] = old_packed[key]
        new_index = {str(body): i for i, body in enumerate(graph["ids"])}
        retained_indices = [new_index[str(body)] for body in old_graph["ids"]]
        with torch.no_grad():
            drive_error = float((old.prepare_drive(old_packed)
                                 - model.prepare_drive(packed)[:, retained_indices]).abs().max())
        if drive_error != 0:
            raise ValueError("Transferred eye/body input response differs from source")
        stimuli = {"scalar_missing": {"missing": True}, "temperature_low": {"temperature": 333.15},
                   "temperature_high": {"temperature": 373.15},
                   "load_left_shift": {"loads": (3500, 500, 3500, 500)}}
        responses = {}
        for name, options in stimuli.items():
            encoded = scalar_fixture(settings, scalars, **options)
            packed["organs"] = torch.tensor(encoded.values.copy(), dtype=torch.float32)[None]
            packed["organ_valid"] = torch.tensor(encoded.valid.copy(), dtype=torch.float32)[None]
            state = model.initial_state()
            with torch.no_grad():
                drive = model.prepare_drive(packed)
                for _ in range(64):
                    action, state = model.advance(drive, state)
            responses[name] = {"action_request": action[0].tolist(),
                               "active_input_ports": int((drive != 0).sum()),
                               "active_neurons": int((state != 0).sum())}
        state = model.initial_state()
        drive = model.prepare_drive(packed)
        for _ in range(16):
            action, state = model.advance(drive, state)
        action[:, 1:].sum().backward()
        gradients = {name: float(layer.weight.grad.abs().sum())
                     for name, layer in zip(model.adapter.group_names,
                                             model.adapter.organ_adapters, strict=True)}
        transfer = {"retained_input_response_max_error": drive_error,
                    "retained_visual_ports": len(constructor["visual_ids"]),
                    "old_organ_groups_copied": len(old.adapter.group_names),
                    "old_feature_to_new_feature": feature_map,
                    "old_model_outputs_or_recurrent_dynamics_preserved": False}
        payload = {"model_kind": "vehicle_signed_rate_v1", "constructor": constructor,
            "graph": {k: v.tolist() for k, v in graph.items()}, "model": model.state_dict(),
            "configuration": config, "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "driving_trained": False, "optimizer_updates": 0}
        torch.save(payload, run / "initialized-checkpoint.pt")
        restored = VehicleCircuitController(graph, **constructor)
        restored.load_state_dict(torch.load(run / "initialized-checkpoint.pt",
                                            weights_only=True)["model"])
        a, b = model.initial_state(), restored.initial_state()
        with torch.no_grad():
            for _ in range(12):
                x, a = model(packed, a)
                y, b = restored(packed, b)
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        write_json(run / "port-mapping.json", groups)
        write_json(run / "transfer.json", transfer)
        write_json(run / "stimulus-responses.json", responses)
        write_json(run / "gradient-diagnostics.json", gradients)
        result = {"status": "completed", "nodes": len(graph["ids"]),
                  "structural_edges": len(graph["pre"]), "organ_features": expanded.feature_count,
                  "input_response_transfer_error": drive_error, "checkpoint_restore_max_error": 0,
                  "visual_gradient_l1": float(model.adapter.visual_adapter.weight.grad.abs().sum()),
                  "thermal_groups_with_gradient": sum(v > 0 for k, v in gradients.items()
                                                       if k.startswith("thermal_")),
                  "load_groups_with_gradient": sum(v > 0 for k, v in gradients.items()
                                                    if k.startswith("load_")),
                  "checkpoint_sha256": sha256(run / "initialized-checkpoint.pt"),
                  "driving_trained": False, "optimizer_updates": 0,
                  "stimulus_source": "synthetic RGB and declared organ stimuli; not AC",
                  "mirrors_implemented": False, "actuator_messages_sent": 0,
                  "goal_achieved": False}
        write_json(run / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error), "goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
