"""Probe the actual flight candidate on GPU with explicit sign uncertainty and no actuators."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/flight-circuit-prototype-20261029.json")
    parser.add_argument("--visual-calibration", required=True)
    parser.add_argument("--scene-replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no fallback")
    source = Path(args.visual_calibration)
    visual_receipt = json.loads((source / "result.json").read_text(encoding="utf8"))
    if not visual_receipt["calibration_gate_passed"]:
        raise ValueError("Visual adapter gate required")
    if sha256(source / "responses.npz") != visual_receipt["responses_sha256"]:
        raise ValueError("Visual adapter response changed")
    with load_array(source / "responses.npz", allow_pickle=False) as archive:
        target, predicted, protocol = (
            archive[key] for key in ("targets", "prediction", "protocols")
        )
        visual_examples = {
            sign: predicted[(protocol == "validation") & (target[:, 2] * sign > 0)].mean(axis=0)
            for sign in (-1, 1)
        }
    scene_root = Path(args.scene_replay)
    scene_receipt = json.loads((scene_root / "result.json").read_text(encoding="utf8"))
    if sha256(scene_root / "responses.npz") != scene_receipt["responses_sha256"]:
        raise ValueError("Retained scene response changed")
    with load_array(scene_root / "responses.npz", allow_pickle=False) as archive:
        retinal = archive["retina_intensity"][[20, -1]]
        motion = archive["motion_activity"][[20, -1]]
        scene_examples = np.concatenate([retinal[:, :, None, :], motion], axis=2)
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=863050,
            spec_sha256=sha256(args.spec),
            visual_response_sha256=visual_receipt["responses_sha256"],
            scene_response_sha256=scene_receipt["responses_sha256"],
            glutamate_signs=[-1, 0, 1],
            dt=0.01,
            steps=200,
            pulse_indices=[50, 100],
            scope="untrained technical prototype, no flight reflex or driving success claim",
            actuator_messages_sent=0,
            optimizer_updates=0,
        ),
    )
    cases = [
        "baseline",
        "visual_yaw_negative",
        "visual_yaw_positive",
        "body_yaw_negative",
        "body_yaw_positive",
        "body_roll_negative",
        "body_roll_positive",
        "body_pitch_negative",
        "body_pitch_positive",
        "goal_turn_negative",
        "goal_turn_positive",
        "goal_forward_negative",
        "goal_forward_positive",
        "spatial_scene_change",
    ]
    batch = len(cases)
    visual = torch.zeros((batch, 6), device="cuda")
    body = torch.zeros((batch, 16), device="cuda")
    body[:, 8:12] = 0.25
    body[:, 12:16] = 27.0
    visual_valid = torch.ones_like(visual, dtype=torch.bool)
    body_valid = torch.zeros_like(body, dtype=torch.bool)
    body_valid[:, [0, 1, 4] + list(range(8, 16))] = True
    goals = torch.zeros((batch, 2), device="cuda")
    scene = torch.tensor(
        np.repeat(scene_examples[0:1], batch, axis=0), device="cuda", dtype=torch.float32
    )
    pulse_scene = scene.clone()
    pulse_scene[-1] = torch.tensor(scene_examples[1], device="cuda", dtype=torch.float32)
    scene_valid = torch.ones((batch, 4), device="cuda", dtype=torch.bool)
    pulse_visual, pulse_body, pulse_goals = visual.clone(), body.clone(), goals.clone()
    for sign, index in [(-1, 1), (1, 2)]:
        pulse_visual[index] = torch.tensor(
            visual_examples[sign], device="cuda", dtype=torch.float32
        )
    pulse_body[3, 4], pulse_body[4, 4] = -0.23, 0.23
    pulse_body[5, 1], pulse_body[6, 1] = -0.05, 0.05
    pulse_body[7, 0], pulse_body[8, 0] = -0.05, 0.05
    pulse_goals[9, 0], pulse_goals[10, 0] = -0.5, 0.5
    pulse_goals[11, 1], pulse_goals[12, 1] = -0.5, 0.5
    summaries = []
    for sign in (-1, 0, 1):
        torch.manual_seed(863050)
        model = FlightCircuit(args.spec, glutamate_sign=sign).to("cuda")
        state = model.initial_state(batch)
        intents, muscles = [], []
        with torch.no_grad():
            for step in range(200):
                if 50 <= step < 100:
                    values = (pulse_visual, pulse_body, pulse_goals)
                else:
                    values = (visual, body, goals)
                intent, motor, state = model(
                    values[0],
                    visual_valid,
                    values[1],
                    body_valid,
                    values[2],
                    state,
                    scene=pulse_scene if 50 <= step < 100 else scene,
                    scene_valid=scene_valid,
                )
                intents.append(intent.cpu().numpy())
                muscles.append(motor.cpu().numpy())
        intent_trace, motor_trace = np.stack(intents), np.stack(muscles)
        effect = motor_trace[99] - motor_trace[99, 0]
        if not np.isfinite(motor_trace).all() or np.max(np.abs(effect[1:])) == 0:
            raise ValueError("No finite measured-path motor effect")
        # Every learned sensory projection must receive gradients through measured edges.
        model.zero_grad(set_to_none=True)
        state = model.initial_state(batch)
        for _ in range(60):
            intent, _, state = model(
                pulse_visual,
                visual_valid,
                pulse_body,
                body_valid,
                pulse_goals,
                state,
                scene=pulse_scene,
                scene_valid=scene_valid,
            )
        intent.square().mean().backward()
        gradient_norms = {
            name: float(getattr(model, name).weight.grad.norm().item())
            for name in ("vision", "body", "goal", "context")
        }
        gradient_norms["retinotopic_scene"] = float(model.vision.weight.grad[:, 12:].norm().item())
        gradient_norms["visual_motion"] = float(model.vision.weight.grad[:, :6].norm().item())
        if not all(np.isfinite(value) and value > 0 for value in gradient_norms.values()):
            raise ValueError("A declared input route is not trainable through the graph")
        model.zero_grad(set_to_none=True)
        # Remove graph messages, retaining adapters/readout. No direct sensory shortcut may remain.
        original_matrix = model.circuit.matrix
        model.circuit.matrix = original_matrix * 0
        state = model.initial_state(batch)
        with torch.no_grad():
            for _ in range(60):
                cut_intent, cut_motor, state = model(
                    pulse_visual,
                    visual_valid,
                    pulse_body,
                    body_valid,
                    pulse_goals,
                    state,
                    scene=pulse_scene,
                    scene_valid=scene_valid,
                )
        if torch.any(cut_motor != 0) or torch.any(cut_intent - cut_intent[0:1] != 0):
            raise ValueError("Motor response bypasses removed measured edges")
        model.circuit.matrix = original_matrix
        label = f"glutamate-{sign:+d}"
        torch.save(model.state_dict(), run / (label + "-untrained.pt"))
        np.savez_compressed(
            run / (label + "-responses.npz"),
            intent=intent_trace,
            muscles=motor_trace,
            cases=np.array(cases),
            dt=np.array(0.01),
        )
        summary = dict(
            glutamate_sign=sign,
            sign_counts=model.sign_counts,
            effective_structural_edges=model.effective_edges,
            motor_effect_by_case={name: effect[index].tolist() for index, name in enumerate(cases)},
            input_gradient_norms=gradient_norms,
            zero_edge_motor_response_zero=True,
            trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        )
        summaries.append(summary)
        write_json(run / "progress.json", dict(completed_sign_variants=len(summaries), total=3))
        del model, state
    write_json(
        run / "result.json",
        dict(
            status="completed_untrained_technical_probe",
            termination_reason=("three declared sign hypotheses "
                                "and graph-dependence checks completed"),
            cases=cases,
            sign_variants=summaries,
            structural_nodes=9728,
            structural_edges=1311145,
            body_context_separate_from_haltere_rotation=True,
            required_flight_function_verified=False,
            retinotopic_scene_path_present=True,
            scene_features=576,
            wingbeat_phase_implemented=False,
            electrical_synapses_implemented=False,
            physiological_reproduction_verified=False,
            actual_actor_integration=False,
            actuator_messages_sent=0,
            optimizer_updates=0,
            goal_achieved=False,
        ),
    )
    print(json.dumps(dict(status="completed_untrained_technical_probe", variants=3)))


if __name__ == "__main__":
    main()
