"""Verify saved stream alignment and numerical state continuity, never driving controls."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--body-replay", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument('--full-chronological', action='store_true',
                        help='Compare all saved outputs using original clocks and held batches')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    os.environ["FLYVIS_ROOT_DIR"] = str(Path(args.cache_root).resolve())
    os.environ["MPLBACKEND"] = "Agg"
    source, model = Path(args.replay), Path(args.model_dir)
    receipt = json.loads((source / "result.json").read_text(encoding="utf8"))
    if (
        receipt["status"] != "completed"
        or sha256(source / "responses.npz") != receipt["responses_sha256"]
    ):
        raise ValueError("Successful unchanged stream replay required")
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=863009,
            input_result_sha256=sha256(source / "result.json"),
            checkpoint_sha256=sha256(model / "best_chkpt"),
            author_meta_sha256=sha256(model / '_meta.yaml'),
            input_responses_sha256=receipt["responses_sha256"],
            actuator_messages_sent=0,
        ),
    )
    with load_array(source / "responses.npz", allow_pickle=False) as archive:
        central = archive["central_activity"].copy()
        motion = archive["motion_activity"].copy()
        retina = archive["retina_intensity"].copy()
        clock = archive["decision_monotonic_seconds"].copy()
        captures = archive["capture_ids"].copy()
        phases = archive["phases"].copy()
    body_root = Path(args.body_replay)
    body_receipt = json.loads((body_root / "result.json").read_text(encoding="utf8"))
    if (body_receipt["status"] != "completed"
            or sha256(body_root / "virtual-body.npz") != body_receipt["virtual_body_sha256"]):
        raise ValueError("Successful unchanged complete body replay required")
    with load_array(body_root / "virtual-body.npz", allow_pickle=False) as body:
        np.testing.assert_array_equal(clock, body["decision_monotonic_seconds"])
        np.testing.assert_array_equal(phases, body["phases"])
    count = body_receipt["observations"]
    if (count < 8 or receipt["observations"] != count or len(clock) != count
            or central.shape != (count, 4, 65) or motion.shape != (count, 4, 8, 721)
            or retina.shape != (count, 4, 721)):
        raise ValueError("Full original stream and retinotopic motion maps required")
    if not all(np.isfinite(array).all() for array in (central, motion, retina)):
        raise ValueError("Nonfinite saved visual circuit response")
    duplicate = np.flatnonzero(np.diff(captures) == 0) + 1
    np.testing.assert_array_equal(retina[duplicate], retina[duplicate - 1])
    install_datamate_cache_compat(args.cache_root)
    import flyvis

    torch.set_default_device("cuda")
    config = yaml.safe_load((model / "_meta.yaml").read_text(encoding="utf8"))
    network = flyvis.Network(**config["config"]["network"]).to("cuda")
    weights = torch.load(model / "best_chkpt", map_location="cpu", weights_only=True)["network"]
    network.load_state_dict(weights, strict=True)
    network.eval().requires_grad_(False)
    # Actual four-camera inputs; whole versus partitioned 8-tick original dynamics.
    movie = torch.tensor(np.transpose(retina[:8], (1, 0, 2))[:, :, None, :], device="cuda")
    with torch.no_grad():
        initial = network.steady_state(1.0, 0.005, batch_size=4, value=0.5)
        full = network.simulate(movie, dt=0.005, initial_state=initial)[:, -1]
        state = network.simulate(movie[:, :3], dt=0.005, initial_state=initial, as_states=True)[-1]
        split = network.simulate(movie[:, 3:], dt=0.005, initial_state=state, as_states=True)[-1]
        reset = network.simulate(movie[:, 3:], dt=0.005, initial_state=initial)[:, -1]
        permutation = [2, 0, 3, 1]
        permuted = network.simulate(movie[permutation], dt=0.005, initial_state=initial)[:, -1]
        torch.testing.assert_close(full, split.nodes.activity, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(full[permutation], permuted, rtol=2e-5, atol=2e-6)
        split_error = float((full - split.nodes.activity).abs().max().item())
        batch_error = float((full[permutation] - permuted).abs().max().item())
        reset_difference = float((full - reset).abs().max().item())
        if reset_difference < 1e-5:
            raise ValueError("Continuity counterexample not sensitive enough to state reset")
        chronological = None
        if args.full_chronological:
            with load_array(source/'retina-registration.npz', allow_pickle=False) as registration:
                motion_indices = registration['motion_indices'].copy()
            central_indices = network.connectome.central_cells_index[:]
            current = network.steady_state(1., .005, batch_size=4, value=.5)
            maximum_central, maximum_motion, ticks = 0., 0., 0
            for index in range(count):
                if index:
                    previous = torch.tensor(retina[index-1], device='cuda')
                    for dt, steps in held_input_batches(float(clock[index]-clock[index-1])):
                        held = previous[:, None, None].expand(-1, steps, -1, -1)
                        current = network.simulate(held, dt=dt, initial_state=current,
                                                   as_states=True)[-1]
                        ticks += steps
                actual_central = current.nodes.activity[:, central_indices].cpu().numpy()
                actual_motion = current.nodes.activity[:, motion_indices].cpu().numpy()
                np.testing.assert_allclose(actual_central, central[index], rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(actual_motion, motion[index], rtol=2e-5, atol=2e-6)
                maximum_central = max(maximum_central,
                                      float(np.abs(actual_central-central[index]).max()))
                maximum_motion = max(maximum_motion,
                                     float(np.abs(actual_motion-motion[index]).max()))
                if index % 100 == 0:
                    write_json(run/'chronological-progress.json', dict(
                        compared=index+1, total=count, original_internal_ticks=ticks))
            assert ticks == receipt['internal_ticks']
            chronological = dict(observations_compared=count, original_internal_ticks=ticks,
                maximum_central_error=maximum_central, maximum_motion_error=maximum_motion,
                actual_original_clock_schedule=True, earlier_retina_held_for_interval=True,
                neural_state_resets_after_initialization=0, source_camera_samples_created=0,
                comparison='Saved one-tick calls versus original equal-dt held-input batches')
    import matplotlib.pyplot as plt

    registration = load_array(source / "retina-registration.npz", allow_pickle=False)
    directions = registration["body_directions"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for index, name in enumerate(["front", "left", "right", "rear"]):
        ray = directions[index]
        azimuth = np.rad2deg(np.arctan2(ray[:, 1], ray[:, 0]))
        elevation = np.rad2deg(np.arcsin(ray[:, 2]))
        axes[0].scatter(azimuth, elevation, s=2, label=name)
        axes[1].plot(
            clock - clock[0],
            np.maximum(motion[:, index], 0).mean(axis=(1, 2)),
            label=name,
            linewidth=0.7,
        )
    axes[0].set(
        xlabel="Body azimuth (degrees, left positive)",
        ylabel="Elevation (degrees)",
        title="Registered viewing directions; camera origins remain separate",
    )
    axes[1].set(
        xlabel="Original decision elapsed time (s)",
        ylabel="Mean rectified T4/T5 activity",
        title="Retained real AC images through author visual circuit",
    )
    for axis in axes:
        axis.legend(markerscale=3, fontsize=8)
    figure.savefig(run / "camera-response-summary.png", dpi=160)
    plt.close(figure)
    write_json(
        run / "result.json",
        dict(
            status="completed",
            termination_reason="full saved stream alignment and numerical state tests passed",
            observations=len(clock),
            exact_body_clock_and_phase_alignment=True,
            reused_capture_retina_equal=True,
            repeated_capture_decisions=len(duplicate),
            finite_retinotopic_maps=True,
            partitioned_state_max_error=split_error,
            camera_batch_permutation_max_error=batch_error,
            reset_counterexample_max_difference=reset_difference,
            neurons_per_independent_camera=network.n_nodes,
            measured_fly_optical_calibration=False,
            camera_specific_angular_scales=True,
            actual_actor_integration=False,
            functional_flight_fusion_verified=False,
            physiological_reproduction_verified=False,
            full_chronological_comparison=chronological,
            actuator_messages_sent=0,
            optimizer_updates=0,
            goal_achieved=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="completed",
                split_error=split_error,
                batch_error=batch_error,
                reset_counterexample_difference=reset_difference,
            )
        )
    )


if __name__ == "__main__":
    main()
