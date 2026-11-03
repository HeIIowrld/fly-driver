"""Independently refit training-only ridge and inspect held-out visual motion results."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import CalibratedVisualSelfMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.calibration)
    receipt = json.loads((root / "result.json").read_text(encoding="utf8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))["config"]
    if sha256(root / "responses.npz") != receipt["responses_sha256"]:
        raise ValueError("Calibration response hash mismatch")
    decoder = CalibratedVisualSelfMotion(root)
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=None,
            source_result_sha256=sha256(root / "result.json"),
            calibration_sha256=decoder.calibration_sha256,
            actuator_messages_sent=0,
        ),
    )
    with load_array(root / "responses.npz", allow_pickle=False) as archive:
        x, y, prediction = (archive[key].copy() for key in ("features", "targets", "prediction"))
        time_features = archive["feature_time"].copy()
        train = archive["protocols"] == "train"
        seeds, amplitudes = archive["texture_seeds"].copy(), archive["amplitudes"].copy()
    if set(seeds[train]) & set(seeds[~train]):
        raise ValueError("Training and validation texture seeds overlap")
    plan = manifest["plan"]
    first, stop = plan["reference_baseline_frames"]
    reconstructed = time_features[:, -plan["average_last_frames"] :].mean(axis=1) - time_features[
        :, first:stop
    ].mean(axis=1)
    np.testing.assert_array_equal(x, reconstructed)
    np.testing.assert_array_equal(x[train].mean(axis=0), decoder.center)
    np.testing.assert_array_equal(x[train].std(axis=0), decoder.scale)
    design = np.column_stack([(x - decoder.center) / decoder.scale, np.ones(len(x))])
    penalty = np.diag([np.sqrt(plan["ridge"])] * 6 + [0.0])
    independent = np.linalg.lstsq(
        np.vstack([design[train], penalty]), np.vstack([y[train], np.zeros((7, 6))]), rcond=None
    )[0]
    coefficient_error = float(np.abs(independent - decoder.coefficients).max())
    np.testing.assert_allclose(independent, decoder.coefficients, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(design @ decoder.coefficients, prediction, rtol=1e-12, atol=1e-12)
    selected = ~train
    active = np.linalg.norm(y[selected], axis=1) > 0
    target_axes = np.abs(y[selected][active] / amplitudes[selected][active]).argmax(axis=1)
    predicted_axes = np.abs(prediction[selected][active] / amplitudes[selected][active]).argmax(
        axis=1
    )
    correct = int(np.sum(target_axes == predicted_axes))
    constant = np.r_[-decoder.center / decoder.scale, 1.0] @ decoder.coefficients
    removed = np.broadcast_to(constant, prediction[selected].shape)
    removed_rmse = float(np.sqrt(np.mean(((removed - y[selected]) / amplitudes[selected]) ** 2)))
    full_rmse = float(
        np.sqrt(np.mean(((prediction[selected] - y[selected]) / amplitudes[selected]) ** 2))
    )
    details = []
    for index in np.flatnonzero(selected):
        details.append(
            dict(
                texture_seed=int(seeds[index]),
                target=y[index].tolist(),
                normalized_prediction=(prediction[index] / amplitudes[index]).tolist(),
            )
        )
    write_json(run / "held-out-cases.json", details)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    labels = ["roll", "pitch", "yaw", "forward/R", "left/R", "up/R"]
    for axis, values, title in zip(
        axes,
        (y[selected] / amplitudes[selected], prediction[selected] / amplitudes[selected]),
        ("Held-out target", "Frozen visual network + calibrated readout"),
        strict=True,
    ):
        artist = axis.imshow(values, vmin=-1.5, vmax=1.5, cmap="coolwarm", aspect="auto")
        axis.set_xticks(range(6), labels, rotation=30)
        axis.set_ylabel("Held-out cases: two unseen textures, static and signed motions")
        axis.set_title(title)
        figure.colorbar(artist, ax=axis, shrink=0.7)
    figure.savefig(run / "held-out-motion.png", dpi=160)
    plt.close(figure)
    write_json(
        run / "result.json",
        dict(
            status="completed",
            termination_reason="training-only coefficient and held-out evaluation audits passed",
            coefficient_max_error=coefficient_error,
            training_only_normalization=True,
            texture_seed_sets_disjoint=True,
            stationary_prefix_features_exact=True,
            held_out_nonzero_cases=int(active.sum()),
            dominant_axis_correct=correct,
            all_input_removed_normalized_rmse=removed_rmse,
            full_normalized_rmse=full_rmse,
            intervention_scope=("remove calibrated pooled evidence without retraining; "
                                "not a full flight-neuron lesion"),
            real_game_self_motion_validated=False,
            flight_controller_integrated=False,
            physiological_reproduction_verified=False,
            actuator_messages_sent=0,
            goal_achieved=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="completed",
                coefficient_error=coefficient_error,
                dominant_correct=correct,
                cases=int(active.sum()),
                full_rmse=full_rmse,
                removed_rmse=removed_rmse,
            )
        )
    )


if __name__ == "__main__":
    main()
