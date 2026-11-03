"""Calibrate declared organ feedback on a frozen local-retinal circuit checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.local_retinal_circuit import LocalRetinalController
from flybrain_drive.organ_feedback import calibrate_local_feedback
from flybrain_drive.routed_controller import steering_population_decoder
from scripts.evaluate_local_retinal_controller import read_training_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "poses", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--position-gain", type=float, default=.2)
    parser.add_argument("--yaw-gain", type=float, default=.1)
    args = parser.parse_args()
    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.sensory_imitation import checked_environment
    from flybrain_drive.sensory_runtime import make_sensory_runtime

    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    payload = torch.load(args.checkpoint, weights_only=True)
    calibration = read_training_calibration(config, payload, args.poses)
    graph = {k: np.asarray(v, dtype={"pre": np.int64, "post": np.int64,
                                      "weights": np.float32}.get(k))
             for k, v in payload["graph"].items()}
    model = LocalRetinalController(graph, **payload["constructor"])
    model.load_state_dict(payload["model"])
    run = begin_run(args.output, {"arguments": vars(args), "configuration": config,
        "source_checkpoint_sha256": sha256(args.checkpoint), "script_sha256": sha256(__file__),
        "scope": "unit-stimulus organ transduction calibration; no driving or physiology proof"})
    env = None
    try:
        env = FlyBrainEnv(**checked_environment(config))
        _, packer, sealed = make_sensory_runtime(env, config)
        if sealed["sensory_encoder"] != calibration:
            raise ValueError("Organ and eye calibration differs from visual checkpoint")
        decoder, error = steering_population_decoder(packer.organs.positions, env.vehicle,
                                                     packer.organ_size)
        report = calibrate_local_feedback(model, decoder, packer.organs.rotation_scale,
            position_gain=args.position_gain, yaw_gain=args.yaw_gain)
        report.update(position_decoder_rmse=error, source_checkpoint_sha256=sha256(args.checkpoint))
        payload["organ_feedback"] = report
        payload["model"] = model.state_dict()
        torch.save(payload, run / "checkpoint.pt")
        restored = LocalRetinalController(graph, **payload["constructor"])
        restored.load_state_dict(torch.load(run / "checkpoint.pt", weights_only=True)["model"])
        a, b = model.initial_state(), restored.initial_state()
        drive = model.initial_state()
        drive[:, model.adapter.visual_ports] = .01
        with torch.no_grad():
            for _ in range(20):
                x, a = model.advance(drive, a)
                y, b = restored.advance(drive, b)
                torch.testing.assert_close(x, y, atol=0, rtol=0)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        write_json(run / "calibration.json", report)
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False,
            "checkpoint_restore_max_error": 0, "checkpoint_sha256": sha256(run / "checkpoint.pt")})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error)})
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
