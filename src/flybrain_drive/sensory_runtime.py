"""One calibrated sensory factory shared by collection, evaluation and inference."""

import copy
from dataclasses import asdict

from .body_organs import BodyOrganEncoder, PositionOrganSpec, ScalarOrganSpec
from .mujoco_sensory import MuJoCoSensoryBridge
from .sensory_policy import ObservationPacker


def make_sensory_runtime(env, config):
    settings = copy.deepcopy(config.get("sensory_preprocessing", {}))
    settings.setdefault("visual_encoder", "global_points")
    if settings["visual_encoder"] not in ("global_points", "retinotopic_grid"):
        raise ValueError("Unknown sensory visual encoder")
    eye_settings = settings.setdefault("eye", {"rows": 8, "columns": 12, "sigma_degrees": 1})
    if (eye_settings.get("geometry_path") is not None
            and settings["visual_encoder"] == "retinotopic_grid"):
        raise ValueError("Imported irregular axes are not the synthetic retinotopic grid")
    bridge = MuJoCoSensoryBridge(env, **eye_settings)
    defaults = [PositionOrganSpec("steering_angle", 2, -.5, .5, .15, 1),
                PositionOrganSpec("suspension_position", 4, -.07, .07, .02, 1)]
    settings.setdefault("position_organs", [asdict(p) for p in defaults])
    settings.setdefault("rotation_scale", 5)
    settings.setdefault("specific_force_scale", 20)
    settings.setdefault("wheel_speed_scale", 50)
    settings.setdefault("max_sample_age", .15)
    organs = BodyOrganEncoder([PositionOrganSpec(**spec) for spec in settings["position_organs"]],
                              rotation_scale=settings["rotation_scale"],
                              max_sample_age=settings["max_sample_age"],
                              scalars=[ScalarOrganSpec(**spec)
                                       for spec in settings.get("scalar_organs", [])])
    packer = ObservationPacker(bridge.eyes, organs,
                               specific_force_scale=settings["specific_force_scale"],
                               wheel_speed_scale=settings["wheel_speed_scale"],
                               max_sample_age=settings["max_sample_age"])
    encoder = {"organ_size": packer.organ_size,
               "eye_map_hashes": {eye.side: eye.sha256 for eye in bridge.eyes}}
    if organs.scalars:
        encoder["organ_calibration_sha256"] = organs.calibration_sha256
    if settings["visual_encoder"] == "retinotopic_grid":
        encoder["grid_shape"] = [eye_settings["rows"], eye_settings["columns"]]
    if config.get("sensory_encoder") not in (None, encoder):
        raise ValueError("Checkpoint eye/organ calibration differs from runtime")
    effective = copy.deepcopy(config)
    effective.update(sensory_preprocessing=settings, sensory_encoder=encoder)
    return bridge, packer, effective
