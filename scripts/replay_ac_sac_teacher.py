"""Reconstruct a versioned privileged teacher; never connect to or control the game."""

import argparse
import ast
import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from flybrain_drive.ac_sac_teacher import (
    CHANNELS,
    PrivilegedSACTeacher,
    TeacherReferenceCurve,
    channel_scales,
    encode_teacher_observation,
    relative_request,
)
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.numeric_pickle import load_numeric_pickle

ENV_PATH = "assetto_corsa_gym/AssettoCorsaEnv/ac_env.py"
REF_PATH = "assetto_corsa_gym/AssettoCorsaEnv/reference_lap.py"
CURVE_PATH = "assetto_corsa_gym/AssettoCorsaEnv/curvature.py"
CSV_PATH = "assetto_corsa_gym/AssettoCorsaConfigs/tracks/monza-racing_line.csv"


def numeric_literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id == "TOP_SPEED_MS":
        return 80.
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "np" and node.attr == "pi":
            return np.pi
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return numeric_literal(node.left) / numeric_literal(node.right)
    if isinstance(node, ast.Dict):
        return {numeric_literal(k): numeric_literal(v)
                for k, v in zip(node.keys, node.values, strict=True)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [numeric_literal(value) for value in node.elts]
    raise ValueError("Unreviewed source expression in numeric contract")


def source_attributes(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and any(isinstance(f, ast.FunctionDef) and f.name == "get_obs" for f in node.body)]
    if len(classes) != 1:
        raise ValueError("Expected one upstream observation class")
    wanted = ("obs_enabled_channels", "obs_channels_info")
    return {target.id: numeric_literal(node.value) for node in classes[0].body
            if isinstance(node, ast.Assign) for target in node.targets
            if isinstance(target, ast.Name) and target.id in wanted}


def reviewed_function(path, name, namespace, *, method=False):
    """Execute only previously inspected numeric definitions from SHA-verified files.

    No upstream module imports, plugin startup, pickle loader or actuator code runs.
    """
    tree = ast.parse(path.read_text(encoding="utf8"))
    candidates = []
    bodies = [node.body for node in tree.body if isinstance(node, ast.ClassDef)] if method else [
        tree.body]
    for body in bodies:
        candidates.extend(node for node in body if isinstance(node, ast.FunctionDef)
                          and node.name == name)
    if len(candidates) != 1:
        raise ValueError("Expected one reviewed numerical function: " + name)
    selected = copy.deepcopy(candidates[0])
    if any(isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal))
           for node in ast.walk(selected)):
        raise ValueError("Imports/global changes are outside the reviewed numeric fragment")
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def source_reference(root, reference):
    from scipy.interpolate import UnivariateSpline, interp1d

    namespace = {"np": np, "UnivariateSpline": UnivariateSpline, "interp1d": interp1d,
                 "__builtins__": {"len": len, "range": range, "int": int,
                                   "print": lambda *_: None}}
    calculate = reviewed_function(root / REF_PATH, "calculate_distance_from_xy", namespace)
    convert = reviewed_function(root / REF_PATH, "convert_to_distance", namespace)
    curvature = reviewed_function(root / CURVE_PATH, "curvature_splines", namespace)
    points = np.genfromtxt(root / CSV_PATH, delimiter=",", names=True)
    distance = calculate(points["pos_x"], points["pos_y"])
    curve = curvature(points["pos_x"], points["pos_y"])
    dist_grid, curve_grid = convert(distance, distance), convert(distance, curve)
    error = max(float(np.max(np.abs(dist_grid-reference.distance))),
                float(np.max(np.abs(curve_grid-reference.curvature))))
    proxy = SimpleNamespace(channels_dist=["position.x", "position.y", "lapDistance", "yaw",
                                           "curvature"], distance_ch_dist=dist_grid,
                            td=np.c_[np.zeros((len(dist_grid), 4)), curve_grid])
    for name in ("distSegment2Index", "getLADVector", "get_curvature_segment"):
        setattr(proxy, name, MethodType(reviewed_function(root / REF_PATH, name, namespace,
                                                          method=True), proxy))
    return proxy, error


def reference_observation(root, reference_proxy):
    attributes = source_attributes(root / ENV_PATH)
    if tuple(attributes["obs_enabled_channels"]) != CHANNELS:
        raise ValueError("Source observation order differs from reviewed channel list")
    scales = np.array([attributes["obs_channels_info"][key] for key in CHANNELS])
    proxy = SimpleNamespace(obs_enabled_channels=list(CHANNELS), obs_channels_scales=scales,
        enable_sensors=True, ref_lap=reference_proxy, action_dim=3, add_previous_obs_to_state=True,
        previous_obs_to_state_dim=25, obs_channels_info=attributes["obs_channels_info"],
        config=SimpleNamespace(enable_task_id_in_obs=False))
    namespace = {"np": np, "MAX_RAY_LEN": 200., "PAST_ACTIONS_WINDOW": 3,
                 "CURV_LOOK_AHEAD_DISTANCE": 300., "CURV_LOOK_AHEAD_VECTOR_SIZE": 12,
                 "CURV_NORMALIZATION_CONSTANT": .1, "__builtins__": {"len": len, "range": range}}
    function = reviewed_function(root / ENV_PATH, "get_obs", namespace, method=True)
    return lambda state, history: function(proxy, state, history)[0], scales


def numpy_policy(state, observations):
    result = observations.astype(np.float32)
    for index in (0, 2, 4, 6):
        result = (result @ state[f"net.{index}.weight"].numpy().T
                  + state[f"net.{index}.bias"].numpy())
        if index != 6:
            result = np.maximum(result, 0)
    return np.tanh(result[:, :3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ac-sac-replay.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    run = begin_run(args.output, config)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    try:
        recipe = Path(config["recipe_root"])
        source = json.loads((recipe / "source.json").read_text(encoding="utf8"))
        for record in source["files"]:
            if sha256(recipe / record["path"]) != record["sha256"]:
                raise ValueError("Reviewed numerical source hash changed")
        record_path = Path(config["record_path"])
        if sha256(record_path) != config["record_sha256"]:
            raise ValueError("Published recorded sample hash changed")
        data = load_numeric_pickle(record_path.read_bytes())
        states = data["states"]
        if len(states) < 4 or len(states) > 100000:
            raise ValueError("Finite episode record with preceding history required")
        steps = np.array([row["steps"] for row in states])
        times = np.array([row["timestamp_ac"] for row in states])
        if not (np.diff(steps) > 0).all() or not (np.diff(times) > 0).all():
            raise ValueError("Recorded sequence/time must increase within the source file")
        reference = TeacherReferenceCurve(recipe / "legacy" / CSV_PATH)
        original_ref, curve_error = source_reference(recipe / "legacy", reference)
        original_obs, scales = reference_observation(recipe / "legacy", original_ref)
        np.testing.assert_array_equal(scales, channel_scales(config["profile"]))
        indices = np.arange(3, len(states))
        observations, modern, source_observations = [], [], []
        for index in indices:
            row, history = states[index], states[max(0, index-3):index]
            curve = reference.segment(row["LapDist"])
            observations.append(encode_teacher_observation(row, history, curve,
                                                             profile=config["profile"]))
            modern.append(encode_teacher_observation(row, history, curve, profile="current_2025"))
            source_observations.append(original_obs(row, history))
        observations = np.asarray(observations)
        obs_error = float(np.max(np.abs(observations - source_observations)))
        teacher = PrivilegedSACTeacher(config["checkpoint"], config["checkpoint_sha256"])
        inference = teacher.predict(observations)
        actions = inference["deterministic_policy_action"]
        raw_weights = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
        neural_error = float(np.max(np.abs(actions - numpy_policy(raw_weights, observations))))
        wrong_profile_difference = np.max(np.abs(
            actions-teacher.predict(np.asarray(modern))["deterministic_policy_action"]), axis=0)
        recorded_absolute = np.array([[row[f"current_action_abs_{i}"] for i in range(3)]
                                       for row in states])
        raw_actions = np.array([[row[f"actions_{i}"] for i in range(3)] for row in states])
        predicted = np.array([relative_request(a, prior, config["legacy_control_max_delta"],
            [-1]*3, [1]*3, product_dtype=config["legacy_product_dtype"])
            for a, prior in zip(raw_actions[1:], recorded_absolute[:-1], strict=True)])
        codec_error = np.max(np.abs(predicted-recorded_absolute[1:]), axis=0)
        requested = np.array([relative_request(a, prior, config["legacy_control_max_delta"],
            [-1]*3, [1]*3, product_dtype=config["legacy_product_dtype"])
            for a, prior in zip(actions, recorded_absolute[indices], strict=True)])
        corners = ("fl", "fr", "rl", "rr")
        temperatures = np.array([[row[c+"_tire_temperature_core"] for c in corners]
                                 for row in states])
        loads = np.array([[row[c+"_wheel_load"] for c in corners] for row in states])
        image_candidates = sorted({key for row in states for key, value in row.items()
            if any(name in key.lower() for name in ("image", "rgb", "camera"))
            or isinstance(value, bytes) or np.asarray(value).ndim >= 2})
        np.savez_compressed(run / "privileged-teacher-replay.npz", record_indices=indices,
            teacher_observations=observations, teacher_policy_delta_action=actions,
            suggested_absolute_request=requested, log_std=inference["log_std"],
            recorded_core_temperature_raw=temperatures, recorded_wheel_load_raw=loads)
        thresholds = config["thresholds"]
        checks = {
            "observation_order_and_scaling":
                obs_error <= thresholds["observation_max_absolute_error"],
            "reference_numeric_parity": curve_error <= thresholds["reference_max_absolute_error"],
            "torch_numpy_policy_parity":
                neural_error <= thresholds["action_numpy_torch_max_absolute_error"],
            "relative_request_parity_with_record":
                float(codec_error.max()) <= thresholds[
                    "recorded_relative_codec_max_absolute_error"]}
        summary = {"status": "completed", "checks": checks, "all_reconstruction_checks_passed":
            all(checks.values()), "recorded_rows": len(states),
            "teacher_replayed_rows": len(indices),
            "unknown_initial_history_rows_excluded": 3, "state_fields": len(states[0]),
            "observation_max_absolute_error": obs_error,
            "reference_max_absolute_error": curve_error,
            "torch_numpy_action_max_absolute_error": neural_error,
            "recorded_relative_request_max_absolute_error": codec_error.tolist(),
            "wrong_modern_profile_action_max_difference": wrong_profile_difference.tolist(),
            "core_temperature_raw_min": temperatures.min(0).tolist(),
            "core_temperature_raw_max": temperatures.max(0).tolist(),
            "wheel_load_raw_min": loads.min(0).tolist(),
            "wheel_load_raw_max": loads.max(0).tolist(),
            "image_or_encoded_bytes_candidates": image_candidates,
            "source_monotonic_median_interval_seconds": float(np.median(np.diff(times))),
            "reviewed_upstream_numeric_fragments_executed": True,
            "upstream_plugin_or_module_imports_executed": False,
            "unrestricted_pickle_loading": False,
            "exact_training_run_manifest_available": False,
            "actor_stream_created": False, "vision_student_training_ready": False,
            "teacher_game_driving_verified": False, "actuator_messages_sent": 0,
            "goal_achieved": False}
        write_json(run / "result.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error), "goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
