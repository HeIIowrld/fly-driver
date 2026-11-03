"""Causal four-view author vision -> virtual body -> measured flight circuit, inference only.

This module exposes intentions, never actuators. Each instance owns one explicit episode;
any bad source/clock failure invalidates it rather than resetting or retiming hidden state.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .ac_csp_vision import validate_frame_aid_state
from .episode_storage import load_array
from .flight_commands import COMMAND_GOAL_CONTRACT
from .flight_learning import speed_goals
from .flight_navigation import VisualFlightCircuit
from .flyvis_camera import VIEW_ORDER, FlyvisCameraProjector, integration_intervals
from .flyvis_compat import install_datamate_cache_compat
from .graph import sha256
from .virtual_body import VirtualBodyEncoder


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def held_input_batches(elapsed):
    """Group equal internal dt values; never create or advance a camera sample."""
    steps = integration_intervals(elapsed)
    batches = []
    for dt in steps:
        if batches and dt == batches[-1][0]:
            batches[-1] = (dt, batches[-1][1] + 1)
        else:
            batches.append((dt, 1))
    return batches


class CameraSequence:
    """Validate identity/causality without committing partial source state."""

    def __init__(self, *, maximum_interval=.08, maximum_camera_age=.15):
        if (not np.isfinite([maximum_interval, maximum_camera_age]).all()
                or not 0 < maximum_interval <= .08 or not 0 < maximum_camera_age <= .15):
            raise ValueError("Explicit original-interval and camera-age budgets required")
        self.maximum_interval, self.maximum_camera_age = maximum_interval, maximum_camera_age
        self.clock, self.generation, self.capture, self.signature = None, None, None, None

    def preview(self, clock, camera, pixels):
        validate_frame_aid_state(camera)
        if (type(clock) not in (int, float) or not np.isfinite(clock) or clock < 0
                or not isinstance(camera.get("generation"), str)
                or not camera["generation"] or type(camera.get("capture_id")) is not int
                or camera["capture_id"] < 0):
            raise ValueError("Finite clock and exact camera identity required")
        elapsed = None if self.clock is None else clock - self.clock
        if elapsed is not None and not 0 < elapsed <= self.maximum_interval:
            raise ValueError("Original positive interval exceeds stream budget; no gap repair")
        if self.generation is not None and camera["generation"] != self.generation:
            raise ValueError("Camera generation changed; no implicit neural reset")
        if self.capture is not None and camera["capture_id"] < self.capture:
            raise ValueError("Camera capture reversed")
        received = camera.get("received")
        if (type(received) not in (int, float) or not np.isfinite(received)
                or not 0 <= received <= clock):
            raise ValueError("Complete camera receipt must precede decision")
        if (set(pixels) != set(VIEW_ORDER) or len(camera["views"]) != 4
                or {view["view_id"] for view in camera["views"]} != set(VIEW_ORDER)):
            raise ValueError("All four registered views required; no image fallback")
        views = {view["view_id"]: view for view in camera["views"]}
        signature = [received]
        for name in VIEW_ORDER:
            rgb = pixels[name]
            if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                    or rgb.shape != (180, 320, 3)):
                raise ValueError("Exact source RGB pixels required")
            start = views[name]["render_started_lower"]
            if (type(start) not in (int, float) or not np.isfinite(start)
                    or not 0 <= clock - start <= self.maximum_camera_age):
                raise ValueError("Camera source is stale or from future")
            signature.append((start, hashlib.sha256(rgb.tobytes()).hexdigest()))
        signature = tuple(signature)
        if camera["capture_id"] == self.capture and signature != self.signature:
            raise ValueError("Repeated capture bytes or original source times changed")
        return elapsed, signature

    def commit(self, clock, camera, signature):
        self.clock, self.generation = float(clock), camera["generation"]
        self.capture, self.signature = camera["capture_id"], signature


class FlightStream:
    def __init__(self, config_file, *, native_identity, episode_id):
        initialized = time.perf_counter()
        config = read(config_file)
        if config["schema"] != "flybrain.flight_stream.v1" or not episode_id:
            raise ValueError("Explicit stream schema and episode required")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required; no CPU fallback")
        self.config, self.episode_id, self.device = config, episode_id, "cuda"
        self.integration_mode = config.get("visual_integration", "serial")
        if self.integration_mode not in ("serial", "held_input_batch"):
            raise ValueError("Explicit causal visual integration mode required")
        for key in ("camera_config", "body_contract", "retina_registration"):
            if sha256(config[key]) != config[key + "_sha256"]:
                raise ValueError("Stream source changed: " + key)
        training = Path(config["training"])
        receipt, manifest = read(training / "result.json"), read(training / "manifest.json")
        self.spec_file = manifest["config"]["plan"]["spec"]
        if (sha256(receipt["checkpoint"]) != config["checkpoint_sha256"]
                or receipt["checkpoint_sha256"] != config["checkpoint_sha256"]
                or sha256(self.spec_file) != manifest["config"]["spec_sha256"]):
            raise ValueError("Exact reviewed flight checkpoint/spec required")
        payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
        if (payload["schema"] != "flybrain.flight.visual_commands.v6"
                or payload["goal_contract"] != COMMAND_GOAL_CONTRACT
                or payload["spec_sha256"] != sha256(self.spec_file)):
            raise ValueError("Explicit original v6 command contract required")
        self.flight = VisualFlightCircuit(self.spec_file, glutamate_sign=payload["glutamate_sign"])
        self.flight.load_state_dict(payload["model"], strict=True)
        self.flight.to(self.device).eval().requires_grad_(False)
        self.flight.assert_body_boundary_protected()
        self.body = VirtualBodyEncoder(config["body_contract"], native_identity=native_identity)
        self.order = CameraSequence(maximum_interval=config["maximum_interval_seconds"],
                                    maximum_camera_age=config["maximum_camera_age_seconds"])
        self.goals = torch.tensor(speed_goals(config["task_speed_kmh"], 1), device=self.device)
        self.visual_zero = torch.zeros((1, 6), device=self.device)
        self.visual_valid = torch.zeros((1, 6), dtype=torch.bool, device=self.device)
        self.flight_state = self.flight.initial_state(1)
        self.flight_held, self.retina_held, self.retina = None, None, None
        self.last_motion, self.last_features, self.last_body = None, None, None
        self.steps, self.visual_ticks, self.flight_ticks, self.reused_captures = 0, 0, 0, 0
        self.failed = False
        self._load_vision(config)
        self.initialization_seconds = time.perf_counter() - initialized

    def _load_vision(self, config):
        # The author's import changes torch's global default device. Keep that
        # process-wide side effect outside this inference component's contract.
        previous = torch.get_default_device()
        try:
            self._initialize_vision(config)
        finally:
            torch.set_default_device(previous)

    def _initialize_vision(self, config):
        import yaml

        model_dir = Path(config["author_model_dir"])
        if (sha256(model_dir / "best_chkpt") != config["author_weights_sha256"]
                or sha256(model_dir / "_meta.yaml") != config["author_meta_sha256"]):
            raise ValueError("Pinned author weights and metadata required")
        os.environ["FLYVIS_ROOT_DIR"] = str(Path(config["cache_root"]).resolve())
        install_datamate_cache_compat(config["cache_root"])
        import flyvis

        meta = yaml.safe_load((model_dir / "_meta.yaml").read_text(encoding="utf8"))
        with torch.device(self.device), torch.no_grad():
            self.network = flyvis.Network(**meta["config"]["network"]).to(self.device)
            weights = torch.load(model_dir / "best_chkpt", map_location="cpu", weights_only=True)
            self.network.load_state_dict(weights["network"], strict=True)
            self.network.eval().requires_grad_(False)
            indices = self.network.stimulus.input_index[0]
            nodes = self.network.connectome.nodes
            u, v = nodes.u[:][indices], nodes.v[:][indices]
            self.projector = FlyvisCameraProjector(read(config["camera_config"]), u, v)
            names, all_u, all_v = nodes.type[:].astype(str), nodes.u[:], nodes.v[:]
            layers = []
            for name in ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]:
                mapping = {(int(all_u[i]), int(all_v[i])): i for i in np.flatnonzero(names == name)}
                layers.append([mapping[(int(q), int(r))] for q, r in zip(u, v, strict=True)])
            self.motion_indices = torch.tensor(layers, dtype=torch.long, device=self.device)
            with load_array(config["retina_registration"], allow_pickle=False) as expected:
                np.testing.assert_array_equal(expected["image_uv"], self.projector.uv)
                np.testing.assert_array_equal(expected["body_directions"],
                                              self.projector.directions_body)
                np.testing.assert_array_equal(expected["motion_indices"], layers)
            # Compile/cache both inference paths using disposable scratch states. No
            # native samples, source clocks, order commits or observed steps are created.
            grey = np.full((180, 320, 3), 128, np.uint8)
            warm_retinal = torch.tensor(np.stack([self.projector.sample(grey)] * 4),
                                        device=self.device)
            warm_visual = self.network.steady_state(1., .005, batch_size=4, value=.5)
            warm_count = 8 if self.integration_mode == "held_input_batch" else 1
            warm_input = warm_retinal[:, None, None].expand(-1, warm_count, -1, -1)
            warm_visual = self.network.simulate(warm_input, dt=.005,
                initial_state=warm_visual, as_states=True)[-1]
            warm_motion = warm_visual.nodes.activity[:, self.motion_indices]
            warm_scene = torch.cat([warm_retinal[:, None], warm_motion], dim=1)[None]
            warm_features = self.flight.scene_pool(warm_scene)
            warm_body = torch.zeros((1, 16), device=self.device)
            warm_valid = torch.zeros((1, 16), dtype=torch.bool, device=self.device)
            warm_drive = self.flight._drive_tensor(self.visual_zero, self.visual_valid,
                warm_body, warm_valid, self.goals, warm_features)
            warm_flight = self.flight._advance_elapsed_tensor(warm_drive,
                self.flight.initial_state(1), .01)
            self.flight._read_state_tensor(warm_flight)[0].cpu()
            # Initialize the actual numerical episode after the disposable warmup.
            # Its flight state is still the original zero, with no held inputs.
            self.visual_state = self.network.steady_state(1., .005, batch_size=4, value=.5)

    @torch.no_grad()
    def advance(self, *, clock, camera, pixels, native_sample):
        if self.failed:
            raise RuntimeError("Failed stream cannot resume or silently reset")
        started = time.perf_counter()
        try:
            elapsed, signature = self.order.preview(clock, camera, pixels)
            body = self.body.encode(native_sample, episode_id=self.episode_id,
                                    decision_monotonic_seconds=clock)
            if not body.valid[[0, 1, 4]].all():
                raise ValueError("Registered tilt/yaw sample is stale or invalid")
            if elapsed is not None:
                with torch.device(self.device):
                    batches = (held_input_batches(elapsed) if self.integration_mode ==
                               "held_input_batch" else [(dt, 1)
                                for dt in integration_intervals(elapsed)])
                    for dt, count in batches:
                        held = self.retina_held[:, None, None].expand(-1, count, -1, -1)
                        self.visual_state = self.network.simulate(held, dt=dt,
                            initial_state=self.visual_state, as_states=True)[-1]
                        self.visual_ticks += count
                # Same prior-input hold and fractional final tick as sealed flight replay.
                nominal = self.flight.spec["dynamics"]["dt"]
                count = int(elapsed // nominal)
                intervals = [nominal] * count
                remainder = elapsed - count * nominal
                if remainder > 1e-12:
                    intervals.append(remainder)
                for dt in intervals:
                    self.flight_state = self.flight._advance_elapsed_tensor(
                        self.flight_held, self.flight_state, dt)
                    self.flight_ticks += 1
            integrated = time.perf_counter()
            if camera["capture_id"] != self.order.capture:
                self.retina = np.stack([self.projector.sample(pixels[name]) for name in VIEW_ORDER])
            else:
                self.reused_captures += 1
            retinal = torch.tensor(self.retina, device=self.device)
            motion = self.visual_state.nodes.activity[:, self.motion_indices]
            scene = torch.cat([retinal[:, None], motion], dim=1)[None]
            features = self.flight.scene_pool(scene)
            values = torch.tensor(np.array(body.values, dtype=np.float32)[None], device=self.device)
            valid = torch.tensor(np.array(body.valid)[None], device=self.device)
            drive = self.flight._drive_tensor(self.visual_zero, self.visual_valid, values, valid,
                                               self.goals, features)
            intent = self.flight._read_state_tensor(self.flight_state)[0][0].cpu().numpy().copy()
            if (not np.isfinite(intent).all() or not torch.isfinite(motion).all()
                    or not torch.isfinite(features).all()):
                raise ValueError("Nonfinite coupled neural response")
            self.flight_held, self.retina_held = drive, retinal
            self.last_motion, self.last_features, self.last_body = motion, features, body
            self.order.commit(clock, camera, signature)
            self.steps += 1
            finished = time.perf_counter()
            return dict(intent=intent.tolist(), decision_clock=clock,
                capture_id=camera["capture_id"],
                generation=camera["generation"], steps=self.steps,
                processing_seconds=finished - started, integration_seconds=integrated - started,
                output_host_clock=finished, visual_ticks=self.visual_ticks,
                flight_ticks=self.flight_ticks, recurrent_state_reset=False,
                visual_integration_mode=self.integration_mode,
                motion_six_valid=False, actor_qualified=False, actuator_messages_sent=0)
        except Exception:
            self.failed = True
            raise
