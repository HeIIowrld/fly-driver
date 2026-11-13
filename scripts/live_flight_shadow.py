"""Offline-only output codec and live diagnostic state for the pinned flight graph."""

import json
import math
import os
import time
from pathlib import Path

import torch

from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter, ExperimentalBodyFeedbackCircuit
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.local_racing_line import maneuver_from_line
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder, decode_native_extension
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix
from plan_frozen_corner_archive import BODY_CONTRACT


CONTRACTS = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)


class LiveFlightShadow:
    """Stateful flight circuit diagnostics. Owns no actuator interface."""

    def __init__(self, identity, *, preview_horizon_s=.08, sparse_backend="scipy_csr_compact_cpu"):
        if not math.isfinite(preview_horizon_s) or not 0 < preview_horizon_s <= .08:
            raise ValueError("Flight shadow preview must be positive and at most 80 ms")
        self.preview_horizon_s = preview_horizon_s
        with torch.device("cpu"):
            self._initialize(identity, sparse_backend)

    def _initialize(self, identity, sparse_backend):
        torch.set_num_threads(1)
        spec = Path("configs/flight-circuit-separated-v5-20261029.json")
        checkpoint = Path("runs/flight-separated-command-probe-20261029-t1081/checkpoint.pt")
        adapter = Path(os.environ["LOCALAPPDATA"])/"flybrain/runtime-runs/20261101-speed-wheel-turn-control-fit/speed-body-adapter.pt"
        extension_root = Path("runs/thermal-driving-20261103")
        extension = extension_root/"native-fit-02/body-extension.pt"
        contract = extension_root/"extension-data-01/contract.json"
        receipt = Path("runs/vision-line-native-port-20261110-01/result.json")
        expected = json.loads(receipt.read_text(encoding="utf8"))["hashes"]
        for path in (spec, checkpoint, adapter, extension):
            if sha256(path) != expected[str(path)]:
                raise ValueError(f"Changed frozen flight input {path}")
        motor = PhysicalManeuverCircuit(spec, torch.load(checkpoint, map_location="cpu", weights_only=True),
                                        sparse_backend=sparse_backend)
        self.graph = ExperimentalBodyFeedbackCircuit(motor)
        ports = len(motor.backbone.body_ports)
        self.graph.adapter = CenteredSpeedBodyFeedbackAdapter(ports)
        self.graph.adapter.load_state_dict(torch.load(adapter, map_location="cpu", weights_only=True)["model"])
        payload = torch.load(extension, map_location="cpu", weights_only=True)
        if sha256(contract) != payload["extension_contract_sha256"]:
            raise ValueError("Native extension contract differs")
        weights = payload["body_port"]
        self.port = MultisensoryBodyFeedbackPort(weights["reference"],weights["scale"],ports)
        self.port.load_state_dict(weights)
        self.graph.eval().requires_grad_(False)
        self.port.eval().requires_grad_(False)
        self.encoder = NavigationBodyEncoder(BODY_CONTRACT, native_identity=identity)
        self.extension = NativeMultisensoryExtensionEncoder(contract, native_identity=identity)
        self.hashes = {str(p):sha256(p) for p in (spec,checkpoint,adapter,extension,contract,receipt,Path(__file__))}
        self.hook = self.graph.adapter.register_forward_hook(lambda module,args,value: value+self.delta)
        self.reset()
        with torch.no_grad():
            reference_body = torch.zeros((1,21))
            valid = torch.zeros((1,21),dtype=torch.bool)
            valid[:,[4,16]] = True
            state = motor.initial_state(1)
            for _ in range(50):
                _,_,state = motor.advance(torch.zeros((1,2)),reference_body,valid,
                    torch.tensor(racing_goal(90,1)),state,.08,**CONTRACTS)
            self.yaw_offset = float(motor.backbone._read_state_preactivation_tensor(state)[0][0,0])

    def reset(self):
        self.previous = None
        self.state = None

    def _advance(self, command, body, mask, delta, state, elapsed):
        self.delta = delta.repeat(len(command),1)
        count = math.ceil(elapsed/.08)
        for _ in range(count):
            _,_,state = self.graph.advance(command,body.repeat(len(command),1),mask.repeat(len(command),1),
                torch.tensor(racing_goal(90,len(command))),state,elapsed/count,**CONTRACTS)
        return state

    @torch.no_grad()
    def process(self, frame, generation, prediction, snapshot):
        with torch.device("cpu"):
            return self._process(frame, generation, prediction, snapshot)

    def _process(self, frame, generation, prediction, snapshot):
        t = snapshot.received_monotonic_seconds
        raw = snapshot.prefix_bytes
        body = self.encoder.encode(decode_navigation_body_prefix(raw,t), episode_id=generation,
                                   decision_monotonic_seconds=t)
        extra = self.extension.encode(decode_native_extension(raw,t),episode_id=generation,
                                      decision_monotonic_seconds=t)
        if not body.valid[[0,1,4,16]].all() or body.values[16] < 1/3.6:
            self.reset()
            return dict(status="invalid_or_stationary_body", packet=body.packet_id, copy_s=t)
        b = torch.tensor(body.values,dtype=torch.float32)[None]
        mask = torch.tensor(body.valid)[None]
        values = torch.tensor(extra.values,dtype=torch.float32)[None]
        valid = torch.tensor(extra.valid)[None]
        symmetric = values.clone()
        symmetric[:,1] = 0
        for first in (3,7,11,15):
            for wheel in (first,first+2):
                pair = slice(wheel,wheel+2)
                if valid[:,pair].all():
                    symmetric[:,pair] = values[:,pair].mean()
        delta = self.port(values,valid) - self.port(symmetric,valid)
        carry_start = time.perf_counter()
        if self.previous is not None:
            index, episode, stamp, args = self.previous
            elapsed = t-stamp
            if index+1 == frame and episode == generation and 0 < elapsed <= .18:
                self.state = self._advance(*args,self.state,elapsed)
            else:
                self.reset()
        reset = self.state is None
        if reset:
            self.state = self.graph.initial_state(1)
        carry_s = time.perf_counter()-carry_start
        speed = float(body.values[16])
        line = prediction["line"]
        if line is None:
            command = torch.tensor([[0.,speed]])
        else:
            command = torch.as_tensor(maneuver_from_line(line,speed,lookahead_m=12.,
                speed_bound=prediction.get("preview_speed_mps")))[None]
        self.previous = (frame,generation,t,(command,b,mask,delta))
        evidence = dict(packet=body.packet_id, copy_s=t, state_reset=reset,
                        has_path=line is not None, carry_s=carry_s)
        if line is None:
            return dict(evidence,status="no_path")
        neutral = command.clone()
        neutral[:,1] = speed
        forecast_start = time.perf_counter()
        branch = self._advance(torch.cat((command,neutral)),b,mask,delta,self.state.repeat(2,1),
                               self.preview_horizon_s)
        preactivation,_ = self.graph.motor.backbone._read_state_preactivation_tensor(branch)
        yaw = float((preactivation[0,0]-self.yaw_offset).tanh())
        drive = float((preactivation[0,1]-preactivation[1,1]).tanh())
        return dict(evidence,status="computed",command=command[0].tolist(),
                    hypothetical_horizon_s=self.preview_horizon_s, forecast_s=time.perf_counter()-forecast_start,
                    axes=canonical_controls((yaw,drive)).tolist())
