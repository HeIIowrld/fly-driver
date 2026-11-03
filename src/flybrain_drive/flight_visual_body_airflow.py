"""Experimental joint visual, vehicle-body and relative-airflow flight circuit.

Existing pooled four-view FlyVis features and body masks are preserved. Airflow is an
engineering sensory-port modulation; this module neither estimates wind nor actuates a car.
"""

import math

import numpy as np
import torch
from torch import nn

from .flight_airflow_feedback import RelativeAirflowAdapter
from .flight_body_feedback_adapter import (
    CenteredVehicleBodyFeedbackAdapter,
    VehicleBodyFeedbackAdapter,
)
from .flight_embodied_navigation import EmbodiedVisualFlightCircuit, modulate_current
from .relative_airflow_sensor import AirflowObservation


class ExperimentalVisualBodyAirflow(nn.Module):
    """Joint inference boundary; no substitute for physiological or driving validation."""

    def __init__(self, circuit, *, centered_availability=False):
        super().__init__()
        if not isinstance(circuit, EmbodiedVisualFlightCircuit):
            raise ValueError('Exact embodied visual flight circuit required')
        self.circuit = circuit
        if type(centered_availability) is not bool:
            raise ValueError('Explicit availability encoding required')
        adapter = (CenteredVehicleBodyFeedbackAdapter if centered_availability
                   else VehicleBodyFeedbackAdapter)
        self.body_feedback = adapter(len(circuit.backbone.body_ports))
        self.airflow = RelativeAirflowAdapter(len(circuit.backbone.body_ports))

    def initial_state(self, batch):
        return self.circuit.initial_state(batch)

    def drive_observed(self, visual, visual_valid, body, body_valid, goals, scene_features,
                       scene_valid, airflow_observations, decision_times):
        """Consume airflow only at its evaluated decision time, never as a held valid sample.

        Receipt freshness is not proof of physical exposure or source synchronization.
        Each new decision needs a fresh call to relative_airflow with the original receipts.
        """
        batch = len(body)
        if (batch < 1 or len(airflow_observations) != batch or len(decision_times) != batch
                or body.dtype != self.airflow.projection.weight.dtype
                or body.device != self.airflow.projection.weight.device):
            raise ValueError('One airflow observation and decision time per body sample required')
        values, masks = [], []
        for observation, decision in zip(airflow_observations, decision_times, strict=True):
            if (not isinstance(observation, AirflowObservation)
                    or not np.isfinite(decision) or decision < 0
                    or decision != observation.decision_time
                    or type(observation.source_verified) is not bool):
                raise ValueError('Airflow must be evaluated for this exact decision')
            value, mask = observation.velocity_body_m_s, observation.valid
            if (not isinstance(value, np.ndarray) or value.shape != (3,)
                    or value.dtype != np.float32 or not np.isfinite(value).all()
                    or not isinstance(mask, np.ndarray) or mask.shape != (3,)
                    or mask.dtype != np.bool_ or mask.any() and not observation.source_verified):
                raise ValueError('Finite body airflow with honest source validity required')
            values.append(value.copy())
            masks.append(mask.copy())
        airflow = torch.as_tensor(np.stack(values), dtype=body.dtype, device=body.device)
        airflow_valid = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=body.device)
        return self.drive(visual, visual_valid, body, body_valid, goals, scene_features,
                          scene_valid, airflow, airflow_valid)

    def drive(self, visual, visual_valid, body, body_valid, goals, scene_features,
              scene_valid, airflow, airflow_valid):
        batch = len(body)
        if (scene_features.shape != (batch, 576) or scene_valid.shape != (batch, 4)
                or scene_valid.dtype != torch.bool or not scene_valid.all()
                or not torch.isfinite(scene_features).all() or len(airflow) != batch):
            raise ValueError('Four valid original view features and matching batch required')
        drive = self.circuit._drive_tensor(visual, visual_valid, body, body_valid,
                                           goals, scene_features)
        ports = self.circuit.backbone.body_ports
        body_delta = self.body_feedback(body, body_valid)
        airflow_delta = self.airflow(airflow, airflow_valid)
        updated = drive.clone()
        body_drive = modulate_current(drive[:, ports], body_delta)
        updated[:, ports] = modulate_current(body_drive, airflow_delta)
        return updated

    def advance(self, drive, state, elapsed):
        if (not math.isfinite(elapsed) or not 0 < elapsed <= .08
                or state.ndim != 2 or state.shape[1] != self.circuit.backbone.neurons
                or drive.shape != state.shape):
            raise ValueError('Finite original drive/state and bounded elapsed time required')
        # Aggregate the same GPU predicates before transferring their scalar result
        # to the host. This does not skip any validity condition or clip state.
        valid = (torch.isfinite(state).all() & torch.isfinite(drive).all()
                 & (~((state < 0) | (state > 1)).any()))
        if not valid:
            raise ValueError('Finite original drive/state and bounded elapsed time required')
        self.circuit.backbone.assert_body_boundary_protected()
        nominal = self.circuit.backbone.spec['dynamics']['dt']
        count = int(math.floor(elapsed/nominal))
        for _ in range(count):
            state = self.circuit._advance_elapsed_tensor(drive, state, nominal)
        tail = elapsed-count*nominal
        if tail > 1e-12:
            state = self.circuit._advance_elapsed_tensor(drive, state, tail)
        return self.circuit._read_state_tensor(state), state
