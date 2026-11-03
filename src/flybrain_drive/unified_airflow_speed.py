"""Research composition of existing speed and airflow boundaries in one state."""
import torch
from .unified_flight_sensorimotor import UnifiedFlightSensorimotor

@torch.no_grad()
def recurrent_speed_airflow_decision(model, *, decision_time, racing_goals,
                                     scene_features, scene_valid, body_context, airflow):
    """No independent state/controller or actuator; diagnostics only."""
    if type(model) is not UnifiedFlightSensorimotor or model.airflow_feedback is None or model.speed_command_boundary is None:
        raise ValueError('Same unified model with explicit speed and airflow boundaries required')
    command=model._recurrent_speed_command(scene_features,body_context,racing_goals)
    return model._decision(decision_time=decision_time,command=command,racing_goals=racing_goals,
        scene_features=scene_features,scene_valid=scene_valid,body_context=body_context,
        retain_graph=False,airflow=airflow)
