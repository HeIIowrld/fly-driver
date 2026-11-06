"""Read-only readiness before any thermal setter; startup samples are never actor inputs."""
from .initial_thermal_contract import validate_thermal_native

class NativeInitializationReadiness:
    def __init__(self):self.last_packet=None;self.consecutive=0;self.records=[]
    def observe(self,coupled):
        sample=coupled.physics
        if coupled.builtin_ai_controlled:raise ValueError('Native AI active during initialization readiness')
        if self.last_packet is not None and sample.packet_id<self.last_packet:raise ValueError('Native initialization packet reset')
        valid=True
        try:validate_thermal_native(coupled)
        except ValueError:valid=False
        self.records.append(dict(packet=sample.packet_id,receipt=sample.received_monotonic_seconds,
            ready_native=valid,authority=coupled.authority_record(),actor_admitted=False))
        if not valid:self.consecutive=0
        elif sample.packet_id!=self.last_packet:self.consecutive+=1
        self.last_packet=sample.packet_id
        return self.consecutive>=3
