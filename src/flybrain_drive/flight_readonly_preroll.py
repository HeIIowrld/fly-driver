"""Read-only sensory preparation before retaining a continuous flight candidate.

This engineering gate checks exposure, not driving competence or qualification.
The underlying runtime is never reset, advanced synthetically, or given actuators.
"""
import math


class ReadOnlyFlightPreroll:
    def __init__(self,runtime,*,minimum_seconds,minimum_captures=3):
        if (isinstance(minimum_seconds,bool) or not math.isfinite(minimum_seconds)
                or not 0<minimum_seconds<=2 or type(minimum_captures) is not int or minimum_captures<2):
            raise ValueError('Explicit bounded sensory duration and multiple camera captures required')
        if runtime.episode is None or runtime.last_time is not None or runtime.closed or runtime.poisoned:
            raise ValueError('Fresh explicitly started read-only runtime required')
        self.runtime=runtime;self.episode=runtime.episode;self.minimum_seconds=minimum_seconds;self.minimum_captures=minimum_captures
        self.first=None;self.last=None;self.capture_ids=set();self.latest_capture=None;self.failed=False;self.events=[]

    def decide(self,*,capture_id,**inputs):
        if self.failed:raise RuntimeError('Rejected preparation requires an explicit new run')
        try:
            if (self.runtime.episode!=self.episode or self.runtime.last_time!=self.last
                    or type(capture_id) is not int or capture_id<0
                    or self.latest_capture is not None and capture_id<self.latest_capture):
                raise ValueError('Continuous owned episode, recurrence and camera sequence required')
            output=self.runtime.decide(**inputs)
            if output['actor_qualified'] is not False or output['vehicle_control_sent'] is not False:
                raise ValueError('Read-only unqualified output required')
            now=inputs['decision_time']
            if self.first is None:self.first=now
            self.last=now;self.latest_capture=capture_id;self.capture_ids.add(capture_id)
            elapsed=now-self.first
            prepared=elapsed>=self.minimum_seconds and len(self.capture_ids)>=self.minimum_captures
            record=dict(decision_time=now,capture_id=capture_id,integrated_sensor_seconds=elapsed,distinct_captures=len(self.capture_ids),
                prepared=prepared,actor_qualified=False,vehicle_control_sent=False)
            self.events.append(record)
            return dict(output=output,preparation=record,continuation_candidate=output['prediction'] if prepared else None)
        except Exception:
            self.failed=True
            raise
