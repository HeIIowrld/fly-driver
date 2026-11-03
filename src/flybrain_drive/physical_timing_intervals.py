"""Conservative interval comparison for prospective sensor/teacher timing.

This module never certifies clocks, creates labels, or grants training admission.
Inputs must already be independently justified bounds in the same host clock.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeBound:
    lower: float
    upper: float
    clock: str
    evidence_id: str

    def __post_init__(self):
        if (type(self.lower) not in (int,float) or type(self.upper) not in (int,float)
                or not math.isfinite(self.lower) or not math.isfinite(self.upper)
                or not 0<=self.lower<=self.upper or self.clock!='host_qpc_seconds'
                or not isinstance(self.evidence_id,str) or not self.evidence_id):
            raise ValueError('Finite ordered host-QPC interval and evidence identity required')


def compare_physical_timing(*, decision, views, body, action, maximum_view_age, maximum_body_age,
                            minimum_action_delay, maximum_action_delay):
    """Compare worst-case intervals without assuming midpoints or independence.

    None means the relevant physical event is unbounded/unverified, not zero delay.
    ``consistent`` is a numerical comparison only and does not authenticate evidence
    or grant fit permission. Actor inputs must never include teacher action or pose.
    """
    if (not isinstance(decision,TimeBound) or not isinstance(views,tuple) or len(views)!=4
            or any(v is not None and not isinstance(v,TimeBound) for v in (*views,body,action))
            or not all(type(x) in (float,int) and math.isfinite(x) for x in
                (maximum_view_age,maximum_body_age,minimum_action_delay,maximum_action_delay))
            or not 0<maximum_view_age<=.15 or not 0<maximum_body_age<=.05
            or not 0<=minimum_action_delay<=maximum_action_delay<=.2):
        raise ValueError('Four views and explicit bounded timing policy required')
    missing=[name for name,value in zip(('front','left','right','rear','body','action'),(*views,body,action),strict=True) if value is None]
    base=dict(clock='host_qpc_seconds',missing_physical_bounds=missing,
              evidence_authenticated=False,training_admissible=False)
    if missing:return dict(base,status='unresolved_missing_physical_bounds')
    sources=(*views,body)
    ages=[(decision.lower-x.upper,decision.upper-x.lower) for x in sources]
    action_delay=(action.lower-decision.upper,action.upper-decision.lower)
    if any(low<0 or high>limit for (low,high),limit in zip(ages,(maximum_view_age,)*4+(maximum_body_age,),strict=True)):
        return dict(base,status='rejected_sensor_interval',sensor_age_intervals=ages,action_delay_interval=action_delay)
    if action_delay[0]<minimum_action_delay or action_delay[1]>maximum_action_delay:
        return dict(base,status='rejected_action_interval',sensor_age_intervals=ages,action_delay_interval=action_delay)
    return dict(base,status='consistent_bounded_timing_only',sensor_age_intervals=ages,action_delay_interval=action_delay)
