"""Guard episode thermal preparation and independent four-wheel readback."""
import math

THERMAL_COMMAND='set-tyres-temperature-initial'
VERIFY_THERMAL_COMMAND='verify-initial-tyres-temperature'


def validate_thermal_state(state, temperature_c, *, initialization, live_initialization=False):
    if type(temperature_c) not in (int,float) or not math.isfinite(temperature_c) or not 20<=temperature_c<=120:
        raise ValueError('Finite initialization temperature20..120 Celsius required')
    expected=dict(is_online=False,in_menu=False,ai_controlled=False,camera_active=False,
                  expert_probe_active=False,brake_settle_active=False,physics_api_allowed=True)
    if initialization:
        expected['paused']=not live_initialization
        if live_initialization:expected['live']=True
    if (any(state.get(k) is not v for k,v in expected.items())
            or state.get('car_id')!='bmw_z4_gt3' or state.get('track_id')!='monza'
            or state.get('error') or state.get('pending_receipt_id')
            or state.get('capabilities',{}).get('initial_tyre_temperature_v1') is not True):
        raise ValueError('Idle offline BMW/Monza thermal API state required')
    if initialization and live_initialization and state.get('capabilities',{}).get('initial_tyre_temperature_live_v1') is not True:
        raise ValueError('Explicit live initialization capability required')
    values=[state.get(k) for k in ['speed_kmh','gas','brake']]
    if (not all(type(x) in (int,float) and math.isfinite(x) for x in values)
            or abs(values[0])>=.1 or not 0<=values[1]<.02 or not .98<values[2]<=1):
        raise ValueError('Stationary zero-gas full-brake thermal preparation required')
    if initialization and state.get('tyres_initialization_epoch')==state['epoch']:
        raise ValueError('This session already consumed its one thermal initialization')


def validate_thermal_native(coupled):
    fields=coupled.physics.fields
    if (coupled.builtin_ai_controlled is not False or abs(fields['speedKmh'])>=.1
            or not 0<=fields['gas']<.02 or not .98<fields['brake']<=1):
        raise ValueError('Independent same-copy native AI-off stationary brake hold required')


class ThermalReadback:
    """Three distinct increasing post-publication packets, each four wheels±2C.

    Receipt clock is a host observation bound, not native sample/exposure time.
    Paused duplicate packets cannot manufacture evidence by being read again.
    """
    def __init__(self,target,*,after_host_clock,initial_packet):
        if (type(target) not in (int,float) or not math.isfinite(target) or not 20<=target<=120
                or not math.isfinite(after_host_clock) or after_host_clock<0
                or type(initial_packet) is not int or initial_packet<0):
            raise ValueError('Pinned target, publication clock and original packet required')
        self.target,self.after,self.last=target,after_host_clock,initial_packet
        self.consecutive=0;self.observations=[]

    def observe(self,coupled):
        validate_thermal_native(coupled)
        sample=coupled.physics
        if sample.received_monotonic_seconds<=self.after:
            raise ValueError('Readback predates owned publication')
        if sample.packet_id<self.last:
            raise ValueError('Native sequence reset; initialization generation no longer established')
        if sample.packet_id==self.last:return False
        values=list(sample.fields['tyreCoreTemperature'])
        if len(values)!=4 or not all(math.isfinite(v) for v in values):
            raise ValueError('Four finite original native core temperatures required')
        matched=all(abs(v-self.target)<=2. for v in values)
        self.last=sample.packet_id
        self.consecutive=self.consecutive+1 if matched else 0
        self.observations.append(dict(packet_id=sample.packet_id,
            host_receipt=sample.received_monotonic_seconds,temperature_c=values,
            all_four_within_two_c=matched,same_copy_authority=coupled.authority_record()))
        return self.consecutive>=3
