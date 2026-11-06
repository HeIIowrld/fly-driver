"""Read-only startup observation after one owned resume, before any thermal setter.

Nonstationary startup samples are preserved and never admitted to an actor.
Actual thermal admission still requires the original stationary native contract.
"""
import math
import time
from .native_initialization_readiness import NativeInitializationReadiness


def wait_live_stationary(request,before,initial_packet,*,read_state,read_native,poll_terminal,
                         check_ownership,observations,clock=time.perf_counter,sleep=time.sleep,initialized_thermal=False):
    if type(initialized_thermal) is not bool:raise ValueError('Explicit thermal epoch mode required')
    readiness=NativeInitializationReadiness();deadline=clock()+5.;terminal=None;recent={}
    while clock()<deadline:
        heartbeat=read_state();sample=read_native()
        if terminal is None:terminal=poll_terminal()
        state=heartbeat
        if terminal is not None:
            if terminal.get('id')!=request['id'] or terminal.get('command')!='resume' or terminal.get('status')!='api_returned_true':
                raise ValueError('Owned initial resume failed or mismatched; never retry')
            after=terminal.get('after')
            if after is not None and heartbeat['script_clock']<after['script_clock']:state=after
        record=dict(heartbeat=heartbeat,resume_receipt_state_used=state is not heartbeat,
                    admitted_state=state,native=sample.authority_record(),actor_admitted=False)
        observations.append(record)
        check_ownership(heartbeat)
        for current in (heartbeat,state):
            expected=dict(is_online=False,in_menu=False,ai_controlled=False,camera_active=False,
                          expert_probe_active=False,brake_settle_active=False,physics_api_allowed=True)
            if (any(current.get(k) is not v for k,v in expected.items())
                    or current.get('generation')!=before['generation'] or current.get('epoch')!=before['epoch']
                    or current.get('car_id')!='bmw_z4_gt3' or current.get('track_id')!='monza'
                    or current.get('error') or current.get('pending_receipt_id') not in (None,request['id'])
                    or (current.get('tyres_initialization_epoch')==before['epoch'])!=initialized_thermal):
                raise ValueError('Owned offline initialization context changed')
            values=[current.get(k) for k in ('speed_kmh','gas','brake')]
            if (not all(type(x) in (int,float) and math.isfinite(x) for x in values)
                    or abs(values[0])>5 or not 0<=values[1]<.02 or not .98<values[2]<=1):
                raise ValueError('Startup observation exceeded brake-held5km/h preparation envelope')
        fields=sample.physics.fields
        if (sample.builtin_ai_controlled is not False or abs(fields['speedKmh'])>5
                or not 0<=fields['gas']<.02 or not .98<fields['brake']<=1):
            raise ValueError('Native startup authority or brake-held envelope changed')
        if sample.physics.packet_id<initial_packet:raise ValueError('Native packet reset after owned resume')
        ready=False
        if terminal is not None and state.get('paused') is False and state.get('live') is True and sample.physics.packet_id>initial_packet:
            ready=readiness.observe(sample)
            record['ready_native']=readiness.records[-1]['ready_native']
            if record['ready_native']:recent.setdefault(sample.physics.packet_id,sample.authority_record())
        else:record['ready_native']=False
        # The ordinary source heartbeat must also catch up before a thermal request.
        if (ready and heartbeat.get('paused') is False and heartbeat.get('live') is True
                and not heartbeat.get('pending_receipt_id') and abs(heartbeat['speed_kmh'])<.1):
            witnesses=list(recent.values())[-3:]
            if len(witnesses)!=3:raise ValueError('Distinct startup witnesses missing')
            return heartbeat,sample,witnesses,terminal
        sleep(.01)
    raise TimeoutError('Owned resume did not yield three stationary native packets within5s; no thermal call')
