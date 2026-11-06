"""A stalled caller must not keep drive applied in another process."""
import multiprocessing as mp
import time
from dataclasses import replace
from test_current_flight_control_contract import IDENTITY,envelope
from test_current_flight_actuation import Session
from flybrain_drive.current_flight_actuation import FlightActuationLease


def owned_watchdog(pipe):
    session=Session();lease=FlightActuationLease(session,identity=IDENTITY,clock=time.perf_counter)
    now=time.perf_counter();base=envelope();shift=now-1.02
    decision=replace(base,body_first_receipt=base.body_first_receipt+shift,
        view_render_lowers=tuple(v+shift for v in base.view_render_lowers),
        view_render_uppers=tuple(v+shift for v in base.view_render_uppers),
        view_body_first_copy_starts=tuple(v+shift for v in base.view_body_first_copy_starts),
        camera_received=base.camera_received+shift,published_at=base.published_at+shift,
        decision_started=base.decision_started+shift,inference_finished=base.inference_finished+shift)
    lease.arm(prepared_since=now-.6,valid_camera_groups=3);lease.apply(decision)
    pipe.send('driving')
    while lease.state=='armed':lease.poll();time.sleep(.001)
    lease.release(stationary_feedback_verified=False)
    pipe.send(dict(requests=session.requests,reason=lease.stop_reason,released=not session.owned))
    pipe.close()


def test_child_brakes_and_releases_while_caller_does_not_send_any_more_messages():
    context=mp.get_context('spawn');parent,child=context.Pipe();process=context.Process(target=owned_watchdog,args=(child,))
    process.start();child.close()
    try:
        assert parent.poll(10) and parent.recv()=='driving'
        assert parent.poll(1)
        result=parent.recv()
        assert result['reason']=='source_lease_expired' and result['released']
        assert result['requests']==[(-.3,.2,.1),(0.,0.,1.),(0.,0.,1.)]
        process.join(2);assert process.exitcode==0
    finally:
        parent.close()
        if process.is_alive():process.terminate();process.join(2)
