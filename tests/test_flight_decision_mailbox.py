from dataclasses import replace
import time,struct,multiprocessing as mp
import pytest
from flybrain_drive.flight_decision_mailbox import FlightDecisionMailbox,HEADER
from test_current_flight_control_contract import envelope,IDENTITY
from flybrain_drive.current_flight_actuation import FlightActuationLease

def test_mailbox_original_envelope_once_and_terminal():
    m=FlightDecisionMailbox();r=FlightDecisionMailbox(m.spec)
    try:
        assert r.read() is None;m.publish(envelope());assert r.read().record()==envelope().record();assert r.read(1) is None
        m.seal()
        with pytest.raises(RuntimeError,match='terminal'):r.read()
    finally:r.close();m.close();m.unlink()

def test_corrupt_mailbox_is_not_repaired_or_restamped():
    m=FlightDecisionMailbox()
    try:
        m.publish(envelope());m.memory.buf[HEADER.size]^=1
        with pytest.raises(ValueError,match='digest'):m.read()
    finally:m.close();m.unlink()

def test_writer_in_progress_never_returns_partial_decision():
    m=FlightDecisionMailbox()
    try:
        m.publish(envelope());m.memory.buf[:8]=struct.pack('<Q',3);assert m.read() is None
    finally:m.close();m.unlink()

class FakeSdkSession:
    def __init__(self):self.calls=[]
    def submit(self,*args,**kw):self.calls.append((time.perf_counter(),args,kw));return dict(driver_accepted=True)
    def close(self):return None

def _produce(spec,ready):
    m=FlightDecisionMailbox(spec);ready.wait(5);end=time.perf_counter()+.35;i=0
    while time.perf_counter()<end:
        now=time.perf_counter();i+=1
        v=replace(envelope(),decision_id=i,body_first_receipt=now-.002,view_render_lowers=(now-.012,)*4,view_render_uppers=(now-.011,)*4,view_body_first_copy_starts=(now-.015,)*4,camera_received=now-.009,published_at=now-.006,decision_started=now-.001,inference_finished=now)
        m.publish(v);time.sleep(.01)
    m.close()

def _consume(spec,ready,pipe):
    m=FlightDecisionMailbox(spec);session=FakeSdkSession();lease=FlightActuationLease(session,identity=IDENTITY,clock=time.perf_counter,maximum_seconds=1,maximum_decisions=100);ready.set();end=time.perf_counter()+5
    while time.perf_counter()<end:
        v=m.read()
        if v is not None:break
        time.sleep(.001)
    lease.arm(prepared_since=time.perf_counter()-.6,valid_camera_groups=3);last=0
    while lease.state=='armed':
        v=m.read(last)
        if v is not None:lease.apply(v);last=v.decision_id
        lease.poll();time.sleep(.001)
    lease.release(stationary_feedback_verified=True);pipe.send(dict(accepted=last,reason=lease.stop_reason,events=lease.events));pipe.close();m.close()

def test_real_producer_consumer_keep_actuating_while_gpu_caller_is_stalled_then_expire():
    m=FlightDecisionMailbox();ctx=mp.get_context('spawn');ready=ctx.Event();a,b=ctx.Pipe(False);consumer=ctx.Process(target=_consume,args=(m.spec,ready,b));producer=ctx.Process(target=_produce,args=(m.spec,ready))
    try:
        consumer.start();producer.start();b.close();time.sleep(.5) # No main/GPU-side IPC servicing.
        assert a.poll(15);r=a.recv();producer.join(5);consumer.join(5)
        assert producer.exitcode==consumer.exitcode==0 and r['accepted']>=10
        assert r['reason']=='source_lease_expired'
        assert len([e for e in r['events'] if e['kind']=='student_request'])==r['accepted']
    finally:
        for p in [producer,consumer]:
            if p.is_alive():p.terminate();p.join(5)
        a.close();m.close();m.unlink()

def test_stop_flag_independent_of_rpc_and_no_torch_import():
    m=FlightDecisionMailbox();r=FlightDecisionMailbox(m.spec)
    try:m.request_stop();assert r.stop_requested
    finally:r.close();m.close();m.unlink()


def test_failed_native_guard_precedes_every_mailbox_submission():
    from flybrain_drive.flight_decision_mailbox import consume_mailbox_step
    m=FlightDecisionMailbox();s=FakeSdkSession();clock=lambda:1.02
    try:
        m.publish(envelope());lease=FlightActuationLease(s,identity=IDENTITY,clock=clock);lease.arm(prepared_since=.4,valid_camera_groups=3)
        def invalid(now):raise ValueError('Native gear changed')
        with pytest.raises(ValueError,match='gear'):consume_mailbox_step(lease,m,invalid,1.35)
        assert not s.calls and not any(v['kind']=='student_request' for v in lease.events)
    finally:m.close();m.unlink()


def test_old_source_expiry_is_not_reported_as_planned_success():
    from flybrain_drive.flight_decision_mailbox import consume_mailbox_step
    m=FlightDecisionMailbox();s=FakeSdkSession();now=[1.02]
    try:
        lease=FlightActuationLease(s,identity=IDENTITY,clock=lambda:now[0]);lease.arm(prepared_since=.4,valid_camera_groups=3);m.publish(envelope());consume_mailbox_step(lease,m,lambda now:None,1.35)
        now[0]=1.36;consume_mailbox_step(lease,m,lambda now:None,1.35)
        assert lease.stop_reason=='source_lease_expired' and lease.last_decision==1
    finally:m.close();m.unlink()
