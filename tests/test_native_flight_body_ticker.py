import base64,hashlib,struct
from types import SimpleNamespace
import numpy as np
import pytest
from flybrain_drive.native_flight_body_ticker import NativeFlightBodyTicker
from flybrain_drive.per_view_flyvis_stream import VisualFlightObservation
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation

IDENTITY=dict(car_model='bmw_z4_gt3',shared_memory_version='1.7',ac_version='1.16.4')


def row(packet,receipt):
    raw=bytearray(420);struct.pack_into('<i',raw,0,packet);struct.pack_into('<f',raw,28,36.)
    struct.pack_into('<4f',raw,72,1000,2000,3000,4000);struct.pack_into('<4f',raw,152,60,61,62,63)
    raw=bytes(raw)
    return dict(authority=dict(prefix_base64=base64.b64encode(raw).decode(),prefix_sha256=hashlib.sha256(raw).hexdigest(),packet_id=packet,received_monotonic_seconds=receipt),copy=dict(second_copy_end=receipt))


class Runtime:
    episode='one'
    def __init__(self):self.model=SimpleNamespace(previous_body_receipt=None);self.calls=[]
    def decide(self,**kw):
        current=kw['body_context'].decision_body
        old=self.model.previous_body_receipt
        if old is not None and kw['decision_time']-old>.05:raise ValueError('held body stale')
        self.calls.append(kw);self.model.previous_body_receipt=current.original_received_monotonic_seconds
        return dict(prediction=np.array([0.,.1,.2]),inference_seconds=0.,thermal_out_of_training_support=(False,)*4,maximum_body_delta=.25)


def make():
    clock=[1.];runtime=Runtime();rows=[row(1,.9),row(2,.999)]
    ticker=NativeFlightBodyTicker(runtime,native_rows=rows,native_identity=IDENTITY,episode_id='one',clock=lambda:clock[0])
    historical=NavigationBodyObservation('one',0,.98,.995,np.zeros(21),np.ones(21,dtype=bool),ticker.encoder.contract_sha256)
    current=NavigationBodyObservation('one',2,.999,1.,np.zeros(21),np.ones(21,dtype=bool),ticker.encoder.contract_sha256)
    bounds=(.991,.992,.993,.994)
    ctx=per_view_temporal_body_context((historical,)*4,current,render_start_lowers=bounds,camera_receipt_time=.995,decision_time=1.)
    visual=VisualFlightObservation('one','g',1,.995,bounds,np.zeros(576,dtype=np.float32),np.zeros(320,dtype=np.float32),('h',)*4)
    return ticker,runtime,rows,clock,visual,ctx


def test_new_body_ticks_keep_same_visual_and_original_historical_bodies():
    ticker,runtime,rows,clock,visual,ctx=make();ticker.publish(visual,ctx);ticker.step_once()
    for i in range(3):
        clock[0]+=.02;rows.append(row(i+3,clock[0]-.001));ticker.step_once()
    assert len(runtime.calls)==4 and {e['capture_id'] for e in ticker.events}=={1}
    assert all(c['body_context'].visual_contexts[0].original_received_monotonic_seconds==.98 for c in runtime.calls)
    assert runtime.calls[-1]['decision_time']-runtime.calls[0]['decision_time']>.05
    assert all(e['output_body_age_seconds']<.002 for e in ticker.events)
    assert all(e['maximum_native_body_delta']==.25 for e in ticker.events)


def test_duplicate_native_packet_keeps_first_receipt_and_not_an_extra_decision():
    ticker,runtime,rows,clock,visual,ctx=make();ticker.publish(visual,ctx);ticker.step_once()
    clock[0]=1.02;rows.append(row(2,1.019));assert ticker.step_once() is None
    assert ticker.latest.legacy.received_monotonic_seconds==.999 and len(runtime.calls)==1


def test_native_loss_rejects_without_resetting_circuit():
    ticker,runtime,rows,clock,visual,ctx=make();ticker.publish(visual,ctx);ticker.step_once()
    clock[0]=1.06
    with pytest.raises(ValueError,match='stale'):ticker.step_once()
    assert runtime.model.previous_body_receipt==.999 and len(runtime.calls)==1


def test_future_native_packet_never_reaches_runtime():
    ticker,runtime,rows,clock,visual,ctx=make();rows.append(row(3,1.001));ticker.publish(visual,ctx)
    with pytest.raises(ValueError):ticker.step_once()
    assert not runtime.calls


def test_duplicate_visual_publication_is_rejected():
    ticker,runtime,rows,clock,visual,ctx=make();ticker.publish(visual,ctx);ticker.step_once()
    with pytest.raises(ValueError,match='advance'):ticker.publish(visual,ctx)


def test_known_source_failure_rejects_even_when_buffered_sample_is_fresh():
    ticker,runtime,rows,clock,visual,ctx=make();ticker.publish(visual,ctx)
    ticker.source_error=lambda:'producer queue failed'
    with pytest.raises(RuntimeError,match='source failed'):ticker.step_once()
    assert not runtime.calls and ticker.position==0


def test_failed_decision_retains_original_computation_and_detached_rejection():
    ticker,runtime,rows,clock,visual,ctx=make();ticker.publish(visual,ctx)
    def fail(**kwargs):
        runtime.last_rejection={'computation':{'input_body_age_seconds':.001,'inference_seconds':.060}}
        raise ValueError('original deadline exceeded')
    runtime.decide=fail
    with pytest.raises(ValueError,match='original deadline'):ticker.step_once()
    event=ticker.events[-1]
    assert event['status']=='rejected' and event['computation_profile']['finished']>=event['computation_profile']['started']
    runtime.last_rejection['computation']['inference_seconds']=123
    assert event['runtime_rejection']['computation']['inference_seconds']==.060


@pytest.mark.parametrize('flush',[False,True])
def test_explicit_numerical_mode_is_isolated_to_owned_cpu_ticker_thread(flush):
    import threading
    import torch
    torch.set_flush_denormal(False)
    tiny=torch.tensor([1],dtype=torch.int32).view(torch.float32)
    assert int((tiny+0).view(torch.int32)[0])==1
    ticker,*_=make();ticker.flush_denormal=flush;seen=[];done=threading.Event()
    def observe():
        seen.append(int((tiny+0).view(torch.int32)[0]))
        ticker.stop_event.set();done.set()
    ticker.step_once=observe
    ticker.start();assert done.wait(1);ticker.stop()
    assert seen==([0] if flush else [1])
    assert int((tiny+0).view(torch.int32)[0])==1
    if flush:assert ticker.numerics['supported'] and ticker.numerics['reset_to_nonflushing_before_thread_exit']
