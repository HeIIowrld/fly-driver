import pytest
from test_native_flight_body_ticker import make
from flybrain_drive.current_flight_control_ticker import CurrentFlightControlTicker


def ticker():
    old,runtime,rows,clock,visual,context=make()
    t=CurrentFlightControlTicker(runtime,run_id='owned',generation='g',checkpoint_sha256='a'*64,
        native_rows=rows,native_identity=old.encoder.native_identity if hasattr(old.encoder,'native_identity') else dict(car_model='bmw_z4_gt3',shared_memory_version='1.7',ac_version='1.16.4'),
        episode_id='one',clock=lambda:clock[0])
    return t,runtime,rows,clock,visual,context


def test_delivery_never_calls_inference_again_and_retains_source_clocks():
    t,runtime,rows,clock,visual,context=ticker()
    t.publish_source(visual,context,uppers=(.992,.993,.994,.995),first_copy_starts=(.98,)*4)
    t.step_once();value=t.latest_after(0)
    assert len(runtime.calls)==1 and value.body_first_receipt==.999
    assert value.publication_id==1 and value.view_render_lowers==visual.render_start_lowers
    assert t.latest_after(1) is None and len(runtime.calls)==1
    clock[0]=1.06
    with pytest.raises(ValueError):t.latest_after(0)
    assert len(runtime.calls)==1


def test_failed_ticker_never_delivers_buffered_result():
    t,*rest=ticker();runtime,rows,clock,visual,context=rest
    t.publish_source(visual,context,uppers=(.992,.993,.994,.995),first_copy_starts=(.98,)*4)
    t.step_once();t.error={'reason':'source failed'}
    with pytest.raises(RuntimeError):t.latest_after(0)


def test_unqualified_publication_rejected_before_inference():
    t,runtime,rows,clock,visual,context=ticker()
    with pytest.raises(ValueError):t.publish_source(visual,context,uppers=(.992,.993,.994,.995),first_copy_starts=(.9,)*4)
    with pytest.raises(RuntimeError):t.publish(visual,context)
    assert not runtime.calls


def test_mailbox_reads_and_new_publication_do_not_wait_for_inference_or_rebind_old_output():
    import threading
    from dataclasses import replace
    from test_native_flight_body_ticker import row
    t,runtime,rows,clock,visual,context=ticker()
    t.publish_source(visual,context,uppers=(.992,.993,.994,.995),first_copy_starts=(.98,)*4)
    t.step_once();clock[0]=1.02;rows.append(row(3,1.019))
    entered=threading.Event();finish=threading.Event();original=runtime.decide;errors=[]
    def blocking(**kw):
        entered.set();assert finish.wait(1);return original(**kw)
    runtime.decide=blocking
    def infer():
        try:t.step_once()
        except BaseException as e:errors.append(e)
    thread=threading.Thread(target=infer);thread.start();assert entered.wait(1)
    read_done=threading.Event();read_values=[]
    def read():read_values.append(t.latest_after(0));read_done.set()
    read_thread=threading.Thread(target=read);read_thread.start()
    try:
        assert read_done.wait(.2), 'Mailbox must not wait for blocked inference'
        assert read_values[0].decision_id==1
        t.publish_source(replace(visual,capture_id=2),context,uppers=(.992,.993,.994,.995),first_copy_starts=(.98,)*4)
    finally:finish.set();thread.join(1);read_thread.join(1)
    assert not errors and not thread.is_alive()
    value=t.latest_after(1)
    assert value.decision_id==2 and value.capture_id==1 and value.publication_id==1
    assert t.latest_visual[0].capture_id==2


def test_pending_scene_handover_avoids_obsolete_extra_step_without_reset_or_clock_refresh():
    from dataclasses import replace
    from test_native_flight_body_ticker import row
    from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
    from flybrain_drive.temporal_body_context import per_view_temporal_body_context
    import numpy as np
    t,runtime,rows,clock,visual,context=ticker()
    old_bounds=(.9,.901,.902,.903)
    old_history=NavigationBodyObservation('one',0,.89,.91,np.zeros(21),np.ones(21,dtype=bool),t.encoder.contract_sha256)
    old_context=per_view_temporal_body_context((old_history,)*4,context.decision_body,render_start_lowers=old_bounds,camera_receipt_time=.91,decision_time=1.)
    old_visual=replace(visual,render_start_lowers=old_bounds,camera_received=.91)
    t.publish_source(old_visual,old_context,uppers=(.901,.902,.903,.904),first_copy_starts=(.89,)*4);t.step_once()
    clock[0]=1.02;rows.append(row(3,1.019));new_bounds=(1.01,1.011,1.012,1.013)
    t.announce_next_visual(capture_id=2,bounds=new_bounds,camera_received=1.015,context_id=10)
    assert t.step_once() is None and len(runtime.calls)==1
    assert runtime.model.previous_body_receipt==.999
    assert t.wait_events[-1]['reason']=='pending_new_scene_handover'
    clock[0]=1.021
    history=NavigationBodyObservation('one',1,1.,1.015,np.zeros(21),np.ones(21,dtype=bool),t.encoder.contract_sha256)
    current=replace(context.decision_body,packet_id=3,original_received_monotonic_seconds=1.019,decision_monotonic_seconds=1.021)
    new_context=per_view_temporal_body_context((history,)*4,current,render_start_lowers=new_bounds,camera_receipt_time=1.015,decision_time=1.021)
    t.publish_source(replace(visual,capture_id=2,render_start_lowers=new_bounds,camera_received=1.015),new_context,uppers=(1.011,1.012,1.013,1.014),first_copy_starts=(1.,)*4)
    t.step_once();value=t.latest_after(1)
    assert len(runtime.calls)==2 and value.capture_id==2 and value.decision_id==2
    assert value.body_first_receipt==1.019 and value.view_render_lowers==new_bounds


def test_announcement_never_makes_expired_visual_admissible():
    t,runtime,rows,clock,visual,context=ticker()
    t.publish_source(visual,context,uppers=(.992,.993,.994,.995),first_copy_starts=(.98,)*4);t.step_once()
    clock[0]=1.16
    with pytest.raises(ValueError):t.announce_next_visual(capture_id=2,bounds=visual.render_start_lowers,camera_received=.995,context_id=3)
    assert len(runtime.calls)==1


def test_pending_announcement_never_extends_the_original150ms_expiry():
    from test_native_flight_body_ticker import row
    t,runtime,rows,clock,visual,context=ticker()
    t.publish_source(visual,context,uppers=(.992,.993,.994,.995),first_copy_starts=(.98,)*4);t.step_once()
    clock[0]=1.14;rows.append(row(3,1.139))
    t.announce_next_visual(capture_id=2,bounds=(1.13,)*4,camera_received=1.135,context_id=3)
    assert t.step_once() is None
    clock[0]=1.142;rows.append(row(4,1.141))
    with pytest.raises(ValueError,match='View render bound stale'):t.step_once()
    assert len(runtime.calls)==1 and runtime.model.previous_body_receipt==.999


def test_deferred_thread_yields_instead_of_spinning_and_starving_source_and_publication(monkeypatch):
    import threading
    from flybrain_drive import current_flight_control_ticker as module
    t,*_=ticker();seen=[]
    def step():t.stop_event.set();return None
    t.step_once=step
    monkeypatch.setattr(module.time,'sleep',lambda delay:seen.append(delay))
    t.start();t.stop()
    assert seen==[.001]

def test_pending_scene_reserves_time_for_both_old_and_new_inference():
    from dataclasses import replace
    from test_native_flight_body_ticker import row
    from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
    from flybrain_drive.temporal_body_context import per_view_temporal_body_context
    import numpy as np
    t,runtime,rows,clock,visual,context=ticker()
    bounds=(.9,.901,.902,.903)
    history=NavigationBodyObservation('one',0,.89,.91,np.zeros(21),np.ones(21,dtype=bool),t.encoder.contract_sha256)
    old_context=per_view_temporal_body_context((history,)*4,context.decision_body,render_start_lowers=bounds,camera_receipt_time=.91,decision_time=1.)
    t.publish_source(replace(visual,render_start_lowers=bounds,camera_received=.91),old_context,uppers=(.901,.902,.903,.904),first_copy_starts=(.89,)*4);t.step_once()
    clock[0]=1.014;rows.append(row(3,1.013))
    t.announce_next_visual(capture_id=2,bounds=(1.005,1.006,1.007,1.008),camera_received=1.009,context_id=10)
    assert t.step_once() is None
    assert len(runtime.calls)==1 and runtime.model.previous_body_receipt==.999
    assert t.wait_events[-1]['reason']=='pending_new_scene_handover'

def test_pending_wait_never_spends_the_held_body50ms_budget():
    from dataclasses import replace
    from test_native_flight_body_ticker import row
    from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
    from flybrain_drive.temporal_body_context import per_view_temporal_body_context
    import numpy as np
    t,runtime,rows,clock,visual,context=ticker()
    bounds=(.93,.931,.932,.933)
    history=NavigationBodyObservation('one',0,.92,.94,np.zeros(21),np.ones(21,dtype=bool),t.encoder.contract_sha256)
    old_context=per_view_temporal_body_context((history,)*4,context.decision_body,render_start_lowers=bounds,camera_receipt_time=.94,decision_time=1.)
    t.publish_source(replace(visual,render_start_lowers=bounds,camera_received=.94),old_context,uppers=(.931,.932,.933,.934),first_copy_starts=(.92,)*4);t.step_once()
    clock[0]=1.03;rows.append(row(3,1.029))
    t.announce_next_visual(capture_id=2,bounds=(1.02,1.021,1.022,1.023),camera_received=1.025,context_id=10)
    # Keep the actual sensory integration advancing while old pixels remain valid.
    assert t.step_once() is not None
    assert len(runtime.calls)==2 and runtime.model.previous_body_receipt==1.029
