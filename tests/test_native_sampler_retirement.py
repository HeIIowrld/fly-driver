import base64,struct,time
import pytest
from flybrain_drive.native_process_sampler import NativeProcessSampler
from flybrain_drive.native_process_rows import NativeProcessRowsSampler


def raw():
    b=bytearray(420);struct.pack_into('<i',b,0,7);struct.pack_into('<f',b,28,36.)
    return bytes(b)


def test_stop_drains_original_unread_records_without_admitting_them():
    payload=raw();source=NativeProcessSampler(recorded=(payload,),period=.0001,max_samples=96,capacity=128,drain_on_stop=True)
    source.start();first=source.poll(timeout=3);assert first is not None
    time.sleep(.35)
    admitted=source.expected;source.stop(timeout=5)
    assert not source.process.is_alive() and source.process.exitcode==0
    assert admitted==source.expected==1
    assert source.retired_rows
    assert [r['sequence'] for r in source.retired_rows]==list(range(admitted,source.terminal[1]))
    assert all(not r['actor_admitted'] and base64.b64decode(r['row']['authority']['prefix_base64'])==payload for r in source.retired_rows)


def test_rows_adapter_stops_both_pump_and_source_with_retirement_enabled():
    sampler=NativeProcessRowsSampler(recorded=(raw(),),period=.002,maximum_samples=10000,drain_on_stop=True)
    sampler.start();time.sleep(.03);sampler.stop()
    assert not sampler.thread.is_alive() and not sampler.source.process.is_alive()
    assert len(sampler.rows)+len(sampler.source.retired_rows)==sampler.source.terminal[1]


def test_retirement_option_is_explicit_boolean():
    with pytest.raises(ValueError):NativeProcessSampler(recorded=(raw(),),drain_on_stop='yes')

def test_rejected_stale_record_is_preserved_without_actor_admission():
    payload=raw();source=NativeProcessSampler(recorded=(payload,),period=.01,max_samples=20,capacity=128,drain_on_stop=True)
    source.start()
    with pytest.raises(RuntimeError,match='stale or future'):source.poll(timeout=3,clock=lambda:time.perf_counter()+1)
    source.stop()
    assert not source.process.is_alive() and source.expected==0
    assert len(source.rejected_rows)==1 and source.rejected_rows[0]['sequence']==0
    assert source.rejected_rows[0]['actor_admitted'] is False
    assert base64.b64decode(source.rejected_rows[0]['row']['authority']['prefix_base64'])==payload
