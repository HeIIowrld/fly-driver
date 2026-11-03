import time

from flybrain_drive.native_process_rows import NativeProcessRowsSampler
from flybrain_drive.online_body_shadow import OnlineBodyShadow
from test_native_process_sampler import packet


IDENTITY = dict(car_model='bmw_z4_gt3', shared_memory_version='1.7', ac_version='1.16.4')


def test_process_rows_are_consumable_by_existing_shadow():
    sampler = NativeProcessRowsSampler(recorded=tuple(packet(i) for i in range(1, 50)),
                                       period=.005, maximum_samples=49)
    sampler.start()
    try:
        deadline = time.perf_counter()+2
        while len(sampler.rows) < 4 and time.perf_counter() < deadline:
            assert sampler.error is None
            time.sleep(.002)
        assert len(sampler.rows) >= 4
        shadow = OnlineBodyShadow(directory='.', request_id='test', generation=1,
                                  offset_lower=0., native_rows=sampler.rows,
                                  native_identity=IDENTITY)
        shadow._drain_native()
        assert shadow.position >= 4
        assert sampler.rows[0]['copy']['synthetic_replay'] is True
    finally:
        sampler.stop()
    assert not sampler.thread.is_alive()
    assert not sampler.source.process.is_alive()
    assert sampler.error is None
