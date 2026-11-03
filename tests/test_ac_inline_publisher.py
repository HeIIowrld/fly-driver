import json
import time
from test_ac_tcp import load_app
from test_ac_latest_tcp import wait_packet
from flybrain_drive.ac_latest_tcp import LatestTcpTelemetryClient


def test_actual_loopback_source_delivery_is_nonblocking_and_receipts_are_original():
    app=load_app();producer=app.InlineLoopbackPublisher(port=0)
    client=LatestTcpTelemetryClient(port=producer.port,timeout=1)
    try:
        deadline=time.perf_counter()+1
        while producer.connection is None and time.perf_counter()<deadline:time.sleep(.001)
        assert producer.connection is not None and producer.connection.gettimeout()==0
        producer.publish({'speed':2.})
        wait_packet(client,1);first=client.receive()
        producer.publish({'speed':3.})
        wait_packet(client,2);second=client.receive()
        assert first.sequence==1 and second.sequence==2
        assert first.raw['publisher_tcp_delivery']=='inline_nonblocking_loopback_v1'
        assert first.received_monotonic_seconds<=second.received_monotonic_seconds
        assert producer.delivery_failures==0
    finally:client.close();producer.close()
    assert not producer.thread.is_alive() and producer.connection is None


def test_partial_nonblocking_send_closes_framing_instead_of_repairing_data():
    app=load_app();producer=app.InlineLoopbackPublisher(port=0)
    class ShortSocket:
        closed=False
        def send(self,wire):return len(wire)-1
        def close(self):self.closed=True
    sock=ShortSocket()
    try:
        producer.connection=sock
        producer.publish({'speed':2.})
        assert sock.closed and producer.connection is None and producer.delivery_failures==1
        wire=json.loads(producer.latest[1]);stamp=wire['timestamp_ac']
        assert stamp>0 and producer.counter==1
    finally:producer.close()
