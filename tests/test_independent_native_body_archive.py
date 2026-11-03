import json
import struct
import time

from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.independent_native_body_archive import IndependentNativeBodyArchive


def test_owned_native_archive_drains_every_original_copy_before_normal_stop(tmp_path):
    raw=bytearray(AI_STATUS_PREFIX.size);struct.pack_into('<i',raw,0,7)
    path=tmp_path/'body.jsonl'
    writer=IndependentNativeBodyArchive(path,recorded=(bytes(raw),),maximum_samples=300)
    deadline=time.perf_counter()+2
    while writer.samples<12 and time.perf_counter()<deadline:
        writer.check();time.sleep(.005)
    result=writer.close()
    assert result['status']=='completed' and result['samples']>=12
    assert not result['process_alive'] and not result['drain_alive'] and not result['forced_termination']
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows)==result['samples']==result['terminal'][1]
    assert [x['sequence'] for x in rows]==list(range(len(rows)))
    assert all(x['authority']['packet_id']==7 and x['copy']['synthetic_replay'] for x in rows)
    assert all(a['authority']['received_monotonic_seconds']<b['authority']['received_monotonic_seconds']
        for a,b in zip(rows,rows[1:]))


def test_native_authority_error_stays_latched_and_child_is_closed(tmp_path):
    raw=bytearray(AI_STATUS_PREFIX.size);struct.pack_into('<i',raw,0,7)
    struct.pack_into('<i',raw,AI_STATUS_PREFIX.size-4,1)
    writer=IndependentNativeBodyArchive(tmp_path/'body.jsonl',recorded=(bytes(raw),),maximum_samples=300)
    writer.thread.join(2)
    result=writer.close()
    assert result['status']=='failed' and result['samples']==0
    assert not result['process_alive'] and not result['drain_alive']
    assert 'authority' in result['error']
