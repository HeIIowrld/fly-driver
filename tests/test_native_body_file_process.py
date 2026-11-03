import json
import struct
import time

from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.native_body_file_process import NativeBodyFileProcess


def test_native_file_process_archives_without_a_camera_process_data_queue(tmp_path):
    raw=bytearray(AI_STATUS_PREFIX.size);struct.pack_into('<i',raw,0,7)
    path=tmp_path/'body.jsonl'
    writer=NativeBodyFileProcess(path,maximum_samples=300,recorded=(bytes(raw),))
    time.sleep(.08)
    writer.check()
    result=writer.close()
    assert result['status']=='completed' and result['samples']>=8
    assert not result['forced_termination'] and not result['process_alive']
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert result['samples']==len(rows)==result['archive']['accepted']==result['archive']['written']
    assert [r['sequence'] for r in rows]==list(range(len(rows)))
    assert all(r['copy']['synthetic_replay'] for r in rows)


def test_native_file_process_preserves_source_failure_in_terminal_receipt(tmp_path):
    raw=bytearray(AI_STATUS_PREFIX.size);struct.pack_into('<i',raw,0,7)
    struct.pack_into('<i',raw,AI_STATUS_PREFIX.size-4,1)
    writer=NativeBodyFileProcess(tmp_path/'body.jsonl',maximum_samples=300,recorded=(bytes(raw),))
    writer.process.join(2)
    result=writer.close()
    assert result['status']=='failed' and result['samples']==0
    assert not result['process_alive'] and not result['forced_termination']
    assert 'authority' in result['reason']
