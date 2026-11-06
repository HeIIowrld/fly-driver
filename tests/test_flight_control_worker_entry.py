import json,os,subprocess,sys
import pytest

@pytest.mark.parametrize('fail',[False,True])
def test_owned_worker_persists_finally_and_no_children_before_native_exit(tmp_path,fail):
    target=tmp_path/'flight_worker_fixture.py';sentinel=tmp_path/'cleanup.txt';receipt=tmp_path/'receipt.json'
    lines=['from pathlib import Path','try:',"    raise ValueError('preflight fixture')" if fail else '    pass','finally:',"    Path("+repr(str(sentinel))+").write_text('closed')"]
    target.write_text(chr(10).join(lines),encoding='utf8')
    env=os.environ.copy();env['PYTHONPATH']=str(tmp_path)+os.pathsep+env.get('PYTHONPATH','')
    result=subprocess.run([sys.executable,'-m','scripts.flight_control_worker_entry','--module','flight_worker_fixture','--worker-receipt',str(receipt)],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==(1 if fail else 0)
    assert sentinel.read_text()=='closed'
    value=json.loads(receipt.read_text())
    assert value['active_owned_children']==[] and value['target_returned_after_its_finally']


@pytest.mark.skipif(os.name=='nt',reason='CPython Windows has no POSIX resource tracker')
def test_nonactuating_resource_tracker_can_exit_on_parent_eof(tmp_path):
    target=tmp_path/'resource_tracker_fixture.py';receipt=tmp_path/'receipt.json'
    target.write_text("from multiprocessing.resource_tracker import ensure_running; ensure_running()",encoding='utf8')
    env=os.environ.copy();env['PYTHONPATH']=str(tmp_path)+os.pathsep+env.get('PYTHONPATH','')
    result=subprocess.run([sys.executable,'-m','scripts.flight_control_worker_entry','--module','resource_tracker_fixture','--worker-receipt',str(receipt)],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0
    value=json.loads(receipt.read_text())
    assert value['active_owned_children']==[] and len(value['nonactuating_parent_eof_helpers'])==1
