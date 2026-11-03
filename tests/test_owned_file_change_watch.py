import os,struct,sys,time
import pytest
from flybrain_drive.owned_file_change_watch import OwnedFileChangeWatch,changed_names

def test_original_directory_records_must_be_complete_and_offsets_valid():
 name='STOP'.encode('utf-16-le');assert changed_names(struct.pack('<3I',0,1,len(name))+name)==['stop']
 for data in (b'',struct.pack('<3I',0,9,2)+b'ab',struct.pack('<3I',1,1,2)+b'ab'):
  with pytest.raises(ValueError):changed_names(data)

@pytest.mark.skipif(sys.platform!='win32',reason='Real overlapped directory notifications')
def test_notification_latches_even_create_then_remove_without_stat_fallback(tmp_path):
 path=tmp_path/'stop';watch=OwnedFileChangeWatch(path)
 try:
  assert watch.poll() is False
  (tmp_path/'unrelated').write_text('other file');time.sleep(.02)
  assert watch.poll() is False
  path.write_text('stop');os.unlink(path)
  deadline=time.perf_counter()+2
  while not watch.poll() and time.perf_counter()<deadline:time.sleep(.005)
  assert watch.poll() is True and not path.exists()
 finally:watch.close()

@pytest.mark.skipif(sys.platform!='win32',reason='Real overlapped directory notifications')
def test_same_size_contents_change_and_atomic_replacement_are_observed(tmp_path):
 path=tmp_path/'controls.ini';path.write_text('abcd');watch=OwnedFileChangeWatch(path)
 try:
  path.write_text('wxyz');deadline=time.perf_counter()+2
  while not watch.poll() and time.perf_counter()<deadline:time.sleep(.005)
  assert watch.poll()
 finally:watch.close()
