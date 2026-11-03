import ctypes,json,sys,uuid,faulthandler
from pathlib import Path
import pytest
from flybrain_drive.ac_metadata_mmf import MetadataMMF,OwnedAckMapping,decode_packet,SIZE

@pytest.mark.skipif(sys.platform!='win32',reason='Original Windows named maps')
def test_separate_stop_map_is_owned_once_and_never_rewrites_frame_ack():
    rid='mmf-stream-'+uuid.uuid4().hex
    channel=MetadataMMF(rid,'generation-a',clock=lambda:123.,stop_signal=True)
    try:
        untouched=channel.ack.snapshot()
        with pytest.raises(FileExistsError):OwnedAckMapping(rid,direction='stop')
        first=channel.request_stop();original=channel.stop.snapshot()
        assert decode_packet(original,original,0)==first
        assert channel.request_stop()==first and channel.stop.snapshot()==original
        assert channel.ack.snapshot()==untouched
        assert json.loads(first)==dict(schema='flybrain.camera.owned_stop.v1',request_id=rid,generation='generation-a',stop=True,issued_host_qpc=123.)
    finally:channel.close()

def test_actual_luajit_stop_checks_checksum_owner_generation_and_explicit_stop():
    module=pytest.importorskip('lupa.luajit21')
    lua=module.LuaRuntime(unpack_returned_tuples=True,encoding=None)
    import hashlib
    lua.globals().hash_python=lambda b:hashlib.sha256(b).hexdigest().encode()
    lua.globals().decode_python=lambda b:lua.table_from({k.encode():v.encode() if isinstance(v,str) else v for k,v in json.loads(b).items()})
    lua.execute(b"ac={checksumSHA256=function(s) return hash_python(s) end}; JSON={parse=function(s) return decode_python(s) end}; generation='generation-a'; timingCall=function(stage,fn,...) return fn(...) end")
    source=Path('integrations/ac_csp/candidates/FlyBrainCamera-source-body-witness-v6-live-mmf.lua').read_text(encoding='utf8')
    block=source[source.index('local mmfLayout ='):source.index('local function closeActive()')]
    lua.execute((block+"\nlocal ffi=require('ffi'); slot=ffi.new('struct { '..mmfLayout..' }[1]'); owner={request={id='mmf-stream-"+'a'*32+"'},mmfStop=slot[0]}; import=function(s) ffi.copy(slot,s,#s) end; check=function() return cameraStopRequested(owner) end").encode())
    from flybrain_drive.ac_metadata_mmf import encode_packet
    assert lua.globals().check() is False
    payload=dict(schema='flybrain.camera.owned_stop.v1',request_id='mmf-stream-'+'a'*32,generation='generation-a',stop=True,issued_host_qpc=12.)
    for field,bad in [('request_id','foreign'),('generation','foreign'),('stop',False)]:
        value=dict(payload);value[field]=bad
        lua.globals()[b'import'](encode_packet(0,json.dumps(value).encode()))
        restore=sys.platform=='win32' and faulthandler.is_enabled()
        if restore:faulthandler.disable() # Expected caught LuaJIT assertion, matching existing FFI tests.
        try:
            with pytest.raises(module.LuaError,match='identity'):lua.globals().check()
        finally:
            if restore:faulthandler.enable()
    lua.globals()[b'import'](encode_packet(0,json.dumps(payload).encode()))
    assert lua.globals().check() is True
