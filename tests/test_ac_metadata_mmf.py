import ctypes
import faulthandler
import hashlib
import struct
import sys
import uuid
from pathlib import Path

import pytest

from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_metadata_mmf import (
    HEADER,
    SIZE,
    OwnedAckMapping,
    channel_name,
    decode_packet,
    encode_packet,
)


def test_torn_odd_old_and_skipped_publications_are_distinguished():
    packet = encode_packet(1, b'{"index":1}')
    assert decode_packet(packet, packet, 1) == b'{"index":1}'
    assert decode_packet(packet, packet, 2) is None
    assert decode_packet(packet, bytes(SIZE), 1) is None
    odd = bytearray(packet)
    struct.pack_into('<I', odd, 8, 3)
    assert decode_packet(bytes(odd), bytes(odd), 1) is None
    with pytest.raises(ValueError, match='skipped'):
        decode_packet(packet, packet, 0)


def test_stable_corruption_is_not_silently_treated_as_a_missing_frame():
    corrupt = bytearray(encode_packet(0, b'{"index":0}'))
    corrupt[HEADER.size+2] ^= 1
    with pytest.raises(ValueError, match='checksum'):
        decode_packet(bytes(corrupt), bytes(corrupt), 0)


@pytest.mark.skipif(sys.platform != 'win32', reason='Real Windows named mappings')
def test_ack_ownership_collision_never_clears_original_and_readonly_peer_receives():
    request = 'mmf-probe-'+uuid.uuid4().hex
    owner = OwnedAckMapping(request)
    reader = None
    try:
        owner.publish(0, b'owned exact ACK bytes')
        reader = ExistingNamedMapping(channel_name(request, 'ack'), SIZE)
        original = reader.read()
        with pytest.raises(FileExistsError):
            OwnedAckMapping(request)
        assert reader.read() == original
        assert decode_packet(original, reader.read(), 0) == b'owned exact ACK bytes'
        with pytest.raises(ValueError, match='once'):
            owner.publish(0, b'replay')
        assert reader.read() == original
    finally:
        if reader:
            reader.close()
        owner.close()
        owner.close()


def test_actual_luajit_ffi_and_python_packet_layout_interoperate_in_both_directions():
    lua_module = pytest.importorskip('lupa.luajit21')
    lua = lua_module.LuaRuntime(unpack_returned_tuples=True, encoding=None)
    lua.globals().hash_python = lambda b: hashlib.sha256(b).hexdigest().encode()
    lua.execute(b'ac={checksumSHA256=function(text) return hash_python(text) end}')
    source = Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua').read_text('utf8')
    block = source.split('-- BEGIN owned MMF metadata protocol:')[1].split(
        '-- END owned MMF metadata protocol.')[0]
    # The first split line is explanatory comment text.
    block = '--'+block
    lua.execute((block+'''
      local ffi = require('ffi')
      slot = ffi.new('struct { '..mmfLayout..' }[1]')
      publish = function(index, text) mmfPublish(slot[0], index, text) end
      consume = function(index) return mmfRead(slot[0], index) end
      snapshot = function() return ffi.string(slot, ffi.sizeof(slot)) end
      import = function(text) ffi.copy(slot, text, #text) end
    ''').encode())
    lua.globals().publish(0, b'{"original":true}')
    packet = lua.globals().snapshot()
    assert len(packet) == SIZE
    assert decode_packet(packet, packet, 0) == b'{"original":true}'
    lua.globals()[b'import'](encode_packet(1, b'{"capture_id":1}'))
    assert lua.globals().consume(1) == b'{"capture_id":1}'
    assert lua.globals().consume(2) is None
    corrupt = bytearray(encode_packet(2, b'checksum failure'))
    corrupt[HEADER.size] ^= 1
    lua.globals()[b'import'](bytes(corrupt))
    restore = sys.platform == 'win32' and faulthandler.is_enabled()
    if restore:
        faulthandler.disable()
    try:
        with pytest.raises(lua_module.LuaError, match='checksum'):
            lua.globals().consume(2)
    finally:
        if restore:
            faulthandler.enable()


def test_channel_names_cannot_target_native_ac_or_an_arbitrary_user_map():
    with pytest.raises(ValueError):
        channel_name('Local\\acpmf_physics', 'ack')
    with pytest.raises(ValueError):
        channel_name('mmf-probe-'+uuid.uuid4().hex, 'native')
    assert ctypes.sizeof(ctypes.c_uint32) == 4
