"""Owned duplex metadata slots; never write native AC memory or producer-owned memory."""

import ctypes
import hashlib
import json
import re
import struct
import time

from .ac_image_buffer import ExistingNamedMapping

MAGIC = 0x46424D31
CAPACITY = 32768
HEADER = struct.Struct('<4I64s')
SIZE = HEADER.size+CAPACITY


def channel_name(request_id, direction):
    if (not re.fullmatch(r'mmf-(?:probe|stream)-[0-9a-f]{32}', request_id)
            or direction not in ('frame', 'ack', 'stop')):
        raise ValueError('Fresh owned GUID and explicit channel direction required')
    return f'Local\\FlyBrainMMF_{direction}_{request_id}'


def encode_packet(index, payload):
    if (type(index) is not int or not 0 <= index < 4000 or not isinstance(payload, bytes)
            or not 0 < len(payload) <= CAPACITY):
        raise ValueError('Bounded exact index and immutable metadata bytes required')
    return (HEADER.pack(MAGIC, 1, 2*(index+1), len(payload),
                        hashlib.sha256(payload).hexdigest().encode('ascii'))
            + payload + bytes(CAPACITY-len(payload)))


def decode_packet(first, second, index):
    if len(first) != SIZE or len(second) != SIZE or type(index) is not int or not 0 <= index < 4000:
        raise ValueError('Exact slot size and bounded expected index required')
    if first != second:
        return None
    magic, version, sequence, length, digest = HEADER.unpack_from(first)
    if sequence == 0 or sequence % 2:
        return None
    if magic != MAGIC or version != 1 or not 0 < length <= CAPACITY:
        raise ValueError('MMF slot header differs')
    if sequence < 2*(index+1):
        return None
    if sequence > 2*(index+1):
        raise ValueError('MMF skipped an unacknowledged frame')
    payload = first[HEADER.size:HEADER.size+length]
    if hashlib.sha256(payload).hexdigest().encode('ascii') != digest:
        raise ValueError('Stable MMF payload checksum differs')
    return payload


class OwnedAckMapping:
    """Create only our unique ACK map; reject an already-existing named object."""

    def __init__(self, request_id, *, direction='ack'):
        from ctypes import wintypes

        self.handle, self.view = None, None
        self.last_index = -1
        if direction not in ('ack','stop'):raise ValueError('Only host-owned ACK or stop channel permitted')
        self.name = channel_name(request_id, direction)
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateFileMappingW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                               wintypes.DWORD, wintypes.DWORD, wintypes.LPCWSTR]
        self.api.CreateFileMappingW.restype = wintypes.HANDLE
        self.api.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.DWORD, ctypes.c_size_t]
        self.api.MapViewOfFile.restype = ctypes.c_void_p
        self.api.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        self.api.UnmapViewOfFile.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        self.handle = self.api.CreateFileMappingW(wintypes.HANDLE(-1), None, 4, 0, SIZE, self.name)
        error = ctypes.get_last_error()
        if not self.handle:
            raise ctypes.WinError(error)
        if error == 183:  # ERROR_ALREADY_EXISTS: never clear a foreign or prior map.
            self.close()
            raise FileExistsError('Owned ACK map already exists')
        self.view = self.api.MapViewOfFile(self.handle, 2 | 4, 0, 0, SIZE)
        if not self.view:
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def publish(self, index, payload):
        if not self.view:
            raise RuntimeError('ACK map is closed')
        if index != self.last_index+1:
            raise ValueError('ACK publication must be once in exact capture order')
        packet = encode_packet(index, payload)
        # Mark odd first, copy immutable bytes while odd, publish even last.
        ctypes.memmove(self.view+8, struct.pack('<I', 2*index+1), 4)
        provisional = bytearray(packet)
        struct.pack_into('<I', provisional, 8, 2*index+1)
        ctypes.memmove(self.view, bytes(provisional), SIZE)
        ctypes.memmove(self.view+8, packet[8:12], 4)
        self.last_index = index

    def close(self):
        if self.view:
            self.api.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None

    def snapshot(self):
        if not self.view:
            raise RuntimeError('ACK map is closed')
        return ctypes.string_at(self.view, SIZE)


class MetadataMMF:
    def __init__(self, request_id, generation, *, clock=time.perf_counter,stop_signal=False):
        self.request_id, self.generation, self.clock = request_id, generation, clock
        self.frame = None
        self.pending_index = None
        self.last_frame_slot, self.last_ack_slot = None, None
        self.last_observed_slot = None
        self.ack = OwnedAckMapping(request_id)
        try:
            if type(stop_signal) is not bool:raise ValueError('Explicit boolean stop channel required')
            self.stop = OwnedAckMapping(request_id,direction='stop') if stop_signal else None
        except BaseException:
            self.ack.close()
            raise
        self.stop_payload = None

    def request_stop(self):
        if self.stop is None:raise ValueError('Explicit owned stop map required')
        if self.stop_payload is None:
            self.stop_payload=json.dumps(dict(schema='flybrain.camera.owned_stop.v1',
                request_id=self.request_id,generation=self.generation,stop=True,
                issued_host_qpc=self.clock()),allow_nan=False,separators=(',',':')).encode('utf8')
            self.stop.publish(0,self.stop_payload)
        return self.stop_payload

    def poll(self, index):
        if self.pending_index is not None and self.pending_index != index:
            raise ValueError('Original pending frame must be acknowledged first')
        started = self.clock()
        if self.frame is None:
            try:
                self.frame = ExistingNamedMapping(channel_name(self.request_id, 'frame'), SIZE)
            except FileNotFoundError:
                return None
        first, second = self.frame.read(), self.frame.read()
        self.last_observed_slot = first
        payload = decode_packet(first, second, index)
        if payload is None:
            return None
        frame = json.loads(payload)
        if (frame['request_id'] != self.request_id or frame['generation'] != self.generation
                or frame['index'] != index
                or frame.get('metadata_transport') != 'mmf_slot_sha256_v1'):
            raise ValueError('MMF original frame identity differs')
        self.pending_index = index
        self.last_frame_slot = first
        return frame, payload, dict(schema='flybrain.mmf.metadata_receipt.v1',
            sequence=2*(index+1), started_at=started, finished_at=self.clock(),
            raw_sha256=hashlib.sha256(payload).hexdigest(), repeated_slot_copies_equal=True,
            producer_slot_modified=False)

    def acknowledge(self, ack):
        if (ack['request_id'] != self.request_id or ack['generation'] != self.generation
                or self.pending_index != ack['capture_id']):
            raise ValueError('ACK identity differs from owned map')
        payload = json.dumps(ack, allow_nan=False, separators=(',', ':')).encode('utf8')
        self.ack.publish(ack['capture_id'], payload)
        self.last_ack_slot = self.ack.snapshot()
        self.pending_index = None
        return payload

    def close(self):
        if self.frame:
            self.frame.close()
            self.frame = None
        self.ack.close()
        if self.stop:self.stop.close()


def verify_mmf_archive_row(camera, row):
    """Verify both actual map snapshots against original JSON, receipt and GPU ownership."""
    from .ac_shared_texture import verify_shared_frame

    index = row['capture_id']
    raw = camera.read(f'frame-{index:04d}.json')
    ack_raw = camera.read(f'ack-{index:04d}.json')
    slots = camera.read(f'mmf-{index:04d}.mmf')
    proof = row['metadata_mmf_receipt']
    if (len(slots) != 2*SIZE or proof['sequence'] != 2*(index+1)
            or decode_packet(slots[:SIZE], slots[:SIZE], index) != raw
            or decode_packet(slots[SIZE:], slots[SIZE:], index) != ack_raw
            or hashlib.sha256(slots).hexdigest() != proof['slots_sha256']
            or hashlib.sha256(raw).hexdigest() != proof['raw_sha256']
            or hashlib.sha256(raw).hexdigest() != row['file_sha256']
            or proof['producer_slot_modified'] is not False
            or proof['repeated_slot_copies_equal'] is not True
            or row['metadata_transport'] != 'mmf_slot_sha256_v1'):
        raise ValueError('Original MMF slots/metadata/receipt differ')
    frame, ack = json.loads(raw), json.loads(ack_raw)
    if frame.get('metadata_transport') != row['metadata_transport']:
        raise ValueError('Producer metadata transport differs')
    if not proof['started_at'] <= proof['finished_at'] <= min(
            c['started_at'] for c in row['shared_copy_receipts']):
        raise ValueError('MMF metadata must actually precede every GPU copy')
    verify_shared_frame(frame, row, ack)
