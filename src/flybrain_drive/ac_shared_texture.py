"""Read a shared D3D11 texture into our own staging buffer; never writes the source texture."""

import ctypes as ct
import re
import sys
import time
import uuid

import numpy as np


# QueryThreadCycleTime counts scheduled cycles, not elapsed time or a fixed-rate clock.
# Keep these raw; CPU frequency and processor migration prevent seconds conversion.
if sys.platform == 'win32':
    _kernel32_cycles = ct.WinDLL('kernel32', use_last_error=True)
    _kernel32_cycles.GetCurrentThread.argtypes = []
    _kernel32_cycles.GetCurrentThread.restype = ct.c_void_p
    _kernel32_cycles.QueryThreadCycleTime.argtypes = [ct.c_void_p, ct.POINTER(ct.c_ulonglong)]
    _kernel32_cycles.QueryThreadCycleTime.restype = ct.c_int
else:
    _kernel32_cycles = None


def current_thread_cycles():
    if _kernel32_cycles is None:
        return None
    value = ct.c_ulonglong()
    if not _kernel32_cycles.QueryThreadCycleTime(_kernel32_cycles.GetCurrentThread(), ct.byref(value)):
        raise ct.WinError(ct.get_last_error())
    return value.value


class TextureDescription(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        'Width', 'Height', 'MipLevels', 'ArraySize', 'Format', 'SampleCount', 'SampleQuality',
        'Usage', 'BindFlags', 'CPUAccessFlags', 'MiscFlags')]


class MappedResource(ct.Structure):
    _fields_ = [('pData', ct.c_void_p), ('RowPitch', ct.c_uint32), ('DepthPitch', ct.c_uint32)]


class AdapterDescription(ct.Structure):
    _fields_ = [('Description', ct.c_wchar*128), ('VendorId', ct.c_uint32),
        ('DeviceId', ct.c_uint32), ('SubSysId', ct.c_uint32), ('Revision', ct.c_uint32),
        ('DedicatedVideoMemory', ct.c_size_t), ('DedicatedSystemMemory', ct.c_size_t),
        ('SharedSystemMemory', ct.c_size_t), ('LuidLow', ct.c_uint32),
        ('LuidHigh', ct.c_int32), ('Flags', ct.c_uint32)]



def reviewed_adapter_index(adapters):
    """DXGI ordinal is transient; require one reviewed physical GPU identity."""
    matches = [a for a in adapters if (a.get('name'), a.get('vendor_id'), a.get('device_id'))
               == ('AMD Radeon RX 9070 XT', 4098, 30032)]
    if len(matches) != 1:
        raise ValueError('Exactly one reviewed RX9070XT hardware identity required')
    index = matches[0]['index']
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('Bounded enumerated DXGI index required')
    return index


def parse_shared_handle(value):
    if isinstance(value, bool) or not re.fullmatch(r'[0-9]+(?:ULL)?', str(value)):
        raise ValueError('Explicit original unsigned shared texture handle required')
    result = int(str(value).removesuffix('ULL'))
    if not 0 < result < 2**64:
        raise ValueError('Shared texture handle outside64bit range')
    return result


def unpack_rgba_rows(raw, row_pitch, format_code):
    if format_code not in (28, 29, 87, 91) or not 1280 <= row_pitch <= 16384 or (
            len(raw) != row_pitch*180):
        raise ValueError('Exact320x180 RGBA8/BGRA8 staging layout required')
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(180, row_pitch)
    rgba = rows[:, :1280].reshape(180, 320, 4)
    return rgba[:, :, [2, 1, 0] if format_code in (87, 91) else [0, 1, 2]].copy()


def verify_shared_frame(frame, row, ack):
    """Verify original source identity, copy interval and matching release acknowledgment."""
    if frame.get('transport') != 'shared_immutable_v1':
        if row.get('shared_copy_receipts'):
            raise ValueError('Unexpected shared copy receipts for a file producer')
        return
    if (ack['request_id'] != frame['request_id'] or ack['generation'] != frame['generation']
            or ack['capture_id'] != frame['index'] or row['capture_id'] != frame['index']
            or row['generation'] != frame['generation']
            or not np.isfinite([row['received'], ack['copied_before_ack']]).all()):
        raise ValueError('Shared frame acknowledgment identity or causal receipt differs')
    source = {v['view_id']: v for v in frame['views']}
    lower = {v['view_id']: v['render_started_lower'] for v in row['views']}
    copies = row.get('shared_copy_receipts', [])
    if len(copies) != 4 or {c['view_id'] for c in copies} != {'front', 'left', 'right', 'rear'}:
        raise ValueError('Four independent original shared texture copies required')
    for copied in copies:
        view = source[copied['view_id']]
        desc = copied['description']
        if (view['image_writer'] != 'host_lossless_bmp'
                or parse_shared_handle(view['shared_texture_handle']) != copied['original_handle']
                or copied['source_texture_modified'] is not False
                or (desc['Width'], desc['Height'], desc['MipLevels'], desc['ArraySize'],
                    desc['SampleCount']) != (320, 180, 1, 1, 1)
                or desc['Format'] not in (28, 29, 87, 91)
                or not np.isfinite([copied['started_at'], copied['finished_at']]).all()
                or not lower[copied['view_id']] <= copied['started_at']
                    <= copied['finished_at'] <= min(row['received'], ack['copied_before_ack'])):
            raise ValueError('Shared texture identity/layout or read-before-ack differs')


class SharedTextureReader:
    """Own D3D11 device, imported source references and CPU-readable destination textures."""

    def __init__(self, *, adapter_index=None):
        if sys.platform != 'win32':
            raise OSError('Windows D3D11 required')
        self.device, self.context = ct.c_void_p(), ct.c_void_p()
        self.textures = {}
        self.adapter = None
        self.dll = ct.WinDLL('d3d11')
        create = self.dll.D3D11CreateDevice
        create.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_uint32,
            ct.POINTER(ct.c_uint32), ct.c_uint32, ct.c_uint32,
            ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_void_p)]
        create.restype = ct.c_int32
        level = ct.c_uint32()
        adapter = ct.c_void_p()
        try:
            if adapter_index is not None:
                adapter, self.adapter = self._adapter(adapter_index)
            self._checked(create(adapter, 0 if adapter else 1, None, 0, None, 0, 7,
                ct.byref(self.device), ct.byref(level), ct.byref(self.context)),
                'D3D11CreateDevice')
        except BaseException:
            self.close()
            raise
        finally:
            self._release(adapter)
        self.feature_level = int(level.value)

    @classmethod
    def reviewed(cls):
        """Enumerate without opening image handles; fail on absence, ambiguity or reorder."""
        probe, adapters = cls.__new__(cls), []
        for index in range(8):
            try:
                pointer, description = probe._adapter(index)
            except OSError as error:
                if str(error) != 'EnumAdapters1: HRESULT0x887a0002':
                    raise
                break
            try:
                adapters.append(description)
            finally:
                probe._release(pointer)
        else:
            raise ValueError('DXGI inventory exceeds reviewed eight-adapter bound')
        index = reviewed_adapter_index(adapters)
        reader = cls(adapter_index=index)
        if reader.adapter != next(a for a in adapters if a['index'] == index):
            reader.close()
            raise ValueError('DXGI adapter identity changed during device creation')
        return reader

    def _adapter(self, index):
        if type(index) is not int or not 0 <= index < 8:
            raise ValueError('Explicit bounded DXGI adapter index0..7 required')
        factory, adapter = ct.c_void_p(), ct.c_void_p()
        iid = (ct.c_ubyte*16).from_buffer_copy(
            uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
        create = ct.WinDLL('dxgi').CreateDXGIFactory1
        create.argtypes, create.restype = [ct.c_void_p, ct.POINTER(ct.c_void_p)], ct.c_int32
        self._checked(create(ct.byref(iid), ct.byref(factory)), 'CreateDXGIFactory1')
        try:
            self._checked(self._method(factory, 12, ct.c_int32, ct.c_uint32,
                ct.POINTER(ct.c_void_p))(factory, index, ct.byref(adapter)), 'EnumAdapters1')
            description = AdapterDescription()
            self._checked(self._method(adapter, 10, ct.c_int32, ct.POINTER(AdapterDescription))(
                adapter, ct.byref(description)), 'GetDesc1')
            return adapter, dict(index=index, name=description.Description,
                vendor_id=int(description.VendorId), device_id=int(description.DeviceId),
                dedicated_video_bytes=int(description.DedicatedVideoMemory),
                luid_low=int(description.LuidLow), luid_high=int(description.LuidHigh))
        except BaseException:
            self._release(adapter)
            raise
        finally:
            self._release(factory)

    @staticmethod
    def _checked(result, operation):
        if result < 0:
            raise OSError(f'{operation}: HRESULT0x{result & 0xffffffff:08x}')

    @staticmethod
    def _method(pointer, index, result, *arguments):
        if not pointer:
            raise ValueError('Closed COM interface')
        table = ct.cast(pointer, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
        return ct.WINFUNCTYPE(result, ct.c_void_p, *arguments)(table[index])

    def _release(self, pointer):
        if pointer:
            self._method(pointer, 2, ct.c_uint32)(pointer)

    def _open(self, handle):
        if handle in self.textures:
            return self.textures[handle]
        if len(self.textures) >= 8:
            old_source, old_staging, _ = self.textures.pop(next(iter(self.textures)))
            self._release(old_staging)
            self._release(old_source)
        source, staging = ct.c_void_p(), ct.c_void_p()
        iid = (ct.c_ubyte*16).from_buffer_copy(
            uuid.UUID('6f15aaf2-d208-4e89-9ab4-489535d34f9c').bytes_le)
        try:
            opened = self._method(self.device, 28, ct.c_int32,
                ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p))(
                    self.device, ct.c_void_p(handle), ct.byref(iid), ct.byref(source))
            self._checked(opened, 'OpenSharedResource')
            description = TextureDescription()
            self._method(source, 10, None, ct.POINTER(TextureDescription))(
                source, ct.byref(description))
            if ((description.Width, description.Height, description.MipLevels,
                    description.ArraySize, description.SampleCount) != (320, 180, 1, 1, 1)
                    or description.Format not in (28, 29, 87, 91)):
                raise ValueError('Shared texture is not the registered320x180 RGBA8 camera')
            original = {name: int(getattr(description, name)) for name, _ in description._fields_}
            description.Usage, description.BindFlags = 3, 0  # D3D11_USAGE_STAGING
            description.CPUAccessFlags, description.MiscFlags = 0x20000, 0  # CPU READ only
            self._checked(self._method(self.device, 5, ct.c_int32,
                ct.POINTER(TextureDescription), ct.c_void_p, ct.POINTER(ct.c_void_p))(
                    self.device, ct.byref(description), None, ct.byref(staging)), 'CreateTexture2D')
            self.textures[handle] = (source, staging, original)
            return self.textures[handle]
        except BaseException:
            self._release(staging)
            self._release(source)
            raise

    def read(self, handle, *, timeout=.04):
        if not 0 < timeout <= .1:
            raise ValueError('Explicit bounded nonblocking GPU map deadline required')
        handle = parse_shared_handle(handle)
        started = time.perf_counter()
        source, staging, description = self._open(handle)
        # Destination is always our CPU-readable texture; imported game texture is only a source.
        self._method(self.context, 47, None, ct.c_void_p, ct.c_void_p)(
            self.context, staging, source)
        self._method(self.context, 111, None)(self.context)  # submit our own context
        mapped = MappedResource()
        deadline = started+timeout
        while True:
            status = self._method(self.context, 14, ct.c_int32, ct.c_void_p, ct.c_uint32,
                ct.c_uint32, ct.c_uint32, ct.POINTER(MappedResource))(
                    self.context, staging, 0, 1, 0x100000, ct.byref(mapped))
            if status & 0xffffffff != 0x887a000a:  # DXGI_ERROR_WAS_STILL_DRAWING
                self._checked(status, 'Map READ DO_NOT_WAIT')
                break
            if time.perf_counter() >= deadline:
                raise TimeoutError('Shared texture GPU copy did not complete before deadline')
            time.sleep(.0005)
        try:
            if not mapped.pData or not 1280 <= mapped.RowPitch <= 16384:
                raise ValueError('Invalid mapped camera row pitch')
            raw = ct.string_at(mapped.pData, mapped.RowPitch*180)
            rgb = unpack_rgba_rows(raw, mapped.RowPitch, description['Format'])
        finally:
            self._method(self.context, 15, None, ct.c_void_p, ct.c_uint32)(self.context, staging, 0)
        return rgb, dict(started_at=started, finished_at=time.perf_counter(),
                         description=description, row_pitch=int(mapped.RowPitch),
                         original_handle=handle, source_texture_modified=False)

    def read_group(self, handles, *, timeout=.04):
        """Queue four owned destination copies before one Flush; retain every source.

        Deadline bounds polling, not a driver call's execution time. No partial
        result is returned, and the caller must not acknowledge a failed group.
        """
        handles = [parse_shared_handle(h) for h in handles]
        if len(handles) != 4 or len(set(handles)) != 4 or not 0 < timeout <= .1:
            raise ValueError('Four distinct shared handles and bounded group deadline required')
        started = time.perf_counter()
        self.last_group_trace = dict(started_at=started, timeout_seconds=timeout, stages=[])
        trace = self.last_group_trace
        pinned, resources, result = [], [], []
        try:
            for handle in handles:
                opened = time.perf_counter()
                source, staging, description = self._open(handle)
                # A later _open can evict an older cache entry used by this group.
                for pointer in (source, staging):
                    self._method(pointer, 1, ct.c_uint32)(pointer)  # IUnknown::AddRef
                    pinned.append(pointer)
                stage = dict(original_handle=handle, open_started_at=opened,
                             open_finished_at=time.perf_counter())
                trace['stages'].append(stage)
                resources.append((handle, source, staging, description, stage))
            for _handle, source, staging, _description, stage in resources:
                self._method(self.context, 47, None, ct.c_void_p, ct.c_void_p)(
                    self.context, staging, source)
                stage['copy_enqueued_at'] = time.perf_counter()
            trace['flush_cycles_started'] = current_thread_cycles()
            trace['flush_started_at'] = time.perf_counter()
            self._method(self.context, 111, None)(self.context)
            trace['flush_finished_at'] = time.perf_counter()
            trace['flush_cycles_finished'] = current_thread_cycles()
            deadline = started+timeout
            for handle, _source, staging, description, stage in resources:
                mapped, attempts = MappedResource(), 0
                stage['map_cycles_started'] = current_thread_cycles()
                stage['map_started_at'] = time.perf_counter()
                while True:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Shared four-view GPU group deadline exceeded')
                    status = self._method(self.context, 14, ct.c_int32, ct.c_void_p, ct.c_uint32,
                        ct.c_uint32, ct.c_uint32, ct.POINTER(MappedResource))(
                            self.context, staging, 0, 1, 0x100000, ct.byref(mapped))
                    attempts += 1
                    if status & 0xffffffff != 0x887a000a:
                        self._checked(status, 'Batch Map READ DO_NOT_WAIT')
                        break
                    time.sleep(.0005)
                stage['map_succeeded_at'] = time.perf_counter()
                stage['map_cycles_finished'] = current_thread_cycles()
                stage['map_attempts'] = attempts
                try:
                    if not mapped.pData or not 1280 <= mapped.RowPitch <= 16384:
                        raise ValueError('Invalid mapped group camera row pitch')
                    stage['raw_copy_started_at'] = time.perf_counter()
                    raw = ct.string_at(mapped.pData, mapped.RowPitch*180)
                    stage['raw_copy_finished_at'] = time.perf_counter()
                    stage['rgb_unpack_cycles_started'] = current_thread_cycles()
                    stage['rgb_unpack_thread_cpu_started_at'] = time.thread_time()
                    stage['rgb_unpack_started_at'] = time.perf_counter()
                    rgb = unpack_rgba_rows(raw, mapped.RowPitch, description['Format'])
                    stage['rgb_unpack_finished_at'] = time.perf_counter()
                    stage['rgb_unpack_thread_cpu_finished_at'] = time.thread_time()
                    stage['rgb_unpack_cycles_finished'] = current_thread_cycles()
                finally:
                    stage['unmap_started_at'] = time.perf_counter()
                    self._method(self.context, 15, None, ct.c_void_p, ct.c_uint32)(
                        self.context, staging, 0)
                stage['unmapped_at'] = time.perf_counter()
                result.append((rgb, dict(started_at=started, finished_at=stage['unmapped_at'],
                    description=description, row_pitch=int(mapped.RowPitch), original_handle=handle,
                    source_texture_modified=False, batch_copy=True, stage=dict(stage),
                    flush_finished_at=trace['flush_finished_at'])))
            trace['finished_at'] = time.perf_counter()
            return result
        except BaseException as error:
            trace['error'] = type(error).__name__+': '+str(error)
            trace['failed_at'] = time.perf_counter()
            raise
        finally:
            for pointer in reversed(pinned):
                self._release(pointer)

    def close(self):
        for source, staging, _ in self.textures.values():
            self._release(staging)
            self._release(source)
        self.textures.clear()
        self._release(self.context)
        self._release(self.device)
        self.context, self.device = ct.c_void_p(), ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
