import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.ac_shared_texture import (
    MappedResource,
    SharedTextureReader,
    TextureDescription,
    parse_shared_handle,
    unpack_rgba_rows,
    verify_shared_frame,
)


def fake_group_reader(*, failed_map=False):
    reader = SharedTextureReader.__new__(SharedTextureReader)
    reader.context = ctypes.c_void_p(1)
    operations, buffers = [], []
    reader._open = lambda h: (ctypes.c_void_p(h+10), ctypes.c_void_p(h+20),
                              dict(Format=28, Width=320, Height=180))
    reader._release = lambda p: operations.append(('release', p.value))

    def method(_pointer, slot, _result, *_types):
        def call(*args):
            if slot == 1:
                operations.append(('pin', args[0].value))
                return 2
            if slot == 47:
                operations.append(('copy', args[1].value, args[2].value))
            elif slot == 111:
                operations.append(('flush',))
            elif slot == 14:
                assert args[3:5] == (1, 0x100000)
                operations.append(('map', args[1].value))
                if failed_map:
                    raise OSError('mock map failure')
                raw = bytes([args[1].value-20, 2, 3, 255])*(320*180)
                buffer = ctypes.create_string_buffer(raw)
                buffers.append(buffer)
                mapped = ctypes.cast(args[-1], ctypes.POINTER(MappedResource)).contents
                mapped.pData, mapped.RowPitch = ctypes.addressof(buffer), 1280
                return 0
            elif slot == 15:
                operations.append(('unmap', args[1].value))
        return call
    reader._method = method
    return reader, operations


def test_four_copies_are_queued_before_one_flush_and_any_map():
    reader, operations = fake_group_reader()
    rows = reader.read_group([1, 2, 3, 4])
    kinds = [r[0] for r in operations]
    assert kinds.count('copy') == 4 and kinds.count('flush') == 1
    assert max(i for i, kind in enumerate(kinds) if kind == 'copy') < kinds.index('flush')
    assert kinds.index('flush') < kinds.index('map')
    assert kinds.count('pin') == kinds.count('release') == 8
    assert kinds.count('unmap') == 4
    for handle, (rgb, receipt) in enumerate(rows, 1):
        np.testing.assert_array_equal(rgb, np.broadcast_to([handle, 2, 3], rgb.shape))
        assert receipt['original_handle'] == handle
        assert receipt['source_texture_modified'] is False
        assert receipt['stage']['map_attempts'] == 1


def test_failed_group_releases_all_pins_and_keeps_stage_trace():
    reader, operations = fake_group_reader(failed_map=True)
    with pytest.raises(OSError, match='mock map'):
        reader.read_group([1, 2, 3, 4])
    assert sum(op[0] == 'release' for op in operations) == 8
    assert 'mock map failure' in reader.last_group_trace['error']


def test_group_requires_four_distinct_sources():
    reader, operations = fake_group_reader()
    with pytest.raises(ValueError, match='distinct'):
        reader.read_group([1, 2, 3, 3])
    assert not operations


def test_native_interface_slots_and_layout_match_primary_microsoft_header():
    record = json.loads(Path('assets/external/d3d11-shared-texture-api-20261028/source.json')
                        .read_text(encoding='utf8'))['methods']
    assert record['ID3D11Device']['OpenSharedResource'] == 28
    assert record['ID3D11Device']['CreateTexture2D'] == 5
    assert record['ID3D11DeviceContext']['CopyResource'] == 47
    assert record['ID3D11DeviceContext']['Map'] == 14
    assert record['ID3D11DeviceContext']['Unmap'] == 15
    assert record['ID3D11DeviceContext']['Flush'] == 111
    assert record['ID3D11Texture2D']['GetDesc'] == 10
    assert ctypes.sizeof(TextureDescription) == 44
    dxgi = json.loads(Path('assets/external/d3d11-shared-texture-api-20261028/dxgi-source.json')
                      .read_text(encoding='utf8'))['methods']
    assert dxgi['IDXGIFactory1']['EnumAdapters1'] == 12
    assert dxgi['IDXGIAdapter1']['GetDesc1'] == 10


@pytest.mark.parametrize('value', [True, -1, 0, 'abc', '12.0', str(2**64)])
def test_shared_handle_is_explicit_original_unsigned_integer(value):
    with pytest.raises(ValueError):
        parse_shared_handle(value)
    assert parse_shared_handle('3221226370ULL') == 3221226370


@pytest.mark.parametrize('fmt,expected', [(28, [1, 2, 3]), (29, [1, 2, 3]),
                                       (87, [3, 2, 1]), (91, [3, 2, 1])])
def test_staging_rgb_conversion_preserves_pixels_and_skips_row_padding(fmt, expected):
    row = bytes([1, 2, 3, 255])*320+bytes([199])*64
    pixels = unpack_rgba_rows(row*180, 1344, fmt)
    assert pixels.shape == (180, 320, 3) and pixels.dtype == np.uint8
    assert np.array_equal(pixels, np.broadcast_to(expected, pixels.shape))


def test_unknown_texture_format_or_inexact_stride_cannot_be_used_as_camera():
    with pytest.raises(ValueError):
        unpack_rgba_rows(bytes(1280*180), 1280, 10)
    with pytest.raises(ValueError):
        unpack_rgba_rows(bytes(1279*180), 1279, 28)


@pytest.mark.parametrize('mutation', ['none', 'ack_before_copy', 'wrong_generation',
                                     'wrong_handle', 'source_written', 'copy_after_receipt'])
def test_shared_group_requires_same_source_and_complete_copy_before_ack(mutation):
    names = ['front', 'left', 'right', 'rear']
    frame = dict(transport='shared_immutable_v1', request_id='owned', generation='one', index=3,
        views=[dict(view_id=name, image_writer='host_lossless_bmp',
                    shared_texture_handle=i+1) for i, name in enumerate(names)])
    row = dict(capture_id=3, generation='one', received=2.,
        views=[dict(view_id=name, render_started_lower=1.) for name in names],
        shared_copy_receipts=[dict(view_id=name, original_handle=i+1,
            source_texture_modified=False, started_at=1.1, finished_at=1.2,
            description=dict(Width=320, Height=180, MipLevels=1, ArraySize=1,
                             SampleCount=1, Format=28)) for i, name in enumerate(names)])
    ack = dict(request_id='owned', generation='one', capture_id=3, copied_before_ack=2.1)
    if mutation == 'ack_before_copy':
        ack['copied_before_ack'] = 1.15
    elif mutation == 'wrong_generation':
        ack['generation'] = 'other'
    elif mutation == 'wrong_handle':
        row['shared_copy_receipts'][0]['original_handle'] = 9
    elif mutation == 'source_written':
        row['shared_copy_receipts'][0]['source_texture_modified'] = True
    elif mutation == 'copy_after_receipt':
        row['shared_copy_receipts'][0]['finished_at'] = 2.05
    if mutation == 'none':
        verify_shared_frame(frame, row, ack)
    else:
        with pytest.raises(ValueError):
            verify_shared_frame(frame, row, ack)


@pytest.mark.parametrize('index', [0, 1, 3])
def test_reviewed_adapter_follows_identity_when_dxgi_order_changes(index):
    from flybrain_drive.ac_shared_texture import reviewed_adapter_index

    amd = dict(index=index, name='AMD Radeon RX 9070 XT', vendor_id=4098, device_id=30032)
    other = dict(index=7, name='Intel(R) Graphics', vendor_id=32902, device_id=32103)
    assert reviewed_adapter_index([other, amd]) == index
    for adapters in ([other], [amd, dict(amd, index=2)],
                     [dict(amd, device_id=0)], [dict(amd, vendor_id=0)]):
        with pytest.raises(ValueError):
            reviewed_adapter_index(adapters)
