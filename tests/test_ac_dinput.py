import ctypes
import uuid

from flybrain_drive.ac_dinput import GUID, DeviceInstanceW, utf16_name


def test_dinput_windows_guid_and_utf16_abi_do_not_depend_on_host_wchar_size():
    assert ctypes.sizeof(GUID) == 16 and ctypes.sizeof(DeviceInstanceW) == 1100
    assert DeviceInstanceW.instance_name.offset == 40
    value = uuid.UUID("BF798031-483A-4DA2-AA99-5D64ED369700")
    assert GUID.from_buffer_copy(value.bytes_le).text() == str(value).upper()
    name = (ctypes.c_uint16 * 260)()
    encoded = "vJoy Device".encode("utf-16-le")
    ctypes.memmove(name, encoded, len(encoded))
    assert utf16_name(name) == "vJoy Device"
