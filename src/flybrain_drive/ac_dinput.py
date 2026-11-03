"""Read-only DirectInput game-controller identities; no device creation or acquisition."""

import ctypes
import os
import uuid


class GUID(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_uint8 * 8),
    ]

    def text(self):
        return str(uuid.UUID(bytes_le=bytes(self))).upper()


class DeviceInstanceW(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("instance_guid", GUID),
        ("product_guid", GUID),
        ("device_type", ctypes.c_uint32),
        ("instance_name", ctypes.c_uint16 * 260),
        ("product_name", ctypes.c_uint16 * 260),
        ("ff_driver_guid", GUID),
        ("usage_page", ctypes.c_uint16),
        ("usage", ctypes.c_uint16),
    ]


def utf16_name(value):
    return bytes(value).decode("utf-16-le").split("\0", 1)[0]


def inspect_vjoy_controllers():
    if os.name != "nt":
        raise OSError("Native DirectInput identity discovery requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = (
        [ctypes.c_wchar_p],
        ctypes.c_void_p,
    )
    library = ctypes.WinDLL("dinput8")
    create = library.DirectInput8Create
    create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    create.restype = ctypes.c_int32
    iid = GUID.from_buffer_copy(uuid.UUID("BF798031-483A-4DA2-AA99-5D64ED369700").bytes_le)
    interface = ctypes.c_void_p()
    result = create(
        kernel.GetModuleHandleW(None), 0x0800, ctypes.byref(iid), ctypes.byref(interface), None
    )
    if result < 0 or not interface.value:
        raise OSError("DirectInput8Create failed: " + hex(result & 0xFFFFFFFF))
    table = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int32, ctypes.POINTER(DeviceInstanceW), ctypes.c_void_p
    )
    devices, count, errors = [], 0, []

    @callback_type
    def visit(pointer, _context):
        nonlocal count
        try:
            item = pointer.contents
            if item.size != ctypes.sizeof(DeviceInstanceW):
                raise ValueError("DirectInput device ABI size differs")
            count += 1
            instance, product = utf16_name(item.instance_name), utf16_name(item.product_name)
            if "vjoy" in (instance + " " + product).casefold():
                devices.append(
                    {
                        "instance_guid": item.instance_guid.text(),
                        "product_guid": item.product_guid.text(),
                        "instance_name": instance,
                        "product_name": product,
                        "usage_page": item.usage_page,
                        "usage": item.usage,
                    }
                )
        except Exception as error:
            errors.append(str(error))
        return 1

    enumerate_devices = ctypes.WINFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        callback_type,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )(table[4])
    release = ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(table[2])
    try:
        result = enumerate_devices(interface, 4, visit, None, 1)  # GAMECTRL, ATTACHEDONLY.
        if result < 0 or errors:
            raise OSError(
                "DirectInput enumeration failed: " + str(errors or hex(result & 0xFFFFFFFF))
            )
    finally:
        release(interface)
    return {
        "attached_game_controller_count": count,
        "vjoy_devices": devices,
        "candidate_ac_joy_index": 0 if count == len(devices) == 1 else None,
        "enumeration_order_guaranteed": False,
        "device_acquisitions": 0,
        "actuator_messages_sent": 0,
        "game_binding_verified": False,
    }
