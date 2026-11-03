"""Read pinned AC Gym image ABI; missing exposure metadata stays unknown."""

import ctypes
import os
import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticImage:
    active_index: int
    pixels: np.ndarray
    exposure_id: None = None
    exposure_time: None = None
    synchronization_verified: bool = False


def decode_image_buffer(data, width, height, color_mode):
    if (type(width) is not int or type(height) is not int or min(width, height) < 1
            or width * height > 16_777_216 or color_mode not in ("bgr", "gray")):
        raise ValueError("Explicit bounded image geometry and color mode required")
    channels = 3 if color_mode == "bgr" else 1
    size = width * height * channels
    if len(data) != 4 + 2 * size:
        raise ValueError("Image shared-memory layout/size mismatch")
    active, = struct.unpack_from("<I", data)
    if active not in (0, 1):
        raise ValueError("Invalid active image buffer index")
    pixels = np.frombuffer(data, dtype=np.uint8, count=size, offset=4 + active * size).copy()
    pixels = pixels.reshape((height, width, 3) if channels == 3 else (height, width))
    if channels == 3:
        pixels = pixels[..., ::-1].copy()  # BGR producer -> RGB diagnostic image.
    pixels.setflags(write=False)
    return DiagnosticImage(active, pixels)


class ExistingNamedMapping:
    """Windows OpenFileMapping(FILE_MAP_READ), never create a missing producer mapping."""

    def __init__(self, name, size):
        if os.name != "nt":
            raise OSError("Named mapping reader requires Windows")
        if (not isinstance(name, str) or not 0 < len(name) <= 260 or "\0" in name
                or type(size) is not int or not 0 < size <= 100_663_300):
            raise ValueError("Explicit bounded mapping name/size required")
        from ctypes import wintypes

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        self.api.OpenFileMappingW.restype = wintypes.HANDLE
        self.api.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.DWORD, ctypes.c_size_t]
        self.api.MapViewOfFile.restype = ctypes.c_void_p
        self.api.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        self.api.UnmapViewOfFile.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.OpenFileMappingW(4, False, name)
        if not self.handle:
            error = ctypes.get_last_error()
            if error == 2:
                raise FileNotFoundError(error, "Producer mapping does not exist", name)
            raise ctypes.WinError(error)
        self.view = self.api.MapViewOfFile(self.handle, 4, 0, 0, size)
        if not self.view:
            error = ctypes.get_last_error()
            self.api.CloseHandle(self.handle)
            self.handle = None
            raise ctypes.WinError(error)
        self.size = size

    def read(self):
        if not self.view:
            raise RuntimeError("Mapping is closed")
        return ctypes.string_at(self.view, self.size)

    def close(self):
        if self.view:
            self.api.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
