"""Process-owned high-resolution timer plus wake event; never changes global timer settings."""

import ctypes as ct
import math
import os


class WindowsDeadlineWaiter:
    backend = 'win32_high_resolution_waitable_timer'

    def __init__(self):
        if os.name != 'nt':
            raise OSError('Windows high-resolution deadline waiter required')
        k = self.k = ct.WinDLL('kernel32', use_last_error=True)
        handle, boolean, dword = ct.c_void_p, ct.c_int32, ct.c_uint32
        k.CreateEventW.argtypes = [ct.c_void_p, boolean, boolean, ct.c_wchar_p]
        k.CreateEventW.restype = handle
        k.CreateWaitableTimerExW.argtypes = [ct.c_void_p, ct.c_wchar_p, dword, dword]
        k.CreateWaitableTimerExW.restype = handle
        k.SetWaitableTimer.argtypes = [handle, ct.POINTER(ct.c_int64), ct.c_int32,
                                      ct.c_void_p, ct.c_void_p, boolean]
        k.SetWaitableTimer.restype = boolean
        for name in ('SetEvent', 'ResetEvent', 'CancelWaitableTimer', 'CloseHandle'):
            method = getattr(k, name)
            method.argtypes, method.restype = [handle], boolean
        k.WaitForMultipleObjects.argtypes = [dword, ct.POINTER(handle), boolean, dword]
        k.WaitForMultipleObjects.restype = dword
        self.event, self.timer = k.CreateEventW(None, True, False, None), None
        self.closed = False
        if not self.event:
            raise ct.WinError(ct.get_last_error())
        # HIGH_RESOLUTION, TIMER_MODIFY_STATE | SYNCHRONIZE, unnamed/non-inherited.
        self.timer = k.CreateWaitableTimerExW(None, None, 2, 0x100002)
        if not self.timer:
            error = ct.get_last_error()
            k.CloseHandle(self.event)
            raise ct.WinError(error)
        self.handles = (handle*2)(self.event, self.timer)

    def set(self):
        if self.closed or not self.k.SetEvent(self.event):
            raise RuntimeError('Cannot signal owned deadline wake event')

    def clear(self):
        if self.closed or not self.k.ResetEvent(self.event):
            raise RuntimeError('Cannot reset owned deadline wake event')

    def wait(self, seconds):
        if self.closed or not math.isfinite(seconds) or not 0 <= seconds <= .15:
            raise ValueError('Finite owned wait at most150ms required')
        relative = ct.c_int64(-max(1, math.ceil(seconds*10_000_000)))
        if not self.k.SetWaitableTimer(self.timer, ct.byref(relative), 0, None, None, False):
            raise ct.WinError(ct.get_last_error())
        status = self.k.WaitForMultipleObjects(2, self.handles, False, 1000)
        if status not in (0, 1):
            raise RuntimeError('Owned high-resolution wait failed: '+hex(status))
        return status == 0

    def close(self):
        # Caller must first join the only waiting worker; closing pending handles is undefined.
        if not self.closed:
            self.k.CancelWaitableTimer(self.timer)
            self.k.CloseHandle(self.timer)
            self.k.CloseHandle(self.event)
            self.closed = True
