"""Read-only session/desktop readiness for native AC capture; never log in or switch desktops."""

import ctypes
import os
from ctypes import wintypes

STATE_NAMES = ('active', 'connected', 'connect_query', 'shadow', 'disconnected',
               'idle', 'listen', 'reset', 'down', 'init')


def classify_capture_session(collector, engine, input_desktop_accessible):
    """Classify measured records, not usernames or desktop pixels."""
    reasons = []
    for label, record in [('collector', collector), ('engine', engine)]:
        state = record.get('connect_state')
        session = record.get('session_id')
        if (type(state) is not int or state not in range(len(STATE_NAMES))
                or type(session) is not int or session < 0):
            raise ValueError('Explicit numeric WTS session/state required')
        if state != 0:
            reasons.append(label + ' Windows session is ' + STATE_NAMES[state])
    if collector['session_id'] != engine['session_id']:
        reasons.append('collector and engine are in different Windows sessions')
    if input_desktop_accessible is not True:
        reasons.append('input desktop is inaccessible')
    return dict(collector=collector, engine=engine,
        input_desktop_accessible=input_desktop_accessible, reasons=reasons,
        session_ready_for_capture_attempt=not reasons,
        capture_success_verified=False, foreground_activation_attempted=False)


def inspect_capture_session(engine_pid):
    if os.name != 'nt':
        raise OSError('Native Windows session inspection required')
    if type(engine_pid) is not int or engine_pid <= 0:
        raise ValueError('Explicit live engine PID required')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    wts = ctypes.WinDLL('wtsapi32', use_last_error=True)
    user = ctypes.WinDLL('user32', use_last_error=True)
    kernel.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel.ProcessIdToSessionId.restype = wintypes.BOOL
    wts.WTSQuerySessionInformationW.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_int,
        ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.DWORD)]
    wts.WTSQuerySessionInformationW.restype = wintypes.BOOL
    wts.WTSFreeMemory.argtypes = [wintypes.LPVOID]
    wts.WTSFreeMemory.restype = None
    user.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    user.OpenInputDesktop.restype = wintypes.HANDLE
    user.CloseDesktop.argtypes = [wintypes.HANDLE]
    user.CloseDesktop.restype = wintypes.BOOL

    def process_record(pid):
        session_id = wintypes.DWORD()
        if not kernel.ProcessIdToSessionId(pid, ctypes.byref(session_id)):
            raise ctypes.WinError(ctypes.get_last_error())
        buffer, size = wintypes.LPVOID(), wintypes.DWORD()
        # WTSConnectState=8, WTS_CURRENT_SERVER_HANDLE=NULL.
        if not wts.WTSQuerySessionInformationW(None, session_id.value, 8,
                                              ctypes.byref(buffer), ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if size.value != ctypes.sizeof(ctypes.c_int) or not buffer:
                raise ValueError('WTS state ABI differs')
            state = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_int)).contents.value
        finally:
            wts.WTSFreeMemory(buffer)
        return dict(pid=pid, session_id=session_id.value, connect_state=state)

    collector, engine = process_record(os.getpid()), process_record(engine_pid)
    desktop = user.OpenInputDesktop(0, False, 1)  # DESKTOP_READOBJECTS only.
    error = ctypes.get_last_error() if not desktop else None
    if desktop:
        user.CloseDesktop(desktop)
    result = classify_capture_session(collector, engine, bool(desktop))
    result['input_desktop_error'] = error
    return result


def require_capture_session(engine_pid):
    result = inspect_capture_session(engine_pid)
    if not result['session_ready_for_capture_attempt']:
        raise RuntimeError('AC capture session unavailable: ' + '; '.join(result['reasons']))
    return result
