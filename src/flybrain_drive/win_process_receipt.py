"""Read-only lifetime receipt for one Windows process handle, robust to PID reuse."""

import ctypes
import json
import os
import subprocess
from pathlib import Path
from ctypes import wintypes


def engine_inventory():
    # AssettoCorsa.exe is the launcher, not the native physics engine. Do not
    # operate it, and do not mistake its renderer children for an engine handle.
    command="@(Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('acs.exe','acs_x86.exe') } | Select-Object Name,ProcessId) | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell.exe','-NoProfile','-Command',command],cwd=Path.cwd(),
        capture_output=True,text=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    if r.returncode:raise RuntimeError('Authoritative engine inventory failed')
    v=json.loads(r.stdout) if r.stdout.strip() else []
    return v if isinstance(v,list) else [v]



def require_single_engine(pid, inventory):
    if len(inventory) != 1 or inventory[0]["ProcessId"] != pid:
        raise ValueError("Exactly one native AC engine matching the background window is required")


class ProcessReceipt:
    def __init__(self, pid):
        if os.name != 'nt' or type(pid) is not int or pid <= 0:
            raise ValueError('Explicit Windows process ID required')
        self.pid = pid
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.api.GetExitCodeProcess.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.WaitForSingleObject.restype = wintypes.DWORD
        self.handle = self.api.OpenProcess(0x101000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def snapshot(self):
        if self.handle is None:
            raise ValueError('Original process receipt handle is closed')
        code = wintypes.DWORD()
        if not self.api.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        waited = self.api.WaitForSingleObject(self.handle, 0)
        if waited == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        if waited not in (0, 258):
            raise RuntimeError('Unexpected process wait result: '+str(waited))
        # Some owned runs reported an exit value while their wait remained unsignaled.
        # Only the signaled process object proves that termination has completed.
        signaled = waited == 0
        return dict(pid=self.pid, active=not signaled,
                    exit_code=code.value if signaled else None,
                    exit_code_reported=code.value, termination_signaled=signaled,
                    exit_requested=not signaled and code.value != 259)

    def close(self):
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle = None
