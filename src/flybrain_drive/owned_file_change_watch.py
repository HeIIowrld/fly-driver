"""Nonblocking Windows directory notifications for one immutable owned pathname.

Arm before the baseline read, reject overflow/errors, latch even brief changes.
No desktop APIs, writes, rehash loops or silent rescan after notification loss.
"""
import ctypes,struct
from pathlib import Path
from ctypes import wintypes as W

class _Overlapped(ctypes.Structure):
    _fields_=[('Internal',ctypes.c_size_t),('InternalHigh',ctypes.c_size_t),
        ('Offset',W.DWORD),('OffsetHigh',W.DWORD),('hEvent',W.HANDLE)]

def changed_names(data):
    names=[];offset=0
    while True:
        if offset+12>len(data):raise ValueError('Truncated directory notification')
        following,action,length=struct.unpack_from('<3I',data,offset)
        if action not in (1,2,3,4,5) or length%2 or offset+12+length>len(data):raise ValueError('Invalid directory notification')
        names.append(data[offset+12:offset+12+length].decode('utf-16-le').casefold())
        if following==0:break
        if following<12+length or following%4:raise ValueError('Directory notification next record differs')
        offset+=following
    return names

class OwnedFileChangeWatch:
    def __init__(self,path):
        self.path=Path(path).resolve();self.changed=False;self.handle=None;self.event=None;self.completions=0;self.pending=False
        api=self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.CreateFileW.argtypes=[W.LPCWSTR,W.DWORD,W.DWORD,ctypes.c_void_p,W.DWORD,W.DWORD,W.HANDLE];api.CreateFileW.restype=W.HANDLE
        api.CreateEventW.argtypes=[ctypes.c_void_p,W.BOOL,W.BOOL,W.LPCWSTR];api.CreateEventW.restype=W.HANDLE
        api.ReadDirectoryChangesW.argtypes=[W.HANDLE,ctypes.c_void_p,W.DWORD,W.BOOL,W.DWORD,ctypes.c_void_p,ctypes.POINTER(_Overlapped),ctypes.c_void_p];api.ReadDirectoryChangesW.restype=W.BOOL
        api.GetOverlappedResult.argtypes=[W.HANDLE,ctypes.POINTER(_Overlapped),ctypes.POINTER(W.DWORD),W.BOOL];api.GetOverlappedResult.restype=W.BOOL
        api.CancelIoEx.argtypes=[W.HANDLE,ctypes.POINTER(_Overlapped)];api.CancelIoEx.restype=W.BOOL
        api.ResetEvent.argtypes=[W.HANDLE];api.ResetEvent.restype=W.BOOL
        api.CloseHandle.argtypes=[W.HANDLE];api.CloseHandle.restype=W.BOOL
        try:
            self.handle=api.CreateFileW(str(self.path.parent),1,7,None,3,0x02000000|0x40000000,None)
            if self.handle==ctypes.c_void_p(-1).value:self.handle=None;raise ctypes.WinError(ctypes.get_last_error())
            self.event=api.CreateEventW(None,True,False,None)
            if not self.event:raise ctypes.WinError(ctypes.get_last_error())
            self.buffer=ctypes.create_string_buffer(8192);self.overlap=_Overlapped();self.overlap.hEvent=self.event
            self._arm()
        except BaseException:self.close();raise
    def _arm(self):
        if not self.api.ResetEvent(self.event):raise ctypes.WinError(ctypes.get_last_error())
        self.overlap.Internal=0;self.overlap.InternalHigh=0
        if not self.api.ReadDirectoryChangesW(self.handle,self.buffer,len(self.buffer),False,1|8|16|64,None,ctypes.byref(self.overlap),None):
            raise ctypes.WinError(ctypes.get_last_error())
        self.pending=True
    def poll(self):
        if not self.pending:raise RuntimeError('Original directory watch is not armed')
        for _ in range(4):
            size=W.DWORD()
            if not self.api.GetOverlappedResult(self.handle,ctypes.byref(self.overlap),ctypes.byref(size),False):
                error=ctypes.get_last_error()
                if error==996:return self.changed # ERROR_IO_INCOMPLETE: original notification is pending.
                raise ctypes.WinError(error)
            self.pending=False
            if size.value==0:raise RuntimeError('Directory notification overflow; do not infer unchanged state')
            names=changed_names(self.buffer.raw[:size.value]);self.completions+=1
            self.changed=self.changed or self.path.name.casefold() in names
            self._arm()
        raise RuntimeError('Finite directory notification drain budget exhausted')
    def close(self):
        if self.handle and self.pending:
            self.api.CancelIoEx(self.handle,ctypes.byref(self.overlap))
            size=W.DWORD();self.api.GetOverlappedResult(self.handle,ctypes.byref(self.overlap),ctypes.byref(size),True)
            self.pending=False
        if self.handle:self.api.CloseHandle(self.handle);self.handle=None
        if self.event:self.api.CloseHandle(self.event);self.event=None
