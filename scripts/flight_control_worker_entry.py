"""Exit the owned GPU interpreter only AFTER target cleanup; avoid ROCm finalizer hangs."""
import argparse,multiprocessing as mp,os,runpy,sys,time,traceback,hashlib,json
from pathlib import Path
from flybrain_drive.evidence import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--worker-receipt',required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    sys.argv=[a.module,*args];code=0;error=None
    try:runpy.run_module(a.module,run_name='__main__')
    except SystemExit as value:code=value.code if isinstance(value.code,int) else 1
    except BaseException as value:code=1;error=type(value).__name__+': '+str(value);traceback.print_exc()
    import psutil
    deadline=time.perf_counter()+15.;helpers=[]
    while True:
        children=[];helpers=[];pending=[psutil.Process()]
        while pending:
            parent=pending.pop()
            for child in parent.children(recursive=False):
                try:
                    if not child.is_running() or child.create_time()<parent.create_time():continue
                    command=child.cmdline()
                    console=Path(child.exe()).resolve()==Path(os.environ.get('WINDIR','C:/Windows'),'System32/conhost.exe').resolve()
                    if console:
                        helpers.append(dict(pid=child.pid,kind='Verified Windows System32 console host; closes on parent exit',creation_unix_seconds=child.create_time()))
                    elif any('from multiprocessing.resource_tracker import main' in arg for arg in command):
                        helpers.append(dict(pid=child.pid,kind='CPython resource_tracker; exits on parent pipe EOF',creation_unix_seconds=child.create_time()))
                    else:children.append(child);pending.append(child)
                except psutil.NoSuchProcess:pass
        if children:
            records=[]
            for child in children:
                try:records.append(dict(pid=child.pid,name=child.name(),created=child.create_time(),command_sha256=hashlib.sha256(json.dumps(child.cmdline()).encode()).hexdigest(),executable=child.exe()))
                except psutil.NoSuchProcess:pass
            write_json(str(a.worker_receipt)+'.children.json',records)
        if not children or time.perf_counter()>=deadline:break
        time.sleep(.05)
    write_json(a.worker_receipt,dict(pid=os.getpid(),code=code,error=error,finished=time.perf_counter(),
        active_owned_children=[child.pid for child in children],nonactuating_parent_eof_helpers=helpers,target_returned_after_its_finally=True,
        exit_strategy='Native TerminateProcess on CURRENT interpreter pseudo-handle after target cleanup and zero live owned children; bypass affected ROCm DLL teardown'))
    sys.stdout.flush();sys.stderr.flush()
    if children:raise RuntimeError('Do not bypass finalization while owned children remain live')
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        if not kernel.TerminateProcess(kernel.GetCurrentProcess(),code):raise OSError(ctypes.get_last_error(),'Current interpreter exit failed')
    os._exit(code)

if __name__=='__main__':main()
