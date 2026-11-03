"""Match an EGL device to CUDA device zero by UUID, never by Slurm numbering."""

import ctypes
import ctypes.util
import json
import os
from pathlib import Path


def choose_device(devices, cuda_uuid):
    matches = [row["egl_index"] for row in devices if row.get("cuda_uuid") == cuda_uuid]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one EGL device matching allocated CUDA UUID; got {matches}")
    return matches[0]


def query_devices():
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
    get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def uuid_for(device):
        value = (ctypes.c_ubyte * 16)()
        code = get_uuid(value, device)
        if code:
            raise RuntimeError(f"cuDeviceGetUuid failed: {code}")
        return bytes(value).hex()

    if cuda.cuInit(0):
        raise RuntimeError("CUDA driver initialization failed")
    selected = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(selected), 0):
        raise RuntimeError("Allocated CUDA device zero is unavailable")
    selected_uuid = uuid_for(selected.value)
    egl = ctypes.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p

    def extension(name, result, *arguments):
        address = egl.eglGetProcAddress(name.encode("ascii"))
        if not address:
            raise RuntimeError(f"Missing EGL extension function: {name}")
        return ctypes.CFUNCTYPE(result, *arguments)(address)

    query = extension("eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
    string = extension("eglQueryDeviceStringEXT", ctypes.c_char_p,
                       ctypes.c_void_p, ctypes.c_int)
    attribute = extension("eglQueryDeviceAttribEXT", ctypes.c_uint,
                          ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))
    count = ctypes.c_int()
    if not query(0, None, ctypes.byref(count)) or not count.value:
        raise RuntimeError("No enumerated EGL devices")
    handles = (ctypes.c_void_p * count.value)()
    if not query(count.value, handles, ctypes.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    devices = []
    for index, handle in enumerate(handles[:count.value]):
        extensions = (string(handle, 0x3055) or b"").decode("ascii", errors="replace")
        row = {"egl_index": index, "extensions": extensions, "cuda_uuid": None}
        if "EGL_NV_device_cuda" in extensions.split():
            ordinal = ctypes.c_ssize_t()
            if attribute(handle, 0x323A, ctypes.byref(ordinal)):
                row["cuda_device"] = ordinal.value
                try:
                    row["cuda_uuid"] = uuid_for(ordinal.value)
                except RuntimeError as error:
                    row["query_error"] = str(error)
        devices.append(row)
    return selected_uuid, devices


def main():
    output = Path(os.environ.get("FLYBRAIN_JOB_OUTPUT", "output"))
    receipt = {
        "status": "failed", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
    }
    try:
        uuid, devices = query_devices()
        receipt.update({"cuda_uuid": uuid, "devices": devices})
        index = choose_device(devices, uuid)
        receipt.update({"status": "passed", "egl_index": index})
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "egl-device.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(index, flush=True)


if __name__ == "__main__":
    main()
