"""Fake SDK requests only: no DLL loading, driver installation or game control."""

import ctypes

import pytest

from flybrain_drive.ac_vjoy import AxisBinding, JoystickFrame, VJoySession


class FakeSDK:
    def __init__(self, status=1):
        self.status, self.frames, self.acquires = status, [], 0
        self.fail_update = False

    def vJoyEnabled(self):
        return 1

    def GetVJDStatus(self, device):
        return self.status

    def GetVJDButtonNumber(self, device):
        return 128

    def GetVJDAxisExist(self, device, axis):
        return axis in (0x30, 0x31, 0x32)

    def GetVJDAxisMin(self, device, axis, pointer):
        pointer._obj.value = 10
        return 1

    def GetVJDAxisMax(self, device, axis, pointer):
        pointer._obj.value = 110
        return 1

    def AcquireVJD(self, device):
        self.acquires += 1
        self.status = 0
        return 1

    def RelinquishVJD(self, device):
        self.status = 1

    def UpdateVJD(self, device, pointer):
        if self.fail_update:
            return 0
        self.frames.append(JoystickFrame.from_buffer_copy(bytes(pointer._obj)))
        return 1


def bindings():
    return [
        AxisBinding("steering", 0x30, False),
        AxisBinding("throttle", 0x31, True),
        AxisBinding("brake", 0x32, False),
    ]


def test_sdk_frame_abi_ranges_inversion_and_requests_are_not_applied_receipts():
    assert ctypes.sizeof(JoystickFrame) == 108
    offsets = [getattr(JoystickFrame, n).offset for n in ("wAxisX", "wAxisY", "wAxisZ")]
    assert offsets == [16, 20, 24]
    sdk = FakeSDK()
    client = VJoySession(sdk, 1, bindings())
    with pytest.raises(RuntimeError, match="ownership"):
        client.submit(0.0, 0.0, 1.0)
    client.open()
    first = sdk.frames[0]
    assert (first.wAxisX, first.wAxisY, first.wAxisZ) == (60, 110, 110)
    receipt = client.submit(-1.0, 0.75, 0.25)
    frame = sdk.frames[-1]
    assert (frame.wAxisX, frame.wAxisY, frame.wAxisZ) == (10, 35, 35)
    assert frame.bHats == 0xFFFFFFFF and frame.lButtons == 0
    assert receipt["driver_accepted"] and not receipt["game_applied"]
    assert (receipt['driver_call_started_perf_counter_seconds']
            <= receipt['driver_call_finished_perf_counter_seconds'])
    before = len(sdk.frames)
    with pytest.raises(ValueError, match="bounded"):
        client.submit(0.0, 2.0, 0.0)
    assert len(sdk.frames) == before
    client.close()
    assert not client.owned and sdk.status == 1
    assert sdk.frames[-1].wAxisY == 110  # Inverted zero throttle.


def test_busy_or_missing_axis_never_acquires_and_cleanup_relinquishes_on_failure():
    sdk = FakeSDK(status=2)
    client = VJoySession(sdk, 1, bindings())
    with pytest.raises(RuntimeError, match="FREE"):
        client.open()
    assert sdk.acquires == 0
    sdk.status = 1
    sdk.GetVJDAxisExist = lambda device, axis: axis != 0x32
    with pytest.raises(RuntimeError, match="missing"):
        client.open()
    assert sdk.acquires == 0
    sdk.GetVJDAxisExist = lambda device, axis: axis in (0x30, 0x31, 0x32)
    client.open()
    sdk.fail_update = True
    with pytest.raises(RuntimeError, match="rejected"):
        client.close()
    assert sdk.status == 1 and not client.owned
    sdk.fail_update = False
    sdk.GetVJDAxisExist = lambda device, axis: axis != 0x32
    with pytest.raises(RuntimeError, match="missing"):
        client.open()
    assert sdk.acquires == 1


def test_button_banks_are_explicit_and_neutral_close_releases_every_button():
    sdk = FakeSDK()
    client = VJoySession(sdk, 1, bindings())
    client.open()
    receipt = client.submit(0., 0., 1., pressed_buttons=(0, 31, 32, 127))
    frame = sdk.frames[-1]
    assert frame.lButtons & 0xffffffff == 0x80000001
    assert frame.lButtonsEx1 == 1 and frame.lButtonsEx2 == 0
    assert frame.lButtonsEx3 & 0xffffffff == 0x80000000
    assert receipt['driver_pressed_buttons_zero_based'] == [0, 31, 32, 127]
    before = len(sdk.frames)
    for invalid in [(0, 0), (-1,), (128,), (True,)]:
        with pytest.raises(ValueError):
            client.submit(0., 0., 1., pressed_buttons=invalid)
    sdk.GetVJDButtonNumber = lambda _: 2
    with pytest.raises(ValueError, match='not present'):
        client.submit(0., 0., 1., pressed_buttons=(2,))
    assert len(sdk.frames) == before
    client.close()
    last = sdk.frames[-1]
    assert [last.lButtons, last.lButtonsEx1, last.lButtonsEx2, last.lButtonsEx3] == [0]*4
    assert sdk.status == 1
