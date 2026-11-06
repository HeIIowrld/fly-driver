"""Select and freeze original camera groups independently of GPU inference."""
import queue
import threading
import time

from .ac_csp_vision import NoFreshCSPFrame
from .current_flight_control_contract import FEATURE_SELECTION_MAX_VIEW_AGE
from .immutable_camera_group import freeze_camera_group, require_nonzero_camera_views


class FlightCameraIngress:
    """One queued immutable group. Overload is explicit and terminates the caller."""

    def __init__(self, camera):
        self.camera = camera
        self._pending = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._serve, name="flight-camera-ingress", daemon=True)
        self._thread.start()

    def _serve(self):
        last_capture = -1
        try:
            while not self._stop.is_set():
                try:
                    frame, selected, _ = self.camera.wait_for_candidate(
                        max_age=FEATURE_SELECTION_MAX_VIEW_AGE, timeout=.04)
                except NoFreshCSPFrame:
                    continue
                capture = frame["capture_id"]
                if capture == last_capture:
                    self._stop.wait(.001)
                    continue
                group = freeze_camera_group(frame, self.camera.samples_for(frame), selected_at=selected,
                                            maximum_age=FEATURE_SELECTION_MAX_VIEW_AGE)
                require_nonzero_camera_views(group)
                try:
                    self._pending.put_nowait((frame, group))
                except queue.Full as error:
                    raise TimeoutError("GPU did not consume the previous selected camera group") from error
                last_capture = capture
        except Exception as error:
            self._error = error

    def receive(self):
        if self._error is not None:
            raise RuntimeError("Camera ingress failed") from self._error
        try:
            frame, group = self._pending.get(timeout=.04)
        except queue.Empty as error:
            if self._error is not None:
                raise RuntimeError("Camera ingress failed") from self._error
            raise NoFreshCSPFrame("No selected group ready within 40ms") from error
        if self._error is not None:
            raise RuntimeError("Camera ingress failed") from self._error
        if not 0 <= time.perf_counter() - group["selected_at"] <= .04:
            raise TimeoutError("Selected camera group waited more than 40ms before GPU dispatch")
        return frame, group

    def close(self):
        self._stop.set()
        self._thread.join(.25)
        if self._thread.is_alive():
            raise RuntimeError("Camera ingress did not terminate")
