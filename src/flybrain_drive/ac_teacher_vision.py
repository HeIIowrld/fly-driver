"""Asynchronous exact-game-window evidence writer; no actor crops or exposure-clock claims."""

import shutil
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .ac_wgc import WGCACCapture
from .evidence import write_json
from .graph import sha256


class TeacherVisionRecorder:
    def __init__(self, directory, hwnd, *, max_frames=300, max_bytes=2 * 1024**3,
                 capture_factory=WGCACCapture):
        if (type(max_frames) is not int or not 1 <= max_frames <= 10000
                or type(max_bytes) is not int or not 1024 <= max_bytes <= 8 * 1024**3):
            raise ValueError('Finite image storage/frame budgets required')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.hwnd, self.max_frames, self.max_bytes = hwnd, max_frames, max_bytes
        self.capture_factory = capture_factory
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frames = deque(maxlen=128)
        self.error, self.count, self.bytes = None, 0, 0
        self.write_seconds = []
        self.thread = threading.Thread(target=self._run, name='ac-vision-writer', daemon=True)
        self.thread.start()

    def _run(self):
        import json

        capture = None
        try:
            capture = self.capture_factory(self.hwnd)
            startup_deadline = time.monotonic() + 3.
            with (self.directory / 'frames.jsonl').open('x', encoding='utf8') as stream:
                while not self.stop.is_set():
                    if self.count >= self.max_frames:
                        raise RuntimeError('Vision frame budget exhausted')
                    try:
                        sample = capture.capture(timeout=.2)
                    except TimeoutError:
                        if self.stop.is_set():
                            break
                        if self.count == 0 and time.monotonic() < startup_deadline:
                            continue
                        raise
                    if self.bytes + sample.rgb.nbytes + 256 > self.max_bytes:
                        raise RuntimeError('Vision byte budget exhausted')
                    started = time.monotonic()
                    file = self.directory / f'rgb-{self.count:06d}.npy'
                    with file.open('xb') as output:
                        np.save(output, sample.rgb, allow_pickle=False)
                    row = dict(sample.metadata(), file=file.name, file_sha256=sha256(file),
                        committed_monotonic_seconds=time.monotonic(),
                        foreground_at_write=bool(
                            capture.native.user.GetForegroundWindow() == self.hwnd))
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    with self.lock:
                        self.frames.append(row)
                        self.count += 1
                        self.bytes += file.stat().st_size
                        self.write_seconds.append(time.monotonic() - started)
        except Exception as error:
            self.error = error
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception as error:
                    self.error = self.error or error

    def candidate_before(self, observation_receipt, *, max_age=.2):
        """Only already-written frames whose callback finished before the sensor receipt.

        This is causal host ordering, not proof that pixels and sensors describe the same
        simulation instant. The privileged teacher's action never depends on this image.
        """
        if (not np.isfinite([observation_receipt, max_age]).all()
                or observation_receipt < 0 or not 0 < max_age <= 1):
            raise ValueError('Finite same-host receipt and age bound required')
        if self.error is not None:
            raise RuntimeError('Background vision recorder failed') from self.error
        with self.lock:
            candidates = [r for r in self.frames
                          if r['finished_monotonic_seconds'] <= observation_receipt]
            row = max(candidates, key=lambda r: r['finished_monotonic_seconds']) \
                if candidates else None
        if row is None or observation_receipt - row['finished_monotonic_seconds'] > max_age:
            raise ValueError('No committed preceding image within declared host age bound')
        return dict(file='vision/' + row['file'], file_sha256=row['file_sha256'],
            capture_id=row['capture_id'], rgb_sha256=row['rgb_sha256'],
            callback_finished_monotonic_seconds=row['finished_monotonic_seconds'],
            observation_receipt_monotonic_seconds=observation_receipt,
            host_age_seconds=observation_receipt - row['finished_monotonic_seconds'],
            teacher_action_depends_on_image=False, exposure_synchronization_verified=False,
            camera_crop_calibrated=False, actor_stream_created=False, training_admissible=False)

    def wait_ready(self, *, timeout=3.5):
        if not np.isfinite(timeout) or not 0 < timeout <= 5:
            raise ValueError('Finite vision startup wait required')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError('Background vision startup failed') from self.error
            if self.count:
                return
            self.stop.wait(.005)
        raise TimeoutError('No committed image before startup deadline')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5.)
        if self.thread.is_alive():
            raise TimeoutError('Vision writer did not terminate; do not reuse its output')
        result = dict(status='failed' if self.error else 'stopped', frames=self.count,
            bytes=self.bytes, write_seconds_percentiles=np.percentile(self.write_seconds,
                [50, 95, 99]).tolist() if self.write_seconds else [],
            error=str(self.error) if self.error else None, worker_alive=False,
            exposure_synchronization_verified=False, actor_stream_created=False,
            training_admissible=False)
        write_json(self.directory / 'result.json', result)
        return result

    def archive_to(self, destination):
        """Copy only a terminated writer's files; verify every hash and retain the source."""
        if self.thread.is_alive() or not (self.directory / 'result.json').exists():
            raise RuntimeError('Close the vision writer before archiving')
        destination = Path(destination)
        shutil.copytree(self.directory, destination)  # Never overwrite an existing archive.
        files = []
        for source in sorted(self.directory.rglob('*')):
            if source.is_file():
                relative = source.relative_to(self.directory)
                expected = sha256(source)
                if sha256(destination / relative) != expected:
                    raise ValueError('Vision archive hash mismatch: ' + relative.as_posix())
                files.append(dict(file=relative.as_posix(), sha256=expected,
                                  bytes=source.stat().st_size))
        receipt = dict(source=str(self.directory), files=files, source_retained=True,
                       all_hashes_verified=True, worker_alive=False)
        write_json(destination / 'collection-receipt.json', receipt)
        return receipt
