"""Finite asynchronous CSP camera acquisition; consumers choose preceding fresh frames."""

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .ac_camera_archive import bundle_camera_files
from .ac_capture import WindowsACWindow
from .ac_csp_clock import CSPClockBounds, clock_ping, initialize_clock
from .ac_csp_geometry import camera_geometry, matrix
from .ac_shared_texture import SharedTextureReader
from .evidence import write_json
from .graph import sha256


def read_json(path, *, timeout=.02, clock=time.monotonic, sleep=time.sleep):
    """Retry only an incomplete metadata publication; never alter its source clocks."""
    if isinstance(timeout, bool) or not np.isfinite(timeout) or not 0 <= timeout <= .02:
        raise ValueError('Metadata publication retry bound must be0..20ms')
    deadline = clock()+timeout
    while True:
        try:
            return json.loads(Path(path).read_text(encoding='utf8'))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            remaining = deadline-clock()
            if remaining <= 0:
                raise
            sleep(min(.001, remaining))


class NoFreshCSPFrame(ValueError):
    """A healthy source has not published a preceding frame within the age budget."""


def validate_frame_aid_state(frame):
    """Require explicit producer evidence before publishing pixels to actor consumers.

    Missing legacy metadata is unknown, never proof that an aid was absent. Diagnostic
    captures remain ineligible even when the ideal line happened to be switched off.
    """
    if frame.get('ideal_line_shown') is not False or frame.get('aid_probe_only') is not False:
        raise ValueError('Actor camera requires explicit ideal-line OFF and non-diagnostic frame')


def candidate_at(rows, receipt, max_age):
    if (not np.isfinite([receipt, max_age]).all() or receipt < 0 or not 0 < max_age <= 1):
        raise ValueError('Finite receipt and explicit freshness budget required')
    candidates = [row for row in rows if row['received'] <= receipt]
    row = max(candidates, key=lambda r: r['capture_id']) if candidates else None
    ages = [] if row is None else [receipt-v['render_started_lower'] for v in row['views']]
    if any(not np.isfinite(age) or age < 0 for age in ages):
        raise ValueError('CSP source render time is invalid or follows selection')
    if row is None or any(age > max_age for age in ages):
        raise NoFreshCSPFrame('No complete preceding CSP frame within conservative age budget')
    return row


def archive_camera_files(sources, destination):
    """Copy terminal local camera files with bounded I/O parallelism and one hash readback."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files, names = [], set()
    for source in sources:
        source = Path(source)
        if not source.exists():
            continue
        for file in sorted(source.iterdir()):
            if not file.is_file() or file.suffix not in ('.json', '.jsonl', '.png', '.bmp', '.lua'):
                continue
            if file.name in names:
                raise ValueError('Source archive filename collision')
            names.add(file.name)
            files.append(file)

    def copy_verified(file):
        expected, size = sha256(file), file.stat().st_size
        target = destination/file.name
        # Source mtime/permissions are not part of the camera data contract. Avoid
        # remote metadata writes and duplicate readbacks for each small PNG.
        shutil.copyfile(file, target)
        if sha256(target) != expected:
            raise ValueError('CSP archive copy hash mismatch')
        return dict(file=file.name, sha256=expected, bytes=size)

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='csp-archive') as pool:
        records = list(pool.map(copy_verified, files))
    write_json(destination/'archive.json', records)
    return dict(all_hashes_verified=True, files=len(records), source_retained=True,
                worker_alive=False)


def validate_camera_budget(max_frames, period, max_bytes):
    if (type(max_frames) is not int or not 1 <= max_frames <= 4000
        or not np.isfinite(period) or not .03 <= period <= .25
        or max_frames*period+5 > 150 or type(max_bytes) is not int
        or not 1024 <= max_bytes <= 2*1024**3):
        raise ValueError('Finite camera frames, period, duration and bytes required')


def copy_shared_group(gpu, frame, directory, *, host_clock=time.perf_counter, batched=False):
    """Own all four CPU images before releasing source textures; persist them later.

    Disk writes, hashes and preprocessing must not sit between texture copies.
    The acknowledgment permits only source reuse, not actor publication. The caller
    still validates and persists every image before publishing its actual receipt.
    """
    directory = Path(directory)
    if type(batched) is not bool:
        raise ValueError('Explicit batched copy flag required')
    views = frame['views']
    if len(views) != 4 or {v['view_id'] for v in views} != {'front', 'left', 'right', 'rear'}:
        raise ValueError('Four distinct shared camera views required')
    pixels, receipts = {}, []
    for view in views:
        name = view['view_id']
        if (view['file'] != f"{frame['index']:04d}-{name}.bmp"
                or view.get('image_writer') != 'host_lossless_bmp'
                or (directory/view['file']).exists()):
            raise ValueError('Shared RGB destination already exists or differs')
    copies = (gpu.read_group([v['shared_texture_handle'] for v in views]) if batched else
              [gpu.read(v['shared_texture_handle']) for v in views])
    if len(copies) != 4:
        raise ValueError('Four complete shared copies required before acknowledgment')
    for view, (rgb, copied) in zip(views, copies, strict=True):
        name = view['view_id']
        if rgb.shape != (180, 320, 3) or rgb.dtype != np.uint8:
            raise ValueError('Exact shared RGB image shape and dtype required')
        rgb.setflags(write=False)
        pixels[name] = rgb
        receipts.append(dict(copied, view_id=name))
    ack = dict(request_id=frame['request_id'], generation=frame['generation'],
               capture_id=frame['index'], copied_before_ack=host_clock())
    write_json(directory/f"ack-{frame['index']:04d}.json", ack)
    return pixels, receipts, ack


class CSPVisionRecorder:
    def __init__(self, hwnd, config, *, max_frames=600, period=.03, max_bytes=512*1024**2,
                 pixel_preprocessor=None, image_format='png', verify_png_reference=False,
                 shared_transport=False, async_pixel_precompute=False, batched_shared_copy=False):
        validate_camera_budget(max_frames, period, max_bytes)
        if (image_format not in ('png', 'bmp') or type(verify_png_reference) is not bool
                or verify_png_reference and (image_format != 'bmp' or max_frames > 8)):
            raise ValueError('Explicit lossless format; '
                             'paired reference limited to eight BMP groups')
        if (type(shared_transport) is not bool
                or shared_transport and (image_format != 'bmp' or period < .04)):
            raise ValueError('Shared transport requires explicit lossless BMP and40ms period')
        if (type(async_pixel_precompute) is not bool
                or async_pixel_precompute and (not shared_transport or pixel_preprocessor is None)):
            raise ValueError('Async cache requires immutable shared transport and pure callback')
        self.async_pixel_precompute = async_pixel_precompute
        if type(batched_shared_copy) is not bool or batched_shared_copy and not shared_transport:
            raise ValueError('Batched shared copy requires explicit immutable shared transport')
        self.batched_shared_copy = batched_shared_copy
        self.image_format, self.verify_png_reference = image_format, verify_png_reference
        self.shared_transport = shared_transport
        self.native = WindowsACWindow()
        self.window = self.native.probe(hwnd, require_foreground=False)
        self.root = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-camera'
        status = read_json(self.root / 'status.json')
        if (time.time()-(self.root / 'status.json').stat().st_mtime > 3
                or status.get('patch_version') != '0.2.11' or status.get('state') == 'capturing'
                or not status.get('generation')):
            raise ValueError('Reviewed idle camera producer heartbeat required')
        self.source = (Path(self.window.executable).parent
                       / 'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')
        if sha256(self.source) != sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
            raise ValueError('Deployed camera app differs from source')
        self.request_id = 'async-' + uuid.uuid4().hex
        self.directory = self.root / self.request_id
        self.local = self.root / 'client-recordings' / self.request_id
        self.local.mkdir(parents=True, exist_ok=False)
        shutil.copy2(self.source, self.local / 'deployed-camera.lua')
        self.request = dict(id=self.request_id, frames=max_frames, period_seconds=period,
            deadline_seconds=max_frames*period+5, exposure=.35,
            image_format=image_format, verify_png_reference=verify_png_reference,
            shared_transport=shared_transport,
            expected_generation=status['generation'])
        self.config, self.max_bytes = config, max_bytes
        self.pixel_preprocessor = pixel_preprocessor
        self.host_clock = time.perf_counter
        self.clock = CSPClockBounds(status['generation'],
            host_quantization=time.get_clock_info('perf_counter').resolution)
        self.frames, self.pixels = deque(maxlen=16), {}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.count, self.bytes, self.error, self.triggered = 0, 0, None, False
        self.terminal, self.final_clock = None, None
        self.thread = threading.Thread(target=self._run, name='csp-camera-reader', daemon=True)
        self.thread.start()

    def _run(self):
        gpu = None
        precompute = None
        try:
            if self.async_pixel_precompute:
                from .ac_pixel_precompute import PixelPrecomputeQueue
                precompute = PixelPrecomputeQueue(self.pixel_preprocessor)
            initialize_clock(self.root, self.clock, self.window.hwnd, host_clock=self.host_clock)
            if self.shared_transport:
                gpu = SharedTextureReader.reviewed()
                write_json(self.local/'shared-device.json', gpu.adapter)
            write_json(self.local / 'clock-initial.json', self.clock.record())
            write_json(self.local / 'host-clock.json', dict(
                implementation=time.get_clock_info('perf_counter').implementation,
                resolution=time.get_clock_info('perf_counter').resolution,
                timestamp_domain='host_monotonic_QPC',
                all_sensor_receipts_must_use_same_clock=True))
            if self.clock.high-self.clock.low > .1:
                raise ValueError('Initial clock uncertainty too wide')
            if self.stop.is_set():
                return
            write_json(self.root / 'request.json', self.request)
            self.triggered = True
            deadline = time.monotonic()+self.request['deadline_seconds']+2
            with (self.local / 'stream.jsonl').open('x', encoding='utf8') as stream:
                for index in range(self.request['frames']):
                    path = self.directory / f'frame-{index:04d}.json'
                    while not path.exists():
                        terminal_path = self.directory / 'result.json'
                        if terminal_path.exists():
                            self.terminal = read_json(terminal_path)
                            if self.terminal['status'] not in ('stopped', 'completed'):
                                raise ValueError('CSP producer failed: ' + str(self.terminal))
                            if self.terminal['frames'] != self.count:
                                raise ValueError('CSP terminal frame count disagrees '
                                                 'with received data')
                            return
                        if time.monotonic() > deadline:
                            raise TimeoutError('CSP producer deadline exceeded')
                        if self.native.probe(self.window.hwnd,
                                             require_foreground=False) != self.window:
                            raise ValueError('Original engine changed')
                        time.sleep(.002)
                    frame = read_json(path)
                    if (frame['request_id'] != self.request_id or frame['index'] != index
                            or frame.get('generation') != self.clock.generation
                            or frame.get('camera_layout_revision') != 2
                            or frame['car_id'] != 'bmw_z4_gt3' or frame['track_id'] != 'monza'):
                        raise ValueError('Camera identity, generation or layout changed')
                    validate_frame_aid_state(frame)
                    camera_geometry(frame['views'])
                    if self.shared_transport != (frame.get('transport') == 'shared_immutable_v1'):
                        raise ValueError('Requested immutable shared transport differs')
                    views, pixels, size, shared_copies = [], {}, path.stat().st_size, []
                    shared_pixels, shared_ack = {}, None
                    if self.shared_transport:
                        shared_pixels, shared_copies, shared_ack = copy_shared_group(
                            gpu, frame, self.directory, host_clock=self.host_clock,
                            batched=self.batched_shared_copy)
                    persistence_started = self.host_clock()
                    if self.batched_shared_copy:
                        write_json(self.directory/f'copy-{index:04d}.json',
                                   dict(copies=shared_copies, acknowledged=shared_ack))
                    for view in frame['views']:
                        name = view['view_id']
                        p = matrix(view['projection_matrix_diagnostic'])
                        expected = self.config['views'][name]
                        if (view['file'] != f'{index:04d}-{name}.{self.image_format}'
                                or view.get('image_format', 'png') != self.image_format
                                or not np.isclose(p[0, 0], expected['projection_x'], atol=1e-6)
                                or not np.isclose(p[1, 1], expected['projection_y'], atol=1e-6)
                                or view['requested_exposure'] != .35):
                            raise ValueError('Camera pixels/projection/exposure differs '
                                             'from contract')
                        file = self.directory / view['file']
                        if self.shared_transport:
                            rgb = shared_pixels[name]
                            Image.fromarray(rgb).save(file, format='BMP')
                        else:
                            with Image.open(file) as raw:
                                rgb = np.array(raw.convert('RGB'))
                        if rgb.shape != (180, 320, 3):
                            raise ValueError('CSP image shape changed')
                        if self.verify_png_reference and not self.shared_transport:
                            expected_reference = f'{index:04d}-{name}-reference.png'
                            if view.get('reference_png') != expected_reference:
                                raise ValueError('Paired PNG reference filename differs')
                            reference = self.directory/expected_reference
                            with Image.open(reference) as raw:
                                reference_rgb = np.array(raw.convert('RGB'))
                            if not np.array_equal(rgb, reference_rgb):
                                raise ValueError('Same-shot BMP/PNG RGB differs')
                            size += reference.stat().st_size
                        rgb.setflags(write=False)
                        if self.pixel_preprocessor is not None:
                            if precompute is not None:
                                precompute.submit(rgb)
                            else:
                                self.pixel_preprocessor(rgb)
                        pixels[name] = rgb
                        size += file.stat().st_size
                        views.append(dict(view_id=name, file='vision/'+file.name,
                            file_sha256=sha256(file),
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            render_started_lower=self.clock.render_start_lower(
                                view['render_started_clock'], self.clock.generation)))
                    if self.bytes+size > self.max_bytes:
                        raise ValueError('CSP camera storage budget exhausted')
                    if self.shared_transport:
                        if self.verify_png_reference:
                            reference_receipt = self.directory/f'reference-{index:04d}.json'
                            reference_deadline = min(deadline, time.monotonic()+2)
                            while not reference_receipt.exists():
                                if time.monotonic() >= reference_deadline:
                                    raise TimeoutError('Same-shot post-copy PNG reference missing')
                                time.sleep(.002)
                            confirmed = read_json(reference_receipt)
                            if (confirmed['stage'] != 'after_host_copy'
                                    or confirmed['request_id'] != self.request_id
                                    or confirmed['capture_id'] != index
                                    or confirmed['generation'] != self.clock.generation):
                                raise ValueError('Post-copy PNG reference identity differs')
                            for name, rgb in pixels.items():
                                reference = self.directory/f'{index:04d}-{name}-reference.png'
                                with Image.open(reference) as raw:
                                    reference_rgb = np.array(raw.convert('RGB'))
                                if not np.array_equal(rgb, reference_rgb):
                                    raise ValueError('Shared RGB differs from '
                                                     'post-copy same-shot PNG')
                                size += reference.stat().st_size
                            if self.bytes+size > self.max_bytes:
                                raise ValueError('CSP paired reference storage budget exhausted')
                    received = self.host_clock()
                    row = dict(source_kind='csp_four_camera', capture_id=index,
                        generation=self.clock.generation, received=received,
                        ideal_line_shown=frame['ideal_line_shown'],
                        aid_probe_only=frame['aid_probe_only'],
                        file='vision/'+path.name, file_sha256=sha256(path), views=views)
                    if self.shared_transport:
                        row['shared_copy_receipts'] = shared_copies
                        row['shared_pipeline'] = dict(
                            revision='copy_all_then_ack_then_persist_v1',
                            acknowledged_copy_clock=shared_ack['copied_before_ack'],
                            persistence_started_clock=persistence_started,
                            persistence_finished_clock=received,
                            actor_published_only_after_persistence=True,
                            pixel_precompute_policy='bounded_async_cache_only' if precompute
                                else 'synchronous_before_publication')
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                    with self.lock:
                        if len(self.frames) == self.frames.maxlen:
                            self.pixels.pop(self.frames[0]['capture_id'], None)
                        self.frames.append(row)
                        self.pixels[index] = pixels
                        self.count += 1
                        self.bytes += size
                while not (self.directory / 'result.json').exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('No terminal CSP producer result')
                    time.sleep(.005)
                self.terminal = read_json(self.directory / 'result.json')
                if self.terminal['status'] not in ('stopped', 'completed'):
                    raise ValueError('CSP producer failed')
        except Exception as error:
            self.error = error
        finally:
            if precompute is not None:
                try:
                    precompute.close()
                except Exception as error:
                    self.error = self.error or error
            if gpu is not None:
                trace = getattr(gpu, 'last_group_trace', None)
                if self.batched_shared_copy and trace and trace.get('error'):
                    try:
                        write_json(self.local/'shared-copy-failure.json', trace)
                    except Exception as error:
                        self.error = self.error or error
                gpu.close()
            if self.triggered and self.error is not None:
                self.directory.mkdir(exist_ok=True)
                (self.directory / 'stop').touch(exist_ok=True)
            write_json(self.local / 'consumer-result.json', dict(
                status='failed' if self.error else 'ended', frames=self.count, bytes=self.bytes,
                error=str(self.error) if self.error else None, producer_terminal=self.terminal,
                batched_shared_copy=self.batched_shared_copy,
                pixel_precompute=precompute.statistics() if precompute else None))

    def candidate_before(self, receipt, *, max_age=.15):
        if self.error is not None:
            raise RuntimeError('CSP camera consumer failed') from self.error
        with self.lock:
            row = candidate_at(self.frames, receipt, max_age)
            return dict(row, views=[dict(v) for v in row['views']],
                host_age_upper_seconds=[receipt-v['render_started_lower'] for v in row['views']])

    def wait_for_candidate(self, *, max_age=.11, timeout=.02):
        """Wait only for new camera data; preserve the age limit and fail on producer errors."""
        if not np.isfinite(timeout) or not 0 <= timeout <= .04:
            raise ValueError('Finite camera wait timeout in 0..40ms required')
        started = self.host_clock()
        deadline = started+timeout
        while True:
            selected = self.host_clock()
            if timeout > 0 and selected > deadline:
                raise NoFreshCSPFrame('CSP camera wait deadline exceeded')
            try:
                candidate = self.candidate_before(selected, max_age=max_age)
                finished = self.host_clock()
                if timeout > 0 and finished > deadline:
                    raise NoFreshCSPFrame('CSP camera wait deadline exceeded')
                return candidate, selected, finished-started
            except NoFreshCSPFrame:
                now = self.host_clock()
                if now >= deadline:
                    raise
                time.sleep(min(.001, deadline-now))

    def samples_for(self, candidate):
        validate_frame_aid_state(candidate)
        with self.lock:
            pixels = self.pixels.get(candidate['capture_id'])
            if pixels is None or candidate['generation'] != self.clock.generation:
                raise ValueError('Requested camera candidate expired from owned cache')
            return {v['view_id']: dict(rgb=pixels[v['view_id']],
                capture_id=candidate['capture_id'], generation=candidate['generation'],
                received=candidate['received'], render_started_lower=v['render_started_lower'])
                for v in candidate['views']}

    def wait_ready(self, *, timeout=4.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if self.error:
                raise RuntimeError('CSP stream startup failed') from self.error
            try:
                self.candidate_before(self.host_clock())
                return
            except ValueError:
                time.sleep(.005)
        raise TimeoutError('No fresh CSP camera group before startup deadline')

    def close(self):
        self.stop.set()
        if self.triggered and not (self.directory / 'result.json').exists():
            self.directory.mkdir(exist_ok=True)
            (self.directory / 'stop').touch(exist_ok=True)
        self.thread.join(timeout=5.)
        if self.thread.is_alive():
            raise RuntimeError('CSP camera consumer still running; do not restart or archive')
        if self.triggered:
            deadline = time.monotonic()+2
            while not (self.directory / 'result.json').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Producer terminal state unknown; do not archive or restart')
                time.sleep(.005)
            self.terminal = read_json(self.directory / 'result.json')
        for _ in range(2):
            clock_ping(self.root, self.clock, self.window.hwnd, host_clock=self.host_clock)
        self.final_clock = self.clock.record()
        write_json(self.local / 'clock-final.json', self.final_clock)
        return dict(status='failed' if self.error else 'stopped', frames=self.count,
            bytes=self.bytes, error=str(self.error) if self.error else None,
            worker_alive=False, producer_terminal=self.terminal,
            clock_uncertainty_seconds=self.final_clock['uncertainty_seconds'],
            actor_stream_created=False, physical_exposure_sync_verified=False)

    def archive_to(self, destination, *, bundle=False):
        if self.thread.is_alive():
            raise RuntimeError('Cannot archive a live camera consumer')
        writer = bundle_camera_files if bundle else archive_camera_files
        return writer((self.directory, self.local), destination)
