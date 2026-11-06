"""Finite asynchronous CSP camera acquisition; consumers choose preceding fresh frames."""

import hashlib
import io
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
from .camera_body_alignment import preserved_source_render_metadata
from .ac_camera_mount import configured_mount, validate_frame_mount
from .ac_capture import WindowsACWindow
from .ac_csp_clock import CSPClockBounds, clock_ping, initialize_clock
from .ac_csp_geometry import matrix
from .ac_metadata_mmf import MetadataMMF
from .camera_receive_observer import CameraReceiveObserver
from .ac_shared_texture import SharedTextureReader
from .evidence import write_json
from .graph import sha256
from .immutable_frame_writer import ImmutableFrameWriter


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


def await_owned_capture_start(start_gate, stop):
    """A prepared reader must remain cancellable before any capture request."""
    if start_gate is None:
        return not stop.is_set()
    while not stop.is_set():
        if start_gate.wait(.01):
            return not stop.is_set()
    return False


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
            if not file.is_file() or file.suffix not in (
                    '.json', '.jsonl', '.png', '.bmp', '.lua', '.mmf'):
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


def validate_source_timing_ledger(value, request_id, generation, *, expected_capture_count=None,maximum_capture_count=1200):
    """Diagnostic source clocks only; never alter camera timestamps or admission limits."""
    if maximum_capture_count not in (1200,1600):raise ValueError('Explicit original1200 or pinned1600 capture scope required')
    if (value.get('schema') != 'flybrain.csp.source_timing.v1'
            or value.get('request_id') != request_id or value.get('generation') != generation
            or value.get('clock_domain') != 'csp_os_preciseClock_seconds'
            or value.get('active_capture_closed') is not True
            or value.get('records_frozen') is not True
            or value.get('flushed_after_capture') is not True
            or value.get('actor_input') is not False
            or value.get('timing_observation_only') is not True
            or value.get('per_frame_timing_disk_writes') != 0
            or value.get('max_records') != 131072
            or value.get('record_fields') != ['stage', 'started_clock', 'finished_clock',
                'duration_seconds', 'succeeded', 'update_index', 'capture_index', 'detail']
            or type(value.get('overflow_records')) is not int or value['overflow_records'] < 0
            or not isinstance(value.get('records'), list)
            or not 1 <= len(value['records']) <= 131072
            or value.get('record_count') != len(value['records'])):
        raise ValueError('Exact closed finite original-clock source timing ledger required')
    quiet = value.get('quiescence', {})
    if (any(quiet.get(key) is not True for key in ['paused', 'camera_inactive',
            'expert_probe_inactive', 'brake_settle_inactive', 'pending_session_receipt_inactive'])
            or quiet.get('ai_controlled') is not False
            or not all(type(quiet.get(key)) in (int, float) and np.isfinite(quiet[key])
                       for key in ['observed_clock', 'speed_kmh', 'gas', 'brake'])
            or abs(quiet['speed_kmh']) >= .1 or quiet['gas'] >= .02 or quiet['brake'] <= .98):
        raise ValueError('Source timing flush requires observed inactive paused brake-held control')
    clocks = [value.get(key) for key in ['started_clock', 'capture_closed_clock', 'frozen_clock',
              'finished_clock']]
    if (not all(type(x) in (int, float) for x in clocks) or not np.isfinite(clocks).all()
            or not 0 <= clocks[0] <= clocks[1] <= clocks[2]
            <= quiet['observed_clock'] <= clocks[3]
            or type(value.get('final_capture_count')) is not int
            or not 0 <= value['final_capture_count'] <= maximum_capture_count
            or expected_capture_count is not None
            and value['final_capture_count'] != expected_capture_count
            or type(value.get('update_index')) is not int or value['update_index'] < 0):
        raise ValueError('Source timing original boundary clocks/capture count differ')
    for row in value['records']:
        if not isinstance(row, list) or len(row) != 8:
            raise ValueError('Source timing fixed column record differs')
        clocks = row[1:4]
        if (not all(type(x) in (int, float) for x in clocks) or not np.isfinite(clocks).all()
                or clocks[0] < 0 or clocks[1] < clocks[0]
                or abs(clocks[1]-clocks[0]-clocks[2]) > 1e-10
                or not isinstance(row[0], str) or not row[0] or type(row[4]) is not bool
                or type(row[5]) is not int or not 0 <= row[5] <= value['update_index']
                or row[6] is not False and (type(row[6]) is not int
                    or not 0 <= row[6] <= value['final_capture_count'])
                or not value['started_clock'] <= clocks[0] <= clocks[1] <= value['frozen_clock']):
            raise ValueError('Source timing stage clock/duration/outcome differs')
    return dict(status='complete' if value['overflow_records'] == 0 else 'overflow',
                records=len(value['records']), overflow_records=value['overflow_records'],
                clock_domain=value['clock_domain'], per_frame_timing_disk_writes=0,
                actor_input=False)


def copy_shared_group(gpu, frame, directory, *, host_clock=time.perf_counter, batched=False,
                      ack_writer=None,defer_destination_check=False):
    """Own all four CPU images before releasing source textures; persist them later.

    Disk writes, hashes and preprocessing must not sit between texture copies.
    The acknowledgment permits only source reuse, not actor publication. The caller
    still validates every image before publishing its actual receipt. The opt-in
    v2 stream defers persistence and requires a closed, independently verified journal.
    """
    directory = Path(directory)
    if type(defer_destination_check) is not bool or defer_destination_check and (not batched or ack_writer is None):
        raise ValueError('Deferred destination check requires explicit batched MMF ACK ownership')
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
                or not defer_destination_check and (directory/view['file']).exists()):
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
    try:
        if ack_writer is None:
            write_json(directory/f"ack-{frame['index']:04d}.json", ack)
        else:
            ack_writer(ack)
    except Exception as error:
        if batched and getattr(gpu, 'last_group_trace', None) is not None:
            gpu.last_group_trace.update(error='ACK write: '+str(error), failed_at=host_clock())
        raise
    ack_written = host_clock()
    for copied in receipts:
        copied['ack_write_finished_at'] = ack_written
    return pixels, receipts, ack


class CSPVisionRecorder:
    def __init__(self, hwnd, config, *, max_frames=600, period=.03, max_bytes=512*1024**2,
                 pixel_preprocessor=None, image_format='png', verify_png_reference=False,
                 shared_transport=False, async_pixel_precompute=False, batched_shared_copy=False,
                 async_frame_persistence=False, mmf_metadata=False, source_timing_v1=False,
                 frame_writer_factory=None, preserve_producer_time=False,
                 expected_source_sha256=None, start_gate=None, prepared_event=None,
                 source_body_recording=False,lifecycle_only_camera_status=False,mmf_stop_signal=False,
                 memory_only_receive_checks=False,expected_track="monza"):
        if expected_track not in ("monza", "magione") or (expected_track=="magione"
                and (expected_source_sha256 is None or not mmf_metadata)):
            raise ValueError("An explicit source-pinned MMF camera is required for Magione")
        self.expected_track = expected_track
        if any(event is not None and not isinstance(event, threading.Event)
                for event in (start_gate, prepared_event)):
            raise ValueError('Explicit cancellable capture preparation events required')
        self.start_gate, self.prepared_event = start_gate, prepared_event
        if type(mmf_stop_signal) is not bool or mmf_stop_signal and not mmf_metadata:
            raise ValueError('Explicit owned stop signal requires MMF recording')
        self.mmf_stop_signal,self.stop_channel=mmf_stop_signal,None
        self.source_body_recording=source_body_recording
        if type(memory_only_receive_checks) is not bool or memory_only_receive_checks and not (mmf_metadata and async_frame_persistence and mmf_stop_signal):
            raise ValueError('Memory-only receive checks require owned MMF stop and exclusive deferred persistence')
        self.memory_only_receive_checks=memory_only_receive_checks
        if type(lifecycle_only_camera_status) is not bool or lifecycle_only_camera_status and not mmf_metadata:
            raise ValueError('Explicit lifecycle camera status requires MMF recording')
        if (type(source_body_recording) is not bool or source_body_recording and
                (not mmf_metadata or expected_source_sha256 is None or period < .09
                 or max_frames > 1600)):
            raise ValueError('Source-body recording requires pinned finite 1600-frame 90ms MMF scope')
        if (expected_source_sha256 is not None and
                (not isinstance(expected_source_sha256, str) or len(expected_source_sha256) != 64
                 or any(c not in '0123456789abcdef' for c in expected_source_sha256))):
            raise ValueError('Explicit lowercase SHA-256 producer pin required')
        if type(preserve_producer_time) is not bool:
            raise ValueError('Explicit boolean producer-clock preservation required')
        self.preserve_producer_time = preserve_producer_time
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
        if (type(async_frame_persistence) is not bool or async_frame_persistence and
                (not batched_shared_copy or not shared_transport or verify_png_reference)):
            raise ValueError('Async persistence requires batched immutable BMP without reference')
        self.async_frame_persistence = async_frame_persistence
        if frame_writer_factory is not None and (
                not callable(frame_writer_factory) or not async_frame_persistence):
            raise ValueError('Explicit callable frame writer requires async persistence')
        self.frame_writer_factory = frame_writer_factory
        if (type(mmf_metadata) is not bool or mmf_metadata and
                (not async_frame_persistence or not batched_shared_copy or
                 max_frames > (1600 if source_body_recording else 1200))):
            raise ValueError('MMF stream requires bounded1200-frame batched async persistence')
        self.mmf_metadata = mmf_metadata
        if (type(source_timing_v1) is not bool or source_timing_v1 and
                (not mmf_metadata or max_frames*period+5 > (150 if source_body_recording else 60))):
            raise ValueError('Source timing requires finite MMF scope:60s normal or150s pinned source-body recording')
        self.source_timing_v1 = source_timing_v1
        self.camera_mount = configured_mount(config)
        if self.camera_mount is not None and not mmf_metadata:
            raise ValueError('Exterior-front stream requires explicit reviewed MMF v3 transport')
        self.unpersisted_mmf = None
        self.frame_writer = None
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
        expected_source = (expected_source_sha256 or
                           sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'))
        if sha256(self.source) != expected_source:
            raise ValueError('Deployed camera app differs from source')
        self.request_id = ('mmf-stream-' if mmf_metadata else 'async-') + uuid.uuid4().hex
        self.directory = self.root / self.request_id
        self.local = self.root / 'client-recordings' / self.request_id
        self.local.mkdir(parents=True, exist_ok=False)
        shutil.copy2(self.source, self.local / 'deployed-camera.lua')
        self.request = dict(id=self.request_id, frames=max_frames, period_seconds=period,
            deadline_seconds=max_frames*period+5, exposure=.35,
            image_format=image_format, verify_png_reference=verify_png_reference,
            shared_transport=shared_transport,
            mmf_metadata=mmf_metadata,
            host_persistence_protocol='immutable_ram_v3' if mmf_metadata else None,
            expected_generation=status['generation'])
        if source_body_recording:
            self.request['source_body_recording_v1']=True
        if source_timing_v1:
            self.request['source_timing_v1'] = True
        if lifecycle_only_camera_status:
            self.request['camera_status_policy'] = 'mmf_lifecycle_only_v1'
        if mmf_stop_signal:self.request['mmf_stop_signal_v1']=True
        if self.camera_mount is not None:
            self.request['camera_sensor_layout'] = self.camera_mount
        self.config, self.max_bytes = config, max_bytes
        self.pixel_preprocessor = pixel_preprocessor
        self.host_clock = time.perf_counter
        self.clock = CSPClockBounds(status['generation'],
            host_quantization=time.get_clock_info('perf_counter').resolution)
        self.frames, self.pixels = deque(maxlen=16), {}
        self.copy_journals = []  # Bounded by request frames; serialize only after capture ends.
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.count, self.bytes, self.error, self.triggered = 0, 0, None, False
        self.terminal, self.final_clock = None, None
        self.thread = threading.Thread(target=self._run, name='csp-camera-reader', daemon=True)
        self.thread.start()

    def _make_frame_writer(self):
        factory = (ImmutableFrameWriter if self.frame_writer_factory is None
                   else self.frame_writer_factory)
        return factory(self.directory, self.local/'stream.jsonl', clock=self.host_clock,
                       mmf_metadata=self.mmf_metadata)

    def _run(self):
        gpu = None
        precompute = None
        channel = None
        receive_observer = None
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
            if self.prepared_event is not None:
                self.prepared_event.set()
            if not await_owned_capture_start(self.start_gate, self.stop):
                return
            if self.mmf_metadata:
                channel = MetadataMMF(self.request_id, self.clock.generation, clock=self.host_clock,stop_signal=self.mmf_stop_signal)
                self.stop_channel=channel
            if self.memory_only_receive_checks:
                receive_observer=CameraReceiveObserver(self.native,self.window,self.directory/'result.json',clock=self.host_clock)
            write_json(self.root / 'request.json', self.request)
            self.triggered = True
            deadline = time.monotonic()+self.request['deadline_seconds']+2
            with (self.local / 'stream.jsonl').open('x', encoding='utf8') as stream:
                if self.async_frame_persistence:
                    self.frame_writer = self._make_frame_writer()
                for index in range(self.request['frames']):
                    phases = dict(frame_wait_started_at=self.host_clock())
                    path = self.directory / f'frame-{index:04d}.json'
                    packet = None
                    while True:
                        packet = channel.poll(index) if channel else None
                        if packet is not None or channel is None and path.exists():
                            break
                        terminal_path = self.directory / 'result.json'
                        observed_terminal=(receive_observer.snapshot()['terminal'] if receive_observer is not None else
                            read_json(terminal_path) if terminal_path.exists() else None)
                        if observed_terminal is not None:
                            self.terminal = observed_terminal
                            if self.terminal['status'] not in ('stopped', 'completed'):
                                raise ValueError('CSP producer failed: ' + str(self.terminal))
                            if self.terminal['frames'] != self.count:
                                raise ValueError('CSP terminal frame count disagrees '
                                                 'with received data')
                            return
                        if time.monotonic() > deadline:
                            raise TimeoutError('CSP producer deadline exceeded')
                        if receive_observer is None and self.native.probe(self.window.hwnd,
                                             require_foreground=False) != self.window:
                            raise ValueError('Original engine changed')
                        time.sleep(.002)
                    phases['metadata_mmf_seen_at' if channel else 'frame_path_seen_at'] = (
                        self.host_clock())
                    if channel:
                        frame, raw_metadata, mmf_receipt = packet
                        self.unpersisted_mmf = dict(index=index, raw=raw_metadata,
                            frame_slot=channel.last_frame_slot, receipt=mmf_receipt)
                    else:
                        frame = read_json(path)
                    phases['metadata_loaded_at'] = self.host_clock()
                    if (frame['request_id'] != self.request_id or frame['index'] != index
                            or frame.get('generation') != self.clock.generation
                            or frame['car_id'] != 'bmw_z4_gt3' or frame['track_id'] != self.expected_track):
                        raise ValueError('Camera identity, generation or layout changed')
                    validate_frame_aid_state(frame)
                    validate_frame_mount(frame, self.config)
                    phases['metadata_validated_at'] = self.host_clock()
                    if self.shared_transport != (frame.get('transport') == 'shared_immutable_v1'):
                        raise ValueError('Requested immutable shared transport differs')
                    views, pixels, size, shared_copies = [], {}, (
                        len(raw_metadata) if channel else path.stat().st_size), []
                    shared_pixels, shared_ack = {}, None
                    metadata_files = []
                    if self.shared_transport:
                        if self.frame_writer is not None:
                            self.frame_writer.check_capacity()
                        def acknowledge(ack):
                            self.unpersisted_mmf['ack_raw'] = channel.acknowledge(ack)
                            self.unpersisted_mmf['ack_slot'] = channel.last_ack_slot

                        shared_pixels, shared_copies, shared_ack = copy_shared_group(
                            gpu, frame, self.directory, host_clock=self.host_clock,
                            batched=self.batched_shared_copy,
                            ack_writer=acknowledge if channel else None,
                            defer_destination_check=self.memory_only_receive_checks)
                        if channel:
                            slots = channel.last_frame_slot+channel.last_ack_slot
                            mmf_receipt['slots_sha256'] = hashlib.sha256(slots).hexdigest()
                            metadata_files = [(path.name, raw_metadata),
                                (f'ack-{index:04d}.json', self.unpersisted_mmf['ack_raw']),
                                (f'mmf-{index:04d}.mmf', slots)]
                            size += len(metadata_files[1][1])+len(slots)
                    persistence_started = self.host_clock()
                    if self.batched_shared_copy:
                        phases['ack_write_finished_at'] = shared_copies[0]['ack_write_finished_at']
                        phases['persistence_started_at'] = persistence_started
                        journal = dict(index=index, copies=shared_copies, acknowledged=shared_ack,
                                       phases=phases, status='copied_not_yet_published')
                        assert len(self.copy_journals) < self.request['frames']
                        self.copy_journals.append(journal)
                    immutable_files = []
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
                            if self.frame_writer is not None:
                                encoded = io.BytesIO()
                                Image.fromarray(rgb).save(encoded, format='BMP')
                                content = encoded.getvalue()
                                immutable_files.append((file.name, content))
                            else:
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
                        size += (len(content) if self.frame_writer is not None
                                 else file.stat().st_size)
                        receipt_view = dict(view_id=name, file='vision/'+file.name,
                            file_sha256=hashlib.sha256(content).hexdigest()
                                if self.frame_writer is not None else sha256(file),
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            render_started_lower=self.clock.render_start_lower(
                                view['render_started_clock'], self.clock.generation))
                        witness = preserved_source_render_metadata(view)
                        if witness is not None:
                            receipt_view['source_render_metadata'] = witness
                        if self.preserve_producer_time:
                            source_time = view['render_started_clock']
                            if (isinstance(source_time, bool) or not np.isfinite(source_time)
                                    or source_time < 0):
                                raise ValueError('Original finite producer render clock required')
                            receipt_view['producer_render_started'] = source_time
                        views.append(receipt_view)
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
                    phases['immutable_memory_ready_at' if self.frame_writer is not None
                           else 'persistence_finished_at'] = received
                    row = dict(source_kind='csp_four_camera', capture_id=index,
                        generation=self.clock.generation, received=received,
                        ideal_line_shown=frame['ideal_line_shown'],
                        aid_probe_only=frame['aid_probe_only'],
                        file='vision/'+path.name,
                        file_sha256=hashlib.sha256(raw_metadata).hexdigest() if channel
                            else sha256(path), views=views)
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
                    if self.frame_writer is not None:
                        row['shared_pipeline'].update(
                            revision='copy_all_then_ack_then_async_persist_v2',
                            stream_schema='flybrain.csp.immutable_rgb_stream.v2',
                            immutable_memory_ready_clock=received,
                            actor_published_only_after_persistence=False,
                            persistence_required_before_run_admission=True)
                        row['shared_pipeline'].pop('persistence_finished_clock')
                        if channel:
                            row.update(metadata_transport='mmf_slot_sha256_v1',
                                       metadata_mmf_receipt=mmf_receipt)
                            row['shared_pipeline'].update(
                                revision='copy_all_then_mmf_ack_then_async_persist_v3',
                                stream_schema='flybrain.csp.immutable_rgb_stream.v3')
                        self.frame_writer.submit(index, immutable_files,
                            (json.dumps(row, allow_nan=False)+'\n').encode('utf8'),
                            metadata_files=metadata_files)
                        self.unpersisted_mmf = None
                        phases['persistence_enqueued_at'] = self.host_clock()
                    else:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        stream.flush()
                        phases['event_log_flushed_at'] = self.host_clock()
                    with self.lock:
                        if len(self.frames) == self.frames.maxlen:
                            self.pixels.pop(self.frames[0]['capture_id'], None)
                        self.frames.append(row)
                        self.pixels[index] = pixels
                        self.count += 1
                        self.bytes += size
                    phases['actor_published_at'] = self.host_clock()
                    if self.batched_shared_copy:
                        journal['status'] = 'published'
                while True:
                    terminal_path=self.directory/'result.json'
                    observed_terminal=(receive_observer.snapshot()['terminal'] if receive_observer is not None else
                        read_json(terminal_path) if terminal_path.exists() else None)
                    if observed_terminal is not None:
                        self.terminal=observed_terminal;break
                    if time.monotonic() > deadline:
                        raise TimeoutError('No terminal CSP producer result')
                    time.sleep(.005)
                if self.terminal['status'] not in ('stopped', 'completed'):
                    raise ValueError('CSP producer failed')
        except Exception as error:
            self.error = error
        finally:
            if channel:
                if self.error is not None and self.mmf_stop_signal:
                    write_json(self.local/'owned-stop-signal.json',json.loads(channel.request_stop()))
                if self.error and channel.last_observed_slot is not None:
                    try:
                        with (self.local/'mmf-failure-slot.mmf').open('xb') as file:
                            file.write(channel.last_observed_slot)
                    except Exception as error:
                        self.error = self.error or error
                channel.close()
            if receive_observer is not None:
                proof=receive_observer.close()
                write_json(self.local/'receive-side-observations.json',proof)
                if proof['worker_alive'] or proof['error']:
                    self.error=self.error or RuntimeError('Original camera side observer did not close cleanly')
            if self.unpersisted_mmf is not None:
                pending = self.unpersisted_mmf
                self.directory.mkdir(exist_ok=True)
                for name, content in [(f'frame-{pending["index"]:04d}.json', pending['raw']),
                        (f'incomplete-mmf-{pending["index"]:04d}.mmf',
                         pending['frame_slot']+pending.get('ack_slot', b''))]:
                    try:
                        with (self.directory/name).open('xb') as file:
                            file.write(content)
                    except Exception as error:
                        self.error = self.error or error
                if 'ack_raw' in pending:
                    try:
                        with (self.directory/f'ack-{pending["index"]:04d}.json').open('xb') as file:
                            file.write(pending['ack_raw'])
                    except Exception as error:
                        self.error = self.error or error
                try:
                    write_json(self.local/'incomplete-mmf-receipt.json', pending['receipt'])
                except Exception as error:
                    self.error = self.error or error
            if self.frame_writer is not None:
                try:
                    self.frame_writer.close()
                except Exception as error:
                    self.error = self.error or error
                try:
                    write_json(self.local/'frame-persistence-journal.json',
                               self.frame_writer.statistics())
                except Exception as error:
                    self.error = self.error or error
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
            if self.batched_shared_copy:
                try:
                    write_json(self.local/'batch-copy-journal.json', dict(
                        schema='flybrain.shared_copy.journal.v2',
                        serialized_only_after_capture_loop=True,
                        maximum_groups=self.request['frames'],
                        groups=self.copy_journals))
                except Exception as error:
                    self.error = self.error or error
            if self.triggered and self.error is not None and not self.mmf_stop_signal:
                self.directory.mkdir(exist_ok=True)
                (self.directory / 'stop').touch(exist_ok=True)
            write_json(self.local / 'consumer-result.json', dict(
                status='failed' if self.error else 'ended', frames=self.count, bytes=self.bytes,
                error=str(self.error) if self.error else None, producer_terminal=self.terminal,
                batched_shared_copy=self.batched_shared_copy,
                async_frame_persistence=self.async_frame_persistence,
                mmf_metadata=self.mmf_metadata,
                pixel_precompute=precompute.statistics() if precompute else None))

    def candidate_before(self, receipt, *, max_age=.15):
        if self.frame_writer is not None and self.frame_writer.error is not None:
            raise RuntimeError('CSP immutable frame persistence failed') from (
                self.frame_writer.error)
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
            samples = {v['view_id']: dict(rgb=pixels[v['view_id']],
                capture_id=candidate['capture_id'], generation=candidate['generation'],
                received=candidate['received'], render_started_lower=v['render_started_lower'])
                for v in candidate['views']}
            if self.preserve_producer_time:
                for view in candidate['views']:
                    samples[view['view_id']]['producer_render_started'] = (
                        view['producer_render_started'])
            return samples

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

    def request_stop(self):
        self.stop.set()
        if self.triggered and not (self.directory / 'result.json').exists():
            if self.mmf_stop_signal:
                if self.stop_channel is None:raise RuntimeError('Original stop channel unavailable; no file fallback')
                write_json(self.local/'owned-stop-signal.json',json.loads(self.stop_channel.request_stop()))
            else:
                self.directory.mkdir(exist_ok=True)
                (self.directory / 'stop').touch(exist_ok=True)

    def close(self):
        self.request_stop()
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
        source_timing = None
        if self.source_timing_v1 and self.triggered:
            ledger_path = self.directory / 'source-timing-v1.json'
            source_timing = dict(status='failed', producer_file=str(ledger_path),
                                 archived_file='vision/source-timing-v1.json')
            deadline = time.monotonic()+10  # same owned cleanup; never another control request
            try:
                while not ledger_path.exists() and time.monotonic() < deadline:
                    time.sleep(.005)
                # io.save(..., true) publishes by temporary-file rename. Hash the
                # exact opened bytes that were parsed, rather than reopening a path.
                content = ledger_path.read_bytes()
                source_timing.update(validate_source_timing_ledger(
                    json.loads(content), self.request_id, self.clock.generation,
                    expected_capture_count=self.terminal.get('frames'),
                    maximum_capture_count=1600 if self.source_body_recording else 1200),
                    sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
                if source_timing['status'] == 'overflow' and self.error is None:
                    self.error = RuntimeError('Source timing overflow; incomplete call evidence')
            except (OSError, ValueError) as error:
                source_timing['error'] = str(error)
                if self.error is None:
                    self.error = RuntimeError('Source timing proof unavailable: '+str(error))
            write_json(self.local/'source-timing-proof.json', source_timing)
        for _ in range(2):
            clock_ping(self.root, self.clock, self.window.hwnd, host_clock=self.host_clock)
        self.final_clock = self.clock.record()
        write_json(self.local / 'clock-final.json', self.final_clock)
        result = dict(status='failed' if self.error else 'stopped', frames=self.count,
            bytes=self.bytes, error=str(self.error) if self.error else None,
            worker_alive=False, producer_terminal=self.terminal,
            clock_uncertainty_seconds=self.final_clock['uncertainty_seconds'],
            actor_stream_created=False, physical_exposure_sync_verified=False)
        if source_timing is not None:
            result['source_timing'] = source_timing
        return result

    def archive_to(self, destination, *, bundle=False):
        if self.thread.is_alive():
            raise RuntimeError('Cannot archive a live camera consumer')
        writer = bundle_camera_files if bundle else archive_camera_files
        return writer((self.directory, self.local), destination)
