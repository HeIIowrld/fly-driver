# Read-only Assetto Corsa Python app. Keep syntax compatible with embedded Python 3.3.
# No imports of the learner, no vJoy dependency and no control/reset commands.
import json
import math
import os
import sys
import time

PROTOCOL = "flybrain.ac.telemetry.v1"
PORT = 2348


class SourceNativeSnapshot(object):
    """Existing mapping only; bracket SDK reads with equal native prefix copies.

    QPC is an OS clock, not the physical tick or camera exposure time. Matching
    copies are a consistency witness, not an engine-provided atomicity contract.
    """
    SIZE = 308

    def __init__(self):
        import ctypes
        import base64
        from ctypes import wintypes
        self.ctypes, self.base64 = ctypes, base64
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        api.OpenFileMappingW.restype = wintypes.HANDLE
        api.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, ctypes.c_size_t]
        api.MapViewOfFile.restype = ctypes.c_void_p
        api.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        api.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        self.api = api
        value = ctypes.c_longlong()
        if not api.QueryPerformanceFrequency(ctypes.byref(value)) or value.value <= 0:
            raise RuntimeError('QPC frequency unavailable')
        self.frequency = value.value
        self.handle = api.OpenFileMappingW(4, False, 'Local\\acpmf_physics')
        if not self.handle:
            raise RuntimeError('Existing native physics mapping unavailable')
        self.view = api.MapViewOfFile(self.handle, 4, 0, 0, self.SIZE)
        if not self.view:
            api.CloseHandle(self.handle)
            self.handle = None
            raise RuntimeError('Read-only native physics mapping failed')

    def ticks(self):
        value = self.ctypes.c_longlong()
        if not self.api.QueryPerformanceCounter(self.ctypes.byref(value)):
            raise RuntimeError('QPC read failed')
        return value.value

    def capture(self, ac, acsys):
        for attempt in range(4):
            started = self.ticks()
            before = self.ctypes.string_at(self.view, self.SIZE)
            fields = read_body(ac, acsys)
            fields['privileged_teacher'] = read_privileged_teacher(ac, acsys)
            after = self.ctypes.string_at(self.view, self.SIZE)
            finished = self.ticks()
            if before == after and 0 <= finished-started <= self.frequency*.005:
                fields['source_native_snapshot'] = dict(
                    protocol='flybrain.ac.source_native.v1',
                    prefix_base64=self.base64.b64encode(after).decode('ascii'),
                    qpc_started_ticks=started, qpc_finished_ticks=finished,
                    qpc_frequency_hz=self.frequency, attempts=attempt+1,
                    repeated_copy_equal=True, physical_tick_time_verified=False,
                    camera_exposure_sync_verified=False)
                return fields
        raise RuntimeError('SDK/native same-prefix 5ms bracket unavailable')

    def close(self):
        if self.view:
            self.api.UnmapViewOfFile(self.view)
            self.view = None
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class EmbeddedGcTimings(object):
    """Diagnostic callback only; never disable or force the engine interpreter GC."""
    def __init__(self):
        import gc
        self.gc = gc
        self.started = None
        self.count, self.maximum, self.last = 0, 0., None
        self.callback = self.on_collection
        self.available = hasattr(gc, 'callbacks')
        if self.available:
            gc.callbacks.append(self.callback)

    def on_collection(self, phase, info):
        now = time.perf_counter()
        if phase == 'start':
            self.started = now
        elif phase == 'stop' and self.started is not None:
            elapsed = now-self.started
            self.count += 1
            self.maximum = max(self.maximum, elapsed)
            self.last = (self.started, now, elapsed, info.get('generation'))
            self.started = None

    def snapshot(self):
        return dict(schema='flybrain.telemetry.embedded_gc_timing.v1',
            actor_input=False, diagnostic_callback_available=self.available,
            collector_behavior_changed=False, count=self.count,
            maximum_seconds=self.maximum, last=self.last)

    def close(self):
        if self.available and self.callback in self.gc.callbacks:
            self.gc.callbacks.remove(self.callback)


class StageTimings(object):
    """Constant-size, latched host timings; never a simulation/exposure clock.

    The next packet reports completed calls, including failed calls. A maximum
    survives later fast calls, so latest-only publication does not erase a hitch.
    """
    def __init__(self):
        self.stages = {}

    def record(self, name, started, succeeded):
        finished = time.perf_counter()
        sample = dict(started_host_seconds=started, finished_host_seconds=finished,
                      duration_seconds=finished-started, succeeded=succeeded)
        previous = self.stages.get(name)
        self.stages[name] = dict(count=1 if previous is None else previous['count']+1,
            failures=(0 if previous is None else previous['failures'])+int(not succeeded),
            last=sample, maximum=sample if previous is None or sample['duration_seconds']
            > previous['maximum']['duration_seconds'] else previous['maximum'])

    def snapshot(self):
        return dict(schema='flybrain.telemetry.host_stage_timing.v1', actor_input=False,
                    clock='time.perf_counter', reports_completed_calls_only=True,
                    stages=dict(self.stages))


class Publisher(object):
    def __init__(self, port=PORT):
        # Stock AC may omit the native _socket extension. Do not fail app import.
        global socket, threading
        import socket
        import threading
        self.session = 'ac-tcp-{}-{}'.format(os.getpid(), int(time.time()*1000000))
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Embedded AC omits the IDNA codec. Bytes keep this numeric loopback
        # address out of CPython's Unicode-hostname/IDNA conversion path.
        self.server.bind((b"127.0.0.1", port))
        self.server.listen(1)
        self.server.settimeout(.2)
        self.port = self.server.getsockname()[1]
        self.stopped = threading.Event()
        self.updated = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.counter = 0
        self.thread = threading.Thread(target=self._serve)
        self.thread.daemon = True
        self.thread.start()

    def publish(self, fields):
        self.counter += 1
        fields = dict(fields)
        fields.update(steps=self.counter, timestamp_ac=time.perf_counter(),
                      producer_session_id=self.session)
        wire = (json.dumps(fields, allow_nan=False) + "\n").encode("utf8")
        with self.lock:
            self.latest = (self.counter, wire)
            self.updated.set()

    def _serve(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(.2)
                hello = dict(protocol=PROTOCOL, kind="hello", read_only=True,
                             producer_session_id=self.session, steps=0,
                             timestamp_ac=time.perf_counter(), simulation_clock_verified=False,
                             image_exposure_metadata_available=False)
                connection.sendall((json.dumps(hello) + "\n").encode("utf8"))
                previous = None
                while not self.stopped.is_set():
                    with self.lock:
                        latest = self.latest
                        self.updated.clear()
                    if latest is not None and latest[0] != previous:
                        connection.sendall(latest[1])
                        previous = latest[0]
                    self.updated.wait(.02)
            except (OSError, socket.timeout):
                pass
            finally:
                connection.close()

    def close(self):
        self.stopped.set()
        self.updated.set()
        self.server.close()
        self.thread.join(1)


class InlineLoopbackPublisher(Publisher):
    """Nonblocking loopback delivery at the actual source callback.

    Only accept/hello run in the background. A short/busy socket send closes the
    connection explicitly; no packet fragment is retried as a different sample.
    The callback never waits for a receiver, disk, or a sender thread's GIL turn.
    """
    def __init__(self, port=PORT):
        self.connection = None
        self.delivery_failures = 0
        Publisher.__init__(self, port=port)

    def publish(self, fields):
        self.counter += 1
        row = dict(fields)
        row.update(steps=self.counter, timestamp_ac=time.perf_counter(),
            producer_session_id=self.session,
            publisher_tcp_delivery='inline_nonblocking_loopback_v1',
            publisher_tcp_delivery_failures=self.delivery_failures)
        wire = (json.dumps(row, allow_nan=False)+'\n').encode('utf8')
        with self.lock:
            self.latest = (self.counter, wire)
            connection = self.connection
            if connection is not None:
                try:
                    if connection.send(wire) != len(wire):
                        raise OSError('Partial nonblocking loopback packet; close framing explicitly')
                except (OSError, socket.timeout):
                    self.delivery_failures += 1
                    self.connection = None
                    connection.close()

    def _serve(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                connection.settimeout(.2)
                hello=dict(protocol=PROTOCOL,kind='hello',read_only=True,
                    producer_session_id=self.session,steps=0,timestamp_ac=time.perf_counter(),
                    simulation_clock_verified=False,image_exposure_metadata_available=False,
                    publisher_tcp_delivery='inline_nonblocking_loopback_v1')
                connection.sendall((json.dumps(hello)+'\n').encode('utf8'))
                connection.setblocking(False)
                with self.lock:
                    self.connection = connection
                while not self.stopped.wait(.02):
                    with self.lock:
                        if self.connection is not connection:
                            break
            except (OSError,socket.timeout):
                pass
            finally:
                with self.lock:
                    if self.connection is connection:
                        self.connection = None
                connection.close()


class FilePublisher(object):
    """Bounded latest-only local handoff without native networking extensions."""

    def __init__(self, directory=None):
        if directory is None:
            directory = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                                     "flybrain", "telemetry")
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.session = "ac-file-{}-{}".format(os.getpid(), int(time.time() * 1000000))
        self.counter = 0
        self.closed = False
        self.replace_retries = 0
        self.timings = StageTimings()
        hello = dict(protocol=PROTOCOL, kind="hello", read_only=True,
                     producer_session_id=self.session, steps=0,
                     timestamp_ac=time.perf_counter(), simulation_clock_verified=False,
                     image_exposure_metadata_available=False, transport="local_latest_file")
        self._write("latest.json", hello)
        self._write("producer.json", hello)

    def _write(self, name, fields):
        target = os.path.join(self.directory, name)
        temporary = target + ".partial"
        started, succeeded = time.perf_counter(), False
        try:
            payload = (json.dumps(fields, allow_nan=False) + "\n").encode("utf8")
            succeeded = True
        finally:
            self.timings.record('serialize', started, succeeded)
        started, succeeded = time.perf_counter(), False
        try:
            with open(temporary, "wb") as stream:
                stream.write(payload)
            succeeded = True
        finally:
            self.timings.record('open_write_close', started, succeeded)
        # A Python reader can briefly deny replacement on Windows. Retry the same
        # already serialized measurement only; never read fresh SDK values or re-date it.
        deadline = time.perf_counter() + .003
        started, succeeded = time.perf_counter(), False
        try:
            while True:
                try:
                    os.replace(temporary, target)
                    succeeded = True
                    break
                except PermissionError as error:
                    if (getattr(error, 'winerror', None) not in (5, 32, 33)
                            or time.perf_counter() >= deadline):
                        raise
                    self.replace_retries += 1
                    time.sleep(.0005)
        finally:
            self.timings.record('replace_including_retry', started, succeeded)

    def prepare(self, fields):
        if self.closed:
            raise ValueError("File telemetry publisher is closed")
        self.counter += 1
        row = dict(fields)
        row.update(steps=self.counter, timestamp_ac=time.perf_counter(),
                   producer_session_id=self.session,
                   publisher_replace_retries=self.replace_retries,
                   publisher_replace_retry_limit_seconds=.003,
                   publisher_previous_io_timing=self.timings.snapshot())
        return row

    def publish(self, fields):
        self._write("latest.json", self.prepare(fields))

    def close(self):
        self.closed = True


class LatestFilePublisher(object):
    """One pending immutable measurement; disk I/O never holds the callback lock."""

    def __init__(self, directory=None):
        import threading
        import copy
        self.copy = copy.deepcopy
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stopped = False
        self.pending = None
        self.suspended = False
        self.writing = False
        self.dropped = 0
        self.errors = 0
        self.last_error = None
        self.file = FilePublisher(directory)
        self.thread = threading.Thread(target=self._serve)
        self.thread.daemon = True
        self.thread.start()

    def publish(self, fields):
        # SDK values and timestamp belong to this callback, never the disk worker.
        frozen = self.copy(fields)
        with self.lock:
            if self.suspended:
                raise RuntimeError('Compatibility file publication suspended during TCP')
            if self.stopped:
                raise ValueError('Latest file publisher is closed')
            if self.pending is not None:
                self.dropped += 1
            row = self.file.prepare(frozen)
            row.update(publisher_delivery='bounded_latest_worker_v1',
                       publisher_dropped_before_write=self.dropped,
                       publisher_background_errors=self.errors,
                       publisher_background_last_error=self.last_error)
            self.pending = row
            self.ready.set()

    def _serve(self):
        try:
            while True:
                self.ready.wait(.1)
                with self.lock:
                    if self.stopped:
                        return
                    row, self.pending = self.pending, None
                    if self.suspended:
                        row = None
                    self.writing = row is not None
                    self.ready.clear()
                if row is not None:
                    try:
                        self.file._write('latest.json', row)
                    except Exception as error:
                        with self.lock:
                            self.errors += 1
                            self.last_error = str(error)
                    finally:
                        with self.lock:
                            self.writing = False
        finally:
            self.file.close()

    def suspend_for_tcp(self, active):
        """Clear queued work, then expose the actual in-flight completion fence."""
        with self.lock:
            self.suspended = active
            if active and self.pending is not None:
                self.pending = None
                self.dropped += 1
                self.ready.clear()
            return active and not self.writing

    def close(self):
        with self.lock:
            self.stopped = True
            self.pending = None
            self.ready.set()
        self.thread.join(1)


class HybridPublisher(object):
    """50Hz TCP measurements; low-rate file compatibility outside the driving path.

    The source session is shared for calibration identity. Each transport has its
    own sequence; consumers must never combine sequence spaces inside an episode.
    """
    def __init__(self):
        self.tcp = InlineLoopbackPublisher()
        try:
            self.file = LatestFilePublisher()
        except BaseException:
            self.tcp.close()
            raise
        self.tcp.session = self.file.file.session
        self.last_file = -1.

    def publish(self, fields):
        connected = self.tcp.connection is not None
        quiescent = self.file.suspend_for_tcp(connected)
        row = dict(fields, publisher_delivery='tcp_no_file_while_connected_v2',
            publisher_file_compatibility_suspended=connected,
            publisher_file_io_quiescent=quiescent,
            publisher_compatibility_completed_io_timing=self.file.file.timings.snapshot())
        self.tcp.publish(row)
        now = time.perf_counter()
        if not connected and now-self.last_file >= .1:
            self.file.publish(row)
            self.last_file = now

    def close(self):
        try:
            self.tcp.close()
        finally:
            self.file.close()


def read_body(ac, acsys):
    """Raw SDK values only; native units/axes/steering semantics require measurement."""
    result = {}
    groups = [("LocalAngularVelocity", ["angular_velocity_x", "angular_velocity_y",
                                       "angular_velocity_z"]),
              ("WheelAngularSpeed", ["wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl",
                                     "wheel_speed_rr"]),
              ("SuspensionTravel", ["fl_damper_linear_potentiometer",
                 "fr_damper_linear_potentiometer", "rl_damper_linear_potentiometer",
                 "rr_damper_linear_potentiometer"]),
              ("CurrentTyresCoreTemp", ["fl_tire_temperature_core", "fr_tire_temperature_core",
                                         "rl_tire_temperature_core", "rr_tire_temperature_core"]),
              ("Load", ["fl_wheel_load", "fr_wheel_load", "rl_wheel_load", "rr_wheel_load"])]
    for source, names in groups:
        try:
            values = ac.getCarState(0, getattr(acsys.CS, source))
            if len(values) != len(names):
                raise ValueError("Unexpected SDK width")
            for index, key in enumerate(names):
                value = values[index]
                result[key] = float(value) if math.isfinite(value) else None
        except Exception:
            for key in names:
                result[key] = None
    for source, name in [("Steer", "steerAngle"), ("Gas", "accStatus"),
                         ("Brake", "brakeStatus"), ("SpeedMS", "speed"), ("LapCount", "LapCount")]:
        try:
            value = ac.getCarState(0, getattr(acsys.CS, source))
            result[name] = value if math.isfinite(value) else None
        except Exception:
            result[name] = None
    return result


def read_privileged_teacher(ac, acsys):
    """Pinned AC Gym raw SDK conventions, isolated from the fly actor's body fields.

    Local/world SDK vectors are reordered as upstream structures.Car.update does.
    These virtual-track positions/progress are teacher/evaluation data only.
    """
    result = {}
    groups = [
        ('WorldPosition', [('world_position_x', 2), ('world_position_y', 0),
                           ('world_position_z', 1)]),
        ('LocalVelocity', [('local_velocity_x', 2), ('local_velocity_y', 0),
                           ('local_velocity_z', 1)]),
        ('AccG', [('accelX', 0), ('accelY', 2)]),
        ('SlipAngle', [('SlipAngle_fl', 0), ('SlipAngle_fr', 1),
                       ('SlipAngle_rl', 2), ('SlipAngle_rr', 3)]),
    ]
    for source, destinations in groups:
        try:
            values = ac.getCarState(0, getattr(acsys.CS, source))
            for key, index in destinations:
                value = values[index]
                result[key] = float(value) if math.isfinite(value) else None
        except Exception:
            for key, _ in destinations:
                result[key] = None
    for source, key in [('Gear', 'actualGear'), ('RPM', 'RPM'), ('LastFF', 'LastFF'),
                        ('NormalizedSplinePosition', 'NormalizedSplinePosition'),
                        ('LapInvalidated', 'LapInvalidated')]:
        try:
            value = ac.getCarState(0, getattr(acsys.CS, source))
            result[key] = value if math.isfinite(value) else None
        except Exception:
            result[key] = None
    try:
        length = ac.getTrackLength(0)
        result['track_length_m'] = float(length) if math.isfinite(length) else None
    except Exception:
        result['track_length_m'] = None
    return dict(protocol='flybrain.ac.privileged_teacher.v1', actor_input=False,
                raw=result, source='AC Gym structures.Car.update SDK ordering')


_publisher = None
_native_snapshot = None
_ac = None
_acsys = None
_elapsed = 0.
_clock_candidate_valid = True
_last_emit = -1.
_publish_errors = 0
_last_error_log = -10.
_update_timings = StageTimings()
_startup_diagnostic = {}
_gc_timings = None


def acMain(ac_version):
    global _publisher, _ac, _acsys, _elapsed, _clock_candidate_valid, _last_emit
    global _publish_errors, _last_error_log, _update_timings
    global _startup_diagnostic, _native_snapshot, _gc_timings
    import ac
    import acsys
    _ac, _acsys = ac, acsys
    if _publisher is not None:
        _publisher.close()
    _publisher = None
    _elapsed, _clock_candidate_valid, _last_emit = 0., True, -1.
    _publish_errors, _last_error_log = 0, -10.
    _update_timings = StageTimings()
    if _gc_timings is not None:
        _gc_timings.close()
    _gc_timings = EmbeddedGcTimings()
    _startup_diagnostic = dict(source_file=__file__, python_version=sys.version,
        source_path=list(sys.path), working_directory=os.getcwd(), executable=sys.executable)
    extension_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime33')
    if extension_dir not in sys.path:
        sys.path.insert(0, extension_dir)
    try:
        _native_snapshot = SourceNativeSnapshot()
        _startup_diagnostic['source_native_snapshot'] = 'qpc_same_prefix_v1'
    except Exception as error:
        _startup_diagnostic['source_native_snapshot_error'] = str(error)
        _native_snapshot = None
    try:
        app_window = ac.newApp("FlybrainTelemetry")
        ac.setSize(app_window, 240, 60)
        try:
            # Only this reviewed embedded-Python extension directory is added.
            # It contains the pinned upstream Python3.3 AMD64 _socket.pyd.
            extension_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime33')
            _startup_diagnostic.update(extension_dir=extension_dir,
                socket_extension_exists=os.path.isfile(os.path.join(extension_dir, '_socket.pyd')))
            if extension_dir not in sys.path:
                sys.path.insert(0, extension_dir)
            _publisher = HybridPublisher()
            _startup_diagnostic['transport'] = 'hybrid_tcp50_file10'
            ac.log("Flybrain read-only TCP50/file10 telemetry on localhost:2348")
        except Exception as error:
            _startup_diagnostic.update(transport='file_only_fallback',
                tcp_startup_error=str(error), tcp_error_type=type(error).__name__,
                extension_dir=extension_dir)
            try:
                _publisher = LatestFilePublisher()
            except ImportError:
                _publisher = FilePublisher()
            ac.log("Flybrain read-only local file telemetry enabled; TCP import unavailable: "
                   + str(error))
    except Exception as error:
        ac.log("Flybrain telemetry startup failed: " + str(error))
    return "FlybrainTelemetry"


def acUpdate(deltaT):
    global _elapsed, _clock_candidate_valid, _last_emit, _publish_errors, _last_error_log
    if _publisher is None:
        return
    if not math.isfinite(deltaT) or deltaT < 0:
        _clock_candidate_valid = False
    else:
        _elapsed += deltaT
    now = time.perf_counter()
    if now - _last_emit < .02:
        return
    _last_emit = now
    started = time.perf_counter()
    try:
        fields = (_native_snapshot.capture(_ac, _acsys) if _native_snapshot is not None
                  else read_body(_ac, _acsys))
    except Exception:
        _publish_errors += 1
        return  # A failed bracket is a real missing source sample, never repaired.
    _update_timings.record('sdk_body', started, True)
    started = time.perf_counter()
    if 'privileged_teacher' not in fields:
        fields['privileged_teacher'] = read_privileged_teacher(_ac, _acsys)
    _update_timings.record('sdk_teacher', started, True)
    fields.update(ac_update_delta_seconds=deltaT if math.isfinite(deltaT) else None,
                  elapsed_seconds_candidate=_elapsed,
                  clock_candidate_valid=_clock_candidate_valid,
                  simulation_clock_verified=False, publisher_errors=_publish_errors,
                  publisher_update_timing=_update_timings.snapshot())
    fields['publisher_startup_diagnostic'] = _startup_diagnostic
    fields['embedded_gc_timing'] = _gc_timings.snapshot() if _gc_timings is not None else None
    started, succeeded = time.perf_counter(), False
    try:
        _publisher.publish(fields)
        succeeded = True
    except Exception as error:
        # Embedded AC can crash while formatting an uncaught Python traceback.
        # A transient Windows replace/sharing failure must remain a telemetry gap.
        _publish_errors += 1
        if now - _last_error_log >= 5:
            _last_error_log = now
            try:
                _ac.log('Flybrain telemetry publish failed; retry on next update: ' + str(error))
            except Exception:
                pass
    finally:
        _update_timings.record('publish', started, succeeded)
    if isinstance(_publisher, HybridPublisher):
        # AC can retain the interpreter between short callbacks. Explicitly yield
        # the GIL to the awakened sender instead of waiting for a later frame.
        started = time.perf_counter()
        time.sleep(0)
        _update_timings.record('sender_yield', started, True)


def acShutdown():
    global _publisher, _native_snapshot, _gc_timings
    if _gc_timings is not None:
        _gc_timings.close()
        _gc_timings = None
    if _publisher is not None:
        _publisher.close()
        _publisher = None
    if _native_snapshot is not None:
        _native_snapshot.close()
        _native_snapshot = None
