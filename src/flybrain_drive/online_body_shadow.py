"""Read-only online camera/native receipt shadow; never invokes an actor or control."""

import base64
import hashlib
import json
import threading
import time
from pathlib import Path

from .ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from .per_view_body_history import PerViewBodyHistory

VIEW_ORDER = ('front', 'left', 'right', 'rear')


class OnlineBodyShadow:
    """Own one sequential consumer; the sampler only appends immutable rows."""

    def __init__(self, *, directory, request_id, generation, offset_lower, native_rows,
                 native_identity, clock=time.perf_counter, neural_shadow=None):
        self.directory, self.request_id, self.generation = Path(directory), request_id, generation
        self.offset_lower, self.native_rows, self.clock = offset_lower, native_rows, clock
        self.history = PerViewBodyHistory(episode_id=request_id, native_identity=native_identity)
        self.neural_shadow = neural_shadow
        self.events, self.position, self.index, self.error = [], 0, 0, None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='per-view-body-shadow', daemon=True)

    def start(self):
        self.thread.start()

    def _drain_native(self):
        # Snapshot the published array length. This same consumer drains it before
        # its decision timestamp, so future worker activity cannot enter the input.
        end = len(self.native_rows)
        while self.position < end:
            row = self.native_rows[self.position]
            raw = base64.b64decode(row['authority']['prefix_base64'], validate=True)
            receipt = row['copy']['second_copy_end']
            if (hashlib.sha256(raw).hexdigest() != row['authority']['prefix_sha256']
                    or receipt != row['authority']['received_monotonic_seconds']):
                raise ValueError('Original native copy hash and receipt required')
            sample = decode_physics_prefix(raw[:PHYSICS_PREFIX.size], receipt)
            if sample.packet_id != row['authority']['packet_id']:
                raise ValueError('Original native packet identity required')
            self.history.ingest_native_physics(sample)
            self.position += 1

    def _select_context(self, event, *, camera_receipt, bounds):
        drain_started = self.clock()
        drain_thread_started = time.thread_time()
        drain_process_started = time.process_time()
        self._drain_native()
        decision = self.clock()
        drain_thread_finished = time.thread_time()
        drain_process_finished = time.process_time()
        event['native_drain_started'] = drain_started
        event['native_drain_finished'] = decision
        event['native_drain_thread_cpu_seconds'] = drain_thread_finished-drain_thread_started
        event['native_drain_process_cpu_seconds'] = drain_process_finished-drain_process_started
        ctx = self.history.build(render_start_lowers=bounds, camera_receipt_time=camera_receipt,
                                 decision_time=decision)
        context_ready = self.clock()
        event.update(decision=decision, context_ready=context_ready,
                     native_rows_consumed=self.position,
                     status='missing_current_body' if ctx is None else 'context_built')
        if ctx is not None:
            event['decision_body'] = {'packet': ctx.decision_body.packet_id,
                'receipt': ctx.decision_body.original_received_monotonic_seconds,
                'values': ctx.decision_body.values.tolist(), 'valid': ctx.decision_body.valid.tolist()}
            event['view_body'] = [None if body is None else {'packet': body.packet_id,
                'receipt': body.original_received_monotonic_seconds,
                'values': body.values.tolist(), 'valid': body.valid.tolist()}
                for body in ctx.visual_contexts]
        return ctx

    def consume(self, frame, *, camera_receipt):
        if (frame.get('index') != self.index or frame.get('request_id') != self.request_id
                or frame.get('airflow_diagnostic', {}).get('generation') != self.generation):
            raise ValueError('Original ordered frame and camera generation required')
        views = {v['view_id']: v for v in frame['views']}
        if set(views) != set(VIEW_ORDER) or len(frame['views']) != 4:
            raise ValueError('Four ordered original camera views required')
        bounds = tuple(views[name]['render_started_clock']+self.offset_lower for name in VIEW_ORDER)
        event = {'index': self.index, 'camera_receipt': camera_receipt,
                 'render_start_lower': bounds}
        try:
            if self.neural_shadow is not None and hasattr(self.neural_shadow, 'process_late'):
                event['visual_snapshot'] = self.clock()
                def select_after_vision():
                    return self._select_context(event, camera_receipt=camera_receipt, bounds=bounds)
                event['neural'] = self.neural_shadow.process_late(
                    frame, event, select_after_vision, self.directory)
            else:
                ctx = self._select_context(event, camera_receipt=camera_receipt, bounds=bounds)
                if self.neural_shadow is not None:
                    event['neural'] = self.neural_shadow.process(frame, event, ctx, self.directory)
        except Exception as error:
            event['neural_error'] = {'type': type(error).__name__, 'reason': str(error)}
            self.events.append(event)
            self.index += 1
            raise
        self.events.append(event)
        self.index += 1
        return event

    def _run(self):
        try:
            while True:
                file = self.directory/f'frame-{self.index:04d}.json'
                if file.exists():
                    raw = file.read_bytes()
                    frame = json.loads(raw)
                    receipt = self.clock()
                    event = self.consume(frame, camera_receipt=receipt)
                    event['metadata_sha256'] = hashlib.sha256(raw).hexdigest()
                elif self.stop_event.is_set():
                    return
                else:
                    time.sleep(.002)
        except Exception as error:
            self.error = {'type': type(error).__name__, 'reason': str(error)}

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError('Original body shadow thread remains live')
