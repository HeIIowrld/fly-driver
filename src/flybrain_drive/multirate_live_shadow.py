"""Independent body-rate flight shadow and four-view FlyVis producer; no actuator."""

import base64
import hashlib
import io
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from .flight_stream import held_input_batches
from .flyvis_camera import VIEW_ORDER
from .graph import sha256
from .immutable_camera_group import freeze_camera_group, require_matching_camera_geometry
from .multirate_flight_stream import MultirateFlightStream
from .online_neural_late_body import LateBodyNeuralShadow, output_host_freshness
from .virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


class MultirateLiveShadow(LateBodyNeuralShadow):
    """One vision producer and one body-only owner of the persistent flight state."""

    def _visual_scope(self):
        return nullcontext()

    def _body_scope(self):
        return nullcontext()

    def _body_device_scope(self):
        return torch.device('cuda')

    def _prepare_body_scene(self, scene):
        return scene

    def _body_thread_time(self):
        return None

    def _sync_visual(self):
        torch.cuda.synchronize()

    def _sync_body(self):
        torch.cuda.synchronize()

    def _publish_visual(self, scene, context, frame_index):
        started = time.perf_counter()
        deadline = started+.05
        with self.mail_condition:
            while self.mail is not None and self.mail_version != self.consumed_version:
                if self.ticker_error is not None or self.ticker_stop_reason is not None:
                    raise RuntimeError('Body ticker stopped before pending visual was consumed')
                remaining = deadline-time.perf_counter()
                if remaining <= 0:
                    raise RuntimeError('Previous visual scene remains unconsumed after 50ms')
                self.mail_condition.wait(remaining)
            if self.ticker_error is not None or self.ticker_stop_reason is not None:
                raise RuntimeError('Body ticker stopped before visual publication')
            published = time.perf_counter()
            if max(published-x for x in context.render_start_lowers) > .15:
                raise ValueError('Original visual scene became stale before publication')
            self.mail_version += 1
            self.mail = (self.mail_version, scene, context, published, frame_index)
            return published, published-started

    def _acknowledge_visual(self, version):
        with self.mail_condition:
            if self.mail_version != version:
                raise RuntimeError('Visual publication changed before consumption')
            self.consumed_version = version
            self.mail_condition.notify_all()

    def attach(self, *, native_rows, native_identity, episode_id, body_period=.02):
        if not .01 <= body_period <= .04 or hasattr(self, 'ticker'):
            raise ValueError('One bounded original body ticker required')
        self.native_rows, self.period = native_rows, body_period
        self.body_encoder = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                                  native_identity=native_identity)
        self.episode_id = episode_id
        self.multirate = MultirateFlightStream(self.stream)
        self.mail_lock, self.mail = threading.Lock(), None
        self.mail_condition = threading.Condition(self.mail_lock)
        self.mail_version, self.consumed_version = 0, 0
        self.ticker_events, self.ticker_error, self.ticker_stop_reason = [], None, None
        self.ticker_stop_details = None
        self.stop_event = threading.Event()
        self.ticker = threading.Thread(target=self._body_loop, name='flight-body-ticker', daemon=True)
        self.ticker.start()

    def enable_diagnostic_cuda_timing(self):
        if hasattr(self, 'ticker'):
            raise ValueError('Diagnostic timing must be enabled before ticker start')
        self.diagnostic_cuda_timing = True

    def configure_immutable(self, source_camera_config, config_path):
        """Pin camera compatibility before the owned producer begins."""
        if self.failed or hasattr(self, '_immutable_generation') or self.previous_time is not None:
            raise ValueError('Immutable source must be configured once before visual processing')
        if sha256(config_path) != self.provenance['config_sha256']:
            raise ValueError('Exact initialized shadow config required')
        config = json.loads(Path(config_path).read_text(encoding='utf8'))
        if sha256(config['camera_config']) != config['camera_config_sha256']:
            raise ValueError('Exact initialized camera configuration required')
        camera = json.loads(Path(config['camera_config']).read_text(encoding='utf8'))
        require_matching_camera_geometry(source_camera_config, camera)
        self._immutable_generation, self._immutable_capture = None, None
        self._immutable_signature = None

    def process_immutable(self, candidate, samples, select_after_vision):
        """Admit one original shared-camera group; no actuator and no replay on failure."""
        if self.failed:
            raise RuntimeError('Failed vision shadow cannot be silently resumed')
        try:
            if not hasattr(self, '_immutable_generation'):
                raise ValueError('Explicit compatible camera source must be configured')
            selected = time.perf_counter()
            group = freeze_camera_group(candidate, samples, selected_at=selected)
            signature = (group['received'], tuple((v['rgb_sha256'], v['render_started_lower'])
                                                 for v in group['views']))
            if self._immutable_generation is not None and (
                    group['generation'] != self._immutable_generation
                    or group['capture_id'] < self._immutable_capture):
                raise ValueError('Reversed or restarted camera stream requires stop')
            event = {'index': group['capture_id'], 'camera_receipt': group['received'],
                     'visual_snapshot': selected, 'render_start_lower': [
                         v['render_started_lower'] for v in group['views']],
                     'missing_intermediate_captures': 0 if self._immutable_capture is None
                     else max(0, group['capture_id']-self._immutable_capture-1)}
            if group['capture_id'] == self._immutable_capture:
                if signature != self._immutable_signature:
                    raise ValueError('Repeated capture changed original image or timing')
                return event, {'status': 'duplicate_not_reprocessed', 'vehicle_control': False}
            output = self.process_late(None, event, lambda: select_after_vision(event), None,
                                       _immutable_group=group)
            self._immutable_generation = group['generation']
            self._immutable_capture = group['capture_id']
            self._immutable_signature = signature
            return event, output
        except Exception:
            self.failed = True
            raise

    @torch.no_grad()
    def process_late(self, frame, event, select_after_vision, directory, *, _immutable_group=None):
        if self.failed:
            raise RuntimeError('Failed vision shadow cannot be silently resumed')
        started, visual_time = time.perf_counter(), event['visual_snapshot']
        try:
            if self.previous_time is not None and not 0 < visual_time-self.previous_time <= .15:
                raise ValueError('Original visual gap exceeds 150ms')
            retinal, hashes = [], []
            if _immutable_group is None:
                views = {v['view_id']: v for v in frame['views']}
                for name in VIEW_ORDER:
                    expected = f"{event['index']:04d}-{name}.png"
                    if views[name].get('file') != expected:
                        raise ValueError('Exact original capture filename required')
                    raw = (Path(directory)/expected).read_bytes()
                    hashes.append(hashlib.sha256(raw).hexdigest())
                    with Image.open(io.BytesIO(raw)) as image:
                        retinal.append(self.projector.sample(np.asarray(image.convert('RGB')).copy()))
                hash_kind = 'encoded_original_png'
            else:
                for view in _immutable_group['views']:
                    name = view['view_id']
                    retinal.append(self.projector.sample(_immutable_group['pixels'][name]))
                    hashes.append(view['rgb_sha256'])
                hash_kind = 'immutable_rgb'
            with torch.device('cuda'), self._visual_scope():
                timing = getattr(self, 'diagnostic_cuda_timing', False)
                gpu_start = torch.cuda.Event(enable_timing=True) if timing else None
                gpu_end = torch.cuda.Event(enable_timing=True) if timing else None
                if timing:
                    gpu_start.record()
                current = torch.as_tensor(np.stack(retinal), dtype=torch.float32, device='cuda')
                if self.previous_time is not None:
                    for dt, count in held_input_batches(visual_time-self.previous_time):
                        while count:
                            n = min(count, 16)
                            source = self.held[:, None, None, :].expand(-1, n, -1, -1)
                            self.vis_state = self.network.simulate(source, dt=dt,
                                initial_state=self.vis_state, as_states=True)[-1]
                            count -= n
                motion = self.vis_state.nodes.activity[:, self.motion_indices]
                scene = self.model.backbone.scene_pool(torch.cat((current[:, None], motion), dim=1)[None])
                # Publish only after the copied scene has completed on the producer stream.
                scene_snapshot = scene.detach().clone()
                if timing:
                    gpu_end.record()
                self._sync_visual()
                visual_gpu_span_ms = gpu_start.elapsed_time(gpu_end) if timing else None
                vision_finished = time.perf_counter()
                context = select_after_vision()
                if event['decision'] < vision_finished:
                    raise ValueError('Original body context selected before visual completion')
                published = None
                mailbox_wait = None
                if context is not None:
                    # A bounded wait preserves every scene; no unconsumed replacement.
                    published, mailbox_wait = self._publish_visual(
                        scene_snapshot, context, event['index'])
                self.held, self.previous_time = current.detach().clone(), visual_time
            finished = time.perf_counter()
            return {'status': 'visual_published' if published is not None else 'missing_current_body',
                'intent': None, 'started': started, 'vision_finished': vision_finished,
                'published': published, 'finished': finished,
                'mailbox_wait_seconds': mailbox_wait,
                'processing_seconds': finished-started,
                'visual_cuda_event_span_ms': visual_gpu_span_ms,
                'camera_receipt_to_output_seconds': finished-event['camera_receipt'],
                'body_receipt_age_at_output_seconds': None if context is None else
                    finished-context.decision_body.original_received_monotonic_seconds,
                'view_render_age_upper_at_output_seconds': [finished-x for x in event['render_start_lower']],
                'source_images_sha256': hashes, 'source_images_hash_kind': hash_kind,
                'vehicle_control': False,
                'physical_sample_or_exposure_time_verified': False}
        except Exception:
            self.failed = True
            raise

    def _body_loop(self):
        index, first_packet, latest, previous_id, last_update = 0, None, None, None, None
        try:
            while not self.stop_event.is_set():
                end = len(self.native_rows)
                while index < end:
                    row = self.native_rows[index]
                    raw = base64.b64decode(row['authority']['prefix_base64'], validate=True)
                    t = row['copy']['second_copy_end']
                    if (hashlib.sha256(raw).hexdigest() != row['authority']['prefix_sha256']
                            or t != row['authority']['received_monotonic_seconds']):
                        raise ValueError('Original native bytes/receipt required')
                    sample = decode_physics_prefix(raw[:PHYSICS_PREFIX.size], t)
                    if sample.packet_id != row['authority']['packet_id']:
                        raise ValueError('Original native packet differs')
                    if latest is None:
                        first_packet = sample.packet_id
                        latest = decode_navigation_body_prefix(sample.prefix_bytes, t)
                    elif sample.packet_id < latest.legacy.packet_id:
                        raise ValueError('Native packet reset requires new run')
                    elif sample.packet_id == latest.legacy.packet_id:
                        if hashlib.sha256(sample.prefix_bytes).hexdigest() != latest.legacy.prefix_sha256:
                            raise ValueError('Duplicate native packet changed bytes')
                    else:
                        latest = decode_navigation_body_prefix(sample.prefix_bytes, t)
                    index += 1
                now = time.perf_counter()
                with self.mail_lock:
                    available = self.mail
                if available is None or latest is None or latest.legacy.packet_id == first_packet:
                    time.sleep(.002)
                    continue
                if self.multirate.previous_time is not None and (
                        now-latest.legacy.received_monotonic_seconds > .05):
                    self.ticker_stop_reason = 'current_body_receipt_stale'
                    self.ticker_stop_details = {'time': now, 'packet': latest.legacy.packet_id,
                        'first_receipt': latest.legacy.received_monotonic_seconds,
                        'age_seconds': now-latest.legacy.received_monotonic_seconds,
                        'native_rows_consumed': index}
                    return
                if (latest.legacy.packet_id == previous_id or last_update is not None
                        and now-last_update < self.period):
                    time.sleep(.002)
                    continue
                version, scene, context, published, frame_id = available
                if published > now:
                    raise ValueError('Future scene publication is invalid')
                encode_started = time.perf_counter()
                body = self.body_encoder.encode(latest, episode_id=self.episode_id,
                                                decision_monotonic_seconds=now)
                encode_finished = time.perf_counter()
                new_frame = version != self.consumed_version
                if new_frame:
                    # Acknowledge while holding the publication lock. This update
                    # still uses its immutable copied scene and original context.
                    self._acknowledge_visual(version)
                started = time.perf_counter()
                thread_started = self._body_thread_time()
                with self._body_device_scope(), torch.no_grad(), self._body_scope():
                    timing_enabled = getattr(self, 'diagnostic_cuda_timing', False)
                    gpu_start = torch.cuda.Event(enable_timing=True) if timing_enabled else None
                    gpu_end = torch.cuda.Event(enable_timing=True) if timing_enabled else None
                    if timing_enabled:
                        gpu_start.record()
                    body_scene = self._prepare_body_scene(scene) if new_frame else None
                    intent, _, timing = self.multirate.update(when=now, body=body,
                        goals=self.goals, visual=self.visual, visual_valid=self.visual_valid,
                        scene_valid=self.scene_valid, new_scene=body_scene,
                        new_visual_context=context if new_frame else None)
                    if timing_enabled:
                        gpu_end.record()
                    intent = intent.cpu().numpy()[0].tolist()
                    self._sync_body()
                    body_gpu_span_ms = gpu_start.elapsed_time(gpu_end) if timing_enabled else None
                finished = time.perf_counter()
                thread_finished = self._body_thread_time()
                held = timing['previous_held_body_receipt']
                selected_age = finished-body.original_received_monotonic_seconds
                held_age = None if held is None else finished-held
                visual_ages = [finished-x for x in context.render_start_lowers]
                self.ticker_events.append({'decision': now, 'started': started, 'finished': finished,
                    'body_encode_started': encode_started, 'body_encode_finished': encode_finished,
                    'body_thread_cpu_seconds': None if thread_started is None else thread_finished-thread_started,
                    'body_cuda_event_span_ms': body_gpu_span_ms,
                    'native_rows_consumed': index, 'packet': body.packet_id,
                    'body_receipt': body.original_received_monotonic_seconds,
                    'held_body_receipt': held, 'body_receipt_age_at_output_seconds': selected_age,
                    'held_body_receipt_age_at_output_seconds': held_age,
                    'view_render_age_upper_at_output_seconds': visual_ages,
                    'visual_frame_index': frame_id, 'visual_published': published,
                    'visual_replaced': new_frame, 'intent': intent,
                    'output_host_freshness_passed': output_host_freshness(has_intent=True,
                        selected_body_age=selected_age, held_body_age=held_age, render_ages=visual_ages),
                    'vehicle_control': False})
                previous_id, last_update = body.packet_id, now
        except Exception as error:
            self.ticker_error = {'type': type(error).__name__, 'reason': str(error)}
        finally:
            with self.mail_condition:
                self.mail_condition.notify_all()

    def stop_ticker(self):
        self.stop_event.set()
        self.ticker.join(timeout=5)
        if self.ticker.is_alive():
            raise TimeoutError('Original body ticker remains live')


class SplitCudaMultirateLiveShadow(MultirateLiveShadow):
    """Independent CUDA streams; scene copies are completed before publication."""

    def __init__(self, *, body_stream_priority=0, **kwargs):
        if type(body_stream_priority) is not int or body_stream_priority not in (-1, 0):
            raise ValueError('Explicit default or high body stream priority required')
        super().__init__(**kwargs)
        # Base setup uses disposable warmups on the original stream. All warmup
        # and initialized model buffers must be complete before either worker starts.
        torch.cuda.synchronize()
        self.visual_cuda_stream = torch.cuda.Stream(device='cuda')
        self.body_cuda_stream = torch.cuda.Stream(device='cuda', priority=body_stream_priority)
        if self.body_cuda_stream.priority != body_stream_priority:
            raise ValueError('Runtime did not honor the requested body stream priority')
        self.provenance['cuda_streams'] = 'independent_visual_body_host_synchronized_publication'
        self.provenance['body_stream_priority'] = body_stream_priority

    def _visual_scope(self):
        return torch.cuda.stream(self.visual_cuda_stream)

    def _body_scope(self):
        return torch.cuda.stream(self.body_cuda_stream)

    def _sync_visual(self):
        self.visual_cuda_stream.synchronize()

    def _sync_body(self):
        self.body_cuda_stream.synchronize()
