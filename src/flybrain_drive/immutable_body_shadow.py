"""Single-consumer shared-camera and original native-body shadow, without control."""

import threading
import time

from .online_body_shadow import OnlineBodyShadow


class ImmutableBodyShadow:
    """Connect an existing camera owner and existing body ticker; never start either."""

    def __init__(self, *, camera, neural_shadow, native_rows, native_identity, episode_id,
                 config_path, clock=time.perf_counter):
        if (not hasattr(neural_shadow, 'ticker') or not neural_shadow.ticker.is_alive()
                or not isinstance(episode_id, str) or not episode_id
                or neural_shadow.native_rows is not native_rows
                or neural_shadow.episode_id != episode_id
                or camera.host_clock is not clock or clock is not time.perf_counter):
            raise ValueError('Existing live owned ticker and explicit episode required')
        self.camera, self.neural_shadow, self.clock = camera, neural_shadow, clock
        self.body = OnlineBodyShadow(directory='.', request_id=episode_id,
            generation=camera.clock.generation, offset_lower=0., native_rows=native_rows,
            native_identity=native_identity, clock=clock)
        self.neural_shadow.configure_immutable(camera.config, config_path)
        self.events, self.error, self.owner = [], None, None
        self.last_candidate_signature = None

    def consume(self):
        """Consume one currently available original group or record a terminal failure."""
        if self.error is not None:
            raise RuntimeError('Failed immutable shadow cannot be resumed')
        owner = threading.get_ident()
        if self.owner is None:
            self.owner = owner
        elif self.owner != owner:
            raise RuntimeError('Only the original immutable shadow owner may consume')
        event = {'selection_started': self.clock(), 'vehicle_control': False}
        try:
            ticker = self.neural_shadow
            if ticker.ticker_error is not None or ticker.ticker_stop_reason is not None:
                raise RuntimeError('Original body ticker already stopped or failed')
            if not ticker.ticker.is_alive():
                raise RuntimeError('Original body ticker is not live')
            candidate = self.camera.candidate_before(event['selection_started'])
            event.update(capture_id=candidate['capture_id'], generation=candidate['generation'],
                         camera_received=candidate['received'])
            signature = (candidate['generation'], candidate['capture_id'], candidate['received'],
                         tuple((v['view_id'], v['rgb_sha256'], v['render_started_lower'])
                               for v in candidate['views']))
            if (self.last_candidate_signature is not None
                    and signature[1] == self.last_candidate_signature[1]):
                if signature != self.last_candidate_signature:
                    raise ValueError('Repeated camera candidate changed original metadata')
                event.update(neural={'status': 'duplicate_not_reprocessed',
                                     'vehicle_control': False}, selection_finished=self.clock())
                self.events.append(event)
                return event
            samples = self.camera.samples_for(candidate)

            def select_body(visual_event):
                return self.body._select_context(visual_event,
                    camera_receipt=visual_event['camera_receipt'],
                    bounds=tuple(visual_event['render_start_lower']))

            visual, output = ticker.process_immutable(candidate, samples, select_body)
            event.update(visual=visual, neural=output, selection_finished=self.clock())
            self.last_candidate_signature = signature
            self.events.append(event)
            return event
        except Exception as error:
            self.error = {'type': type(error).__name__, 'reason': str(error)}
            event.update(error=self.error, selection_finished=self.clock())
            self.events.append(event)
            raise
