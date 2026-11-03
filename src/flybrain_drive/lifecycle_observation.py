"""Read-only host-clock observations of published CSP lifecycle state."""

import hashlib
import json
import threading
import time

from .ac_published_state import read_published_json


class LifecycleObserver:
    """Record observations and explicit read failures; never write or control CSP."""

    def __init__(self, path, generation, *, interval=.01, reader=read_published_json):
        if not .005 <= interval <= .05 or not isinstance(generation, str) or not generation:
            raise ValueError('Bounded interval and explicit generation required')
        self.path, self.generation, self.interval, self.reader = path, generation, interval, reader
        self.events, self.stop_event = [], threading.Event()
        self.active_seen, self.pause_after_active = False, threading.Event()
        self.pause_observation = None
        self.thread = threading.Thread(target=self._loop, name='csp-lifecycle-observer', daemon=True)

    def sample(self):
        start = time.perf_counter()
        try:
            status, modified = self.reader(self.path)
            wall = time.time()
            end = time.perf_counter()
            payload = json.dumps(status, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf8')
            fields = {key: status.get(key) for key in ('generation', 'epoch', 'paused', 'ai_controlled',
                'expert_probe_active', 'camera_active', 'in_menu', 'live', 'script_clock')}
            observation = {'started':start, 'finished':end, 'status':fields,
                'source_modified_wall_seconds':modified, 'observed_wall_seconds':wall,
                'fresh':0 <= wall-modified <= 2, 'same_generation':status.get('generation') == self.generation,
                'canonical_payload_sha256':hashlib.sha256(payload).hexdigest()}
            self.events.append(observation)
            if observation['fresh'] and observation['same_generation']:
                if fields['paused'] is False and fields['expert_probe_active'] is True:
                    self.active_seen = True
                elif (self.active_seen and fields['paused'] is True
                      and fields['ai_controlled'] is False and fields['expert_probe_active'] is False
                      and not self.pause_after_active.is_set()):
                    self.pause_observation = observation
                    self.pause_after_active.set()
        except Exception as error:
            self.events.append({'started':start, 'finished':time.perf_counter(),
                                'error':{'type':type(error).__name__, 'reason':str(error)}})

    def _loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise TimeoutError('Read-only lifecycle observer remains live')
