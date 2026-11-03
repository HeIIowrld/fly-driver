"""Finite steering overlay:150ms freshness per request,80ms accepted-command lease."""

import math
import os
import threading
import time

from .win_deadline_wait import WindowsDeadlineWaiter


class TimedSteeringPulse:
    """One owned episode; never re-date inputs, restart after failure, or own the SDK device."""

    def __init__(self, session, emit, *, clock=time.perf_counter):
        self.session, self.emit, self.clock = session, emit, clock
        self.lock = threading.RLock()
        self.wake = WindowsDeadlineWaiter() if os.name == 'nt' else threading.Event()
        self.base, self.source, self.buttons = None, None, ()
        self.deadline, self.sign = None, 0
        self.command_lease_deadline = None
        self.failed, self.closed, self.requests = None, False, 0
        self.thread = threading.Thread(target=self._worker,
                                       name='flight-timed-steering', daemon=True)
        self.thread.start()

    def _check(self):
        if (not self.closed and not self.failed and self.command_lease_deadline is not None
                and self.clock() > self.command_lease_deadline):
            self.failed = 'Accepted command lease expired; full brake override'
            self.wake.set()
        if self.closed or self.failed:
            raise RuntimeError('Timed actuator closed/failed: '+str(self.failed))
        if self.requests >= 512:
            raise RuntimeError('Finite timed actuator request budget exhausted')

    def check(self):
        with self.lock:
            self._check()

    def _write(self, kind, *, safe=False):
        if not safe:
            self._check()
        now = self.clock()
        if not safe and (self.source is None or not 0 <= now-self.source <= .15):
            raise ValueError('Original camera source expired before timed actuator request')
        c = dict(steering=0., throttle=0., brake=1.) if safe else dict(self.base)
        pulse = self.sign*.05 if not safe else 0.
        unclipped = c['steering']+pulse
        c['steering'] = min(.1, max(-.1, unclipped))
        self.requests += 1
        request = self.session.submit(**c, pressed_buttons=() if safe else self.buttons)
        if not safe:
            self.command_lease_deadline = request[
                'driver_call_finished_perf_counter_seconds']+.08
        event = dict(kind=kind, request=request, base_control=self.base,
                     source_render_started_lower=self.source, source_age_at_call=now-self.source
                     if self.source is not None else None, disturbance=pulse,
                     clipped=c['steering'] != unclipped, safe_override=safe,
                     scheduled_pulse_off=self.deadline,
                     accepted_command_lease_deadline=self.command_lease_deadline,
                     command_gap_limit_seconds=.08, request_source_age_limit_seconds=.15)
        self.emit(event)
        return event

    def submit(self, control, *, source_render_started_lower, pressed_buttons=()):
        c = dict(control)
        if (set(c) != {'steering', 'throttle', 'brake'}
                or not all(math.isfinite(v) for v in c.values()) or abs(c['steering']) > .1
                or not 0 <= c['throttle'] <= .12 or not 0 <= c['brake'] <= 1
                or c['throttle'] and c['brake']
                or not math.isfinite(source_render_started_lower)):
            raise ValueError('Finite bounded engineering base control/source required')
        with self.lock:
            self._check()
            if not 0 <= self.clock()-source_render_started_lower <= .15:
                raise ValueError('Never re-date a stale or future source')
            self.base, self.source, self.buttons = c, source_render_started_lower, pressed_buttons
            try:
                event = self._write('timed_base_request')
            except Exception as error:
                self.failed = str(error)
                self.wake.set()
                raise
            self.wake.set()
            return event

    def begin(self, sign):
        with self.lock:
            self._check()
            if type(sign) is not int or sign not in (-1, 1) or self.deadline is not None:
                raise ValueError('One signed280ms pulse required; no overlapping/repeated pulse')
            if self.sign or getattr(self, 'pulse_started', False):
                raise ValueError('Exactly one pulse per owned episode')
            self.sign = sign
            try:
                event = self._write('timed_pulse_on')
                self.deadline = event['request']['driver_call_finished_perf_counter_seconds']+.28
                self.pulse_started = True
                self.emit(dict(kind='timed_pulse_schedule', deadline=self.deadline,
                               pulse_on_request_sequence=event['request']['request_sequence']))
            except Exception as error:
                self.sign = 0
                self.failed = str(error)
                self.wake.set()
                raise
            self.wake.set()
            return event

    def _worker(self):
        try:
            while True:
                with self.lock:
                    if self.closed:
                        return
                    now = self.clock()
                    if (self.command_lease_deadline is not None
                            and now > self.command_lease_deadline):
                        self.failed = 'Accepted command lease expired; full brake override'
                    if self.failed:
                        self.sign = 0
                        self._write('timed_failure_brake', safe=True)
                        return
                    if self.deadline is not None and self.sign and now >= self.deadline:
                        self.sign = 0
                        self._write('timed_pulse_off')
                    targets = [v for v in (self.deadline if self.sign else None,
                                           self.command_lease_deadline)
                               if v is not None]
                    delay = min(.005, max(0., min(targets)-self.clock())) if targets else .005
                    self.wake.clear()
                self.wake.wait(delay)
        except Exception as error:
            with self.lock:
                self.failed = str(error)
                self.sign = 0
                try:
                    self._write('timed_worker_error_brake', safe=True)
                except Exception:
                    pass

    def stop(self):
        """Cancel before session.close; no worker writes are allowed after this returns."""
        failure = None
        try:
            with self.lock:
                if not self.closed:
                    self.closed = True
                    self.sign = 0
                    try:
                        self._write('timed_cancel_brake', safe=True)
                    finally:
                        self.wake.set()
        except Exception as error:
            failure = error
        finally:
            self.thread.join(timeout=1.)
        if self.thread.is_alive():
            raise TimeoutError('Same timed actuator worker still live; do not release/reuse device')
        if hasattr(self.wake, 'close'):
            self.wake.close()
        if failure is not None:
            raise RuntimeError('Cancel request failed after worker stopped') from failure
        return dict(worker_alive=False, requests=self.requests, error=self.failed,
                    wait_backend=getattr(self.wake, 'backend', 'threading_event_non_windows'))
