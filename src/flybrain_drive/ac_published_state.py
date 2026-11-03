"""Bounded retry for producer sharing/atomic-replace gaps; no stale-value fallback."""

import json
import math
import os
import time
from pathlib import Path

from .session_heartbeat_provenance import (
    HeartbeatProvenanceError,
    capture_lease,
    validate_opened_status,
)


def read_published_json(path, *, maximum_wait=.01):
    if not 0 < maximum_wait <= .01:
        raise ValueError('Published JSON sharing retry is bounded to10ms')
    deadline = time.perf_counter()+maximum_wait
    while True:
        try:
            captured = capture_lease(path)
            with Path(path).open('rb') as stream:
                content = stream.read()
                value = json.loads(content.decode('utf8'))
                modified = os.fstat(stream.fileno()).st_mtime
            validate_opened_status(path, content, value, captured_lease=captured)
            validate_async_status_creation(value)
            return value, modified
        except HeartbeatProvenanceError:
            if time.perf_counter() >= deadline:
                raise
            time.sleep(.001)
        except (PermissionError, FileNotFoundError) as error:
            transient = (isinstance(error, FileNotFoundError)
                         or getattr(error, 'winerror', None) in (32, 33)
                         or os.name == 'nt' and error.errno == 13)
            if not transient or time.perf_counter() >= deadline:
                raise
            time.sleep(.001)


def validate_async_status_creation(value):
    publication = value.get('status_publication')
    if publication is not None:
        created = publication.get('source_unix_seconds')
        source_clock = publication.get('source_script_clock')
        # Lua os.time() is quantized down to a whole Unix second.
        # This is a conservative age upper bound, not disk mtime.
        # Wall-clock changes fail closed; physical sensor ages use QPC.
        if (publication.get('schema') != 'flybrain.csp.async_periodic_status.v1'
                or publication.get('policy') != 'single_inflight_no_pending_queue'
                or type(publication.get('sequence')) is not int or publication['sequence'] < 1
                or type(created) not in (int, float) or not math.isfinite(created)
                or created != math.floor(created) or not 0 <= time.time()-created <= 2
                or type(source_clock) not in (int, float) or not math.isfinite(source_clock)
                or source_clock != value.get('script_clock')):
            raise ValueError('Original asynchronous CSP status creation is stale/future or invalid')
