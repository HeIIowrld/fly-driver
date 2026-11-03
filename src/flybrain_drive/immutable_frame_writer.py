"""Bounded FIFO persistence of owned immutable frame bytes; no sensor receipt clock changes."""

import hashlib
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class ImmutableFrameWriter:
    def __init__(self, directory, stream_path, *, maximum_pending=8, clock=time.perf_counter,
                 mmf_metadata=False):
        if type(maximum_pending) is not int or not 1 <= maximum_pending <= 8:
            raise ValueError('Bounded1..8 immutable frame persistence queue required')
        self.directory, self.stream_path = Path(directory), Path(stream_path)
        if type(mmf_metadata) is not bool:
            raise ValueError('Explicit MMF metadata persistence mode required')
        self.mmf_metadata = mmf_metadata
        self.maximum_pending, self.clock = maximum_pending, clock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='immutable-frame-disk')
        self.pending, self.records = deque(), []
        self.submitted, self.maximum_queued = 0, 0
        self.error, self.closed = None, False

    def check_capacity(self):
        if self.closed:
            raise ValueError('Frame writer is closed')
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        if self.error:
            raise RuntimeError('Immutable frame persistence failed') from self.error
        if len(self.pending) >= self.maximum_pending:
            raise BufferError('Immutable frame queue full; no frame may be silently dropped')

    def submit(self, group_id, files, stream_line, *, metadata_files=()):
        self.check_capacity()
        if (group_id != self.submitted or type(group_id) is not int
                or not isinstance(stream_line, bytes)):
            raise ValueError('Increasing exact group ID and immutable metadata bytes required')
        if len(files) != 4 or len({name for name, _ in files}) != 4:
            raise ValueError('Four distinct immutable image files required')
        if self.mmf_metadata:
            names = {f'frame-{group_id:04d}.json', f'ack-{group_id:04d}.json',
                     f'mmf-{group_id:04d}.mmf'}
            if len(metadata_files) != 3 or {n for n, _ in metadata_files} != names:
                raise ValueError('Exact original metadata, ACK and two raw slots required')
        elif metadata_files:
            raise ValueError('Legacy persistence cannot silently accept MMF metadata')
        files = (*files, *metadata_files)
        for name, content in files:
            if (not isinstance(name, str) or Path(name).name != name or ':' in name
                    or '/' in name or '\\' in name or not isinstance(content, bytes)):
                raise ValueError('Flat image filenames and owned immutable bytes required')
        if sum(len(b) for _, b in files) > 1024**2:
            raise ValueError('One frame group exceeds reviewed immutable byte budget')
        submitted = self.clock()
        future = self.pool.submit(self._persist, group_id, tuple(files), stream_line, submitted)
        self.pending.append(future)
        self.submitted += 1
        self.maximum_queued = max(self.maximum_queued, len(self.pending))
        return submitted

    def _persist(self, group_id, files, stream_line, submitted):
        row = dict(group_id=group_id, submitted_at=submitted, started_at=self.clock(), files=[])
        try:
            for name, content in files:
                path = self.directory/name
                expected = hashlib.sha256(content).hexdigest()
                with path.open('xb') as file:
                    file.write(content)
                with path.open('rb') as file:
                    actual = hashlib.file_digest(file, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError('Persisted frame bytes differ from immutable source')
                row['files'].append(dict(file=name, bytes=len(content), sha256=actual))
            with self.stream_path.open('ab') as file:
                file.write(stream_line)
                file.flush()
            row.update(status='persisted', finished_at=self.clock(),
                       stream_line_sha256=hashlib.sha256(stream_line).hexdigest())
            self.records.append(row)
            return row
        except BaseException as error:
            self.error = self.error or error
            row.update(status='failed', error=type(error).__name__+': '+str(error),
                       finished_at=self.clock())
            self.records.append(row)
            raise

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=False)
            self.closed = True
            # Drain every accepted task, including tasks after a failed write.
            while self.pending:
                try:
                    self.pending.popleft().result()
                except BaseException as error:
                    self.error = self.error or error
        if self.error:
            raise RuntimeError('Immutable frame persistence failed') from self.error
        assert len(self.records) == self.submitted

    def statistics(self):
        return dict(schema='flybrain.immutable_frame_persistence.v2' if self.mmf_metadata
                    else 'flybrain.immutable_frame_persistence.v1',
            submitted=self.submitted, maximum_pending=self.maximum_pending,
            maximum_queued=self.maximum_queued, closed=self.closed,
            completed=sum(r['status'] == 'persisted' for r in self.records),
            failed=sum(r['status'] == 'failed' for r in self.records),
            error=str(self.error) if self.error else None, records=self.records)


def verify_persistence_archive(camera, *, expected_mmf=False):
    """Independently verify every deferred group, including frames the actor never selected."""
    journal = camera.json('frame-persistence-journal.json')
    lines = camera.read('stream.jsonl').splitlines(keepends=True)
    records = journal['records']
    count = journal['submitted']
    schema = 'flybrain.immutable_frame_persistence.v2' if expected_mmf else (
        'flybrain.immutable_frame_persistence.v1')
    if (journal['schema'] != schema
            or journal['closed'] is not True or journal['error'] is not None
            or journal['failed'] != 0 or journal['completed'] != count
            or len(records) != count or len(lines) != count or not count
            or not 1 <= journal['maximum_queued'] <= journal['maximum_pending'] <= 8):
        raise ValueError('Deferred persistence journal incomplete or failed')
    rows = {}
    for index, (record, line) in enumerate(zip(records, lines, strict=True)):
        row = json.loads(line)
        pipeline = row['shared_pipeline']
        clocks = [pipeline['acknowledged_copy_clock'], pipeline['persistence_started_clock'],
                  row['received'], record['submitted_at'], record['started_at'],
                  record['finished_at']]
        if (record['group_id'] != index or row['capture_id'] != index
                or record['status'] != 'persisted' or not all(math.isfinite(c) for c in clocks)
                or clocks != sorted(clocks)
                or pipeline['revision'] != ('copy_all_then_mmf_ack_then_async_persist_v3'
                    if expected_mmf else 'copy_all_then_ack_then_async_persist_v2')
                or pipeline['stream_schema'] != ('flybrain.csp.immutable_rgb_stream.v3'
                    if expected_mmf else 'flybrain.csp.immutable_rgb_stream.v2')
                or pipeline['immutable_memory_ready_clock'] != row['received']
                or pipeline['actor_published_only_after_persistence'] is not False
                or pipeline['persistence_required_before_run_admission'] is not True
                or 'persistence_finished_clock' in pipeline
                or hashlib.sha256(line).hexdigest() != record['stream_line_sha256']):
            raise ValueError('Deferred receipt identity, clock or protocol differs')
        files = {f['file']: f for f in record['files']}
        view_names = {Path(v['file']).name for v in row['views']}
        metadata_names = {f'frame-{index:04d}.json', f'ack-{index:04d}.json',
                          f'mmf-{index:04d}.mmf'} if expected_mmf else set()
        if (len(files) != 4+len(metadata_names) or len(record['files']) != len(files)
                or len(row['views']) != 4 or view_names | metadata_names != set(files)):
            raise ValueError('Four distinct persisted views required')
        for view in row['views']:
            name = Path(view['file']).name
            raw = camera.read(name)
            if (name not in files or len(raw) != files[name]['bytes']
                    or hashlib.sha256(raw).hexdigest() != files[name]['sha256']
                    or files[name]['sha256'] != view['file_sha256']):
                raise ValueError('Deferred frame bytes differ from immutable receipt')
        if expected_mmf:
            from .ac_metadata_mmf import verify_mmf_archive_row
            verify_mmf_archive_row(camera, row)
            for name in metadata_names:
                raw = camera.read(name)
                if (len(raw) != files[name]['bytes']
                        or hashlib.sha256(raw).hexdigest() != files[name]['sha256']):
                    raise ValueError('Deferred original MMF metadata differs from persistence')
        rows[index] = row
    return rows
