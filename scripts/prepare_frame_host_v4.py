"""Deterministic new-only V4 camera/recorder/runner/auditor sources; no live action."""
# ruff: noqa: E501 -- Exact source fragments are intentionally kept recognizable.

import argparse
import hashlib
import json
from pathlib import Path

from flybrain_drive.frame_persistence_profile_v4 import HostCopyProfile

BASES = {
    'src/flybrain_drive/ac_csp_vision.py':
        '366496500c3f367523d3f35f41a59c92b1b257bb43c4776b16c6634389179efc',
    'src/flybrain_drive/ac_shared_texture.py':
        'bed33b2b71ba597597a92867d82832e6d3848466d2ed7dca36958fdc317e6eaa',
    'scripts/record_ac_engine_ai_probe_buffered.py':
        '2da45564c9d8c377a8c6c7628918227c92fc50638c6280891521a98f3f333779',
    'scripts/qualify_session_heartbeat_evidence_buffered.py':
        '50636a6bfd2577624683f92635905c91514e1b5e2261933842e2c36ee550dbb4',
    'scripts/audit_ac_engine_ai_probe_buffered.py':
        'd847141255a8a2c139572f936c7edb4e50d530eb2daa81eb49d134da016667f8',
}


def checked(name):
    raw = Path(name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASES[name]:
        raise ValueError('Frozen base source changed: '+name)
    return raw.decode().replace('\r\n','\n')


def once(value, before, after):
    if value.count(before) != 1:
        raise ValueError('Unique fixed source substitution required: '+before[:80])
    return value.replace(before, after)


def build():
    shared = checked('src/flybrain_drive/ac_shared_texture.py')
    shared = once(shared, 'def read_group(self, handles, *, timeout=.04):',
        'def read_group(self, handles, *, timeout=.04, host_trace=None, group_id=None):')
    shared = once(shared, '        handles = [parse_shared_handle(h) for h in handles]\n',
        "        if host_trace is not None:\n"
        "            host_trace.mark(group_id, 'handles_parse_enter')\n"
        '        handles = [parse_shared_handle(h) for h in handles]\n'
        "        if host_trace is not None:\n"
        "            host_trace.mark(group_id, 'handles_parse_exit')\n")
    shared = once(shared, '        started = time.perf_counter()\n        self.last_group_trace',
        "        if host_trace is not None:\n"
        "            host_trace.mark(group_id, 'group_copy_enter')\n"
        '        started = time.perf_counter()\n        self.last_group_trace')
    shared = once(shared, "            trace['finished_at'] = time.perf_counter()\n            return result",
        "            trace['finished_at'] = time.perf_counter()\n"
        "            if host_trace is not None:\n"
        "                host_trace.mark(group_id, 'group_copy_exit')\n            return result")

    vision = checked('src/flybrain_drive/ac_csp_vision.py')
    vision = once(vision, 'from .ac_shared_texture import SharedTextureReader',
        'from .ac_shared_texture_host_v4 import SharedTextureReader')
    vision = once(vision, 'from .graph import sha256\n',
        'from .frame_persistence_profile_v4 import (\n'
        '    HostCopyProfile,\n    HostFrameWriterFactory,\n'
        '    HostPayloadNamespace,\n    HostTrace,\n)\n'
        'from .graph import sha256\n')
    vision = once(vision, '                      ack_writer=None):',
        '                      ack_writer=None, host_namespace=None, host_trace=None):')
    vision = once(vision, '    directory = Path(directory)\n    if type(batched)',
        "    if host_trace is not None:\n"
        "        host_trace.mark(frame['index'], 'helper_enter')\n"
        '    directory = Path(directory)\n    if type(batched)')
    vision = once(vision, "    pixels, receipts = {}, []\n    for view in views:",
        "    if host_namespace is not None:\n"
        "        host_namespace.reserve(frame['index'], views)\n"
        "        host_trace.mark(frame['index'], 'targets_reserved')\n"
        "    pixels, receipts = {}, []\n    for view in views:")
    vision = once(vision, "or (directory/view['file']).exists()):",
        "or host_namespace is None and (directory/view['file']).exists()):")
    vision = once(vision, "gpu.read_group([v['shared_texture_handle'] for v in views]) if batched else",
        "gpu.read_group([v['shared_texture_handle'] for v in views],\n"
        "                host_trace=host_trace, group_id=frame['index']) if batched else")
    vision = once(vision, '    if len(copies) != 4:\n',
        "    if host_trace is not None:\n"
        "        host_trace.mark(frame['index'], 'helper_copy_returned')\n"
        '    if len(copies) != 4:\n')
    vision = once(vision, '    ack_written = host_clock()\n',
        "    ack_written = host_clock()\n"
        "    if host_trace is not None:\n"
        "        host_trace.mark(frame['index'], 'ack_finished')\n")
    vision = once(vision, '                 frame_writer_factory=None):',
        '                 frame_writer_factory=None, host_copy_profile=None):')
    vision = once(vision, '        self.frame_writer_factory = frame_writer_factory\n',
        '        self.frame_writer_factory = frame_writer_factory\n'
        '        if host_copy_profile is not None and (\n'
        '                not isinstance(host_copy_profile, HostCopyProfile)\n'
        '                or not isinstance(frame_writer_factory, HostFrameWriterFactory)\n'
        '                or not mmf_metadata or not batched_shared_copy):\n'
        "            raise ValueError('Explicit hostV4 MMF namespace/factory required')\n"
        '        self.host_namespace = self.host_trace = None\n'
        '        self.owned_cleanup = dict(camera_thread_joined=False,\n'
        '            producer_terminal_known=False, persistence_joined=False,\n'
        '            precompute_joined=False, shared_reader_closed=False)\n')
    vision = once(vision, '        self.local.mkdir(parents=True, exist_ok=False)\n',
        '        self.local.mkdir(parents=True, exist_ok=False)\n'
        '        if host_copy_profile is not None:\n'
        "            self.host_namespace = HostPayloadNamespace.create(self.local,\n"
        "                self.request_id, status['generation'], max_frames)\n"
        '            self.host_trace = HostTrace()\n'
        '            self.frame_writer_factory = frame_writer_factory.bind(\n'
        '                self.host_namespace, self.host_trace)\n')
    vision = once(vision, '        return factory(self.directory, self.local/\'stream.jsonl\',',
        '        directory = self.host_namespace.directory if self.host_namespace else self.directory\n'
        '        return factory(directory, self.local/\'stream.jsonl\',')
    vision = once(vision, "                            ack_writer=acknowledge if channel else None)",
        "                            ack_writer=acknowledge if channel else None,\n"
        "                            host_namespace=self.host_namespace, host_trace=self.host_trace)")
    vision = once(vision, '                            gpu, frame, self.directory, host_clock=self.host_clock,',
        '                            gpu, frame, (self.host_namespace.directory\n'
        '                                if self.host_namespace else self.directory),\n'
        '                            host_clock=self.host_clock,')
    vision = once(vision, '        finally:\n            if channel:',
        "        finally:\n            if self.frame_writer is None:\n"
        "                self.owned_cleanup['persistence_joined'] = True\n"
        "            if precompute is None:\n"
        "                self.owned_cleanup['precompute_joined'] = True\n"
        "            if gpu is None:\n"
        "                self.owned_cleanup['shared_reader_closed'] = True\n"
        '            if channel:')
    vision = once(vision, '                    self.frame_writer.close()\n',
        "                    self.frame_writer.close()\n"
        "                    self.owned_cleanup['persistence_joined'] = True\n")
    vision = once(vision, '                    precompute.close()\n',
        "                    precompute.close()\n"
        "                    self.owned_cleanup['precompute_joined'] = True\n")
    vision = once(vision, '                gpu.close()\n',
        "                gpu.close()\n"
        "                self.owned_cleanup['shared_reader_closed'] = True\n")
    vision = once(vision, "        source_timing = None\n        if self.source_timing_v1", 
        "        self.owned_cleanup['camera_thread_joined'] = True\n"
        "        self.owned_cleanup['producer_terminal_known'] = (not self.triggered or\n"
        "            self.terminal.get('status') in ('stopped', 'completed')\n"
        "            and self.terminal.get('request_id') == self.request_id\n"
        "            and type(self.terminal.get('frames')) is int\n"
        "            and self.count <= self.terminal['frames'] <= self.request['frames'])\n"
        "        if self.host_namespace is not None:\n"
        "            if not all(self.owned_cleanup[k] for k in (\n"
        "                    'persistence_joined','precompute_joined','shared_reader_closed')):\n"
        "                raise RuntimeError('Owned host cleanup unknown; retain trace, no archive')\n"
        "            write_json(self.local/'host-payload-namespace-v4.json',\n"
        "                       self.host_namespace.record())\n"
        "            write_json(self.local/'host-precopy-trace-v4.json', self.host_trace.record())\n"
        "        source_timing = None\n        if self.source_timing_v1")
    vision = once(vision, '        return writer((self.directory, self.local), destination)',
        '        roots = (self.directory, self.local) + ((self.host_namespace.directory,)\n'
        '            if self.host_namespace is not None else ())\n'
        '        return writer(roots, destination)')

    recorder = checked('scripts/record_ac_engine_ai_probe_buffered.py')
    recorder = recorder.replace('flybrain_drive.ac_csp_vision import',
        'flybrain_drive.ac_csp_vision_host_v4 import')
    recorder = recorder.replace('frame_persistence_profile_v3', 'frame_persistence_profile_v4')
    recorder = recorder.replace('BufferedFrameWriterFactory', 'HostFrameWriterFactory')
    recorder = recorder.replace('load_buffered_frame_profile', 'load_host_copy_profile')
    recorder = recorder.replace("'src/flybrain_drive/ac_csp_vision.py'",
        "'src/flybrain_drive/ac_csp_vision_host_v4.py'")
    recorder = once(recorder, "    'src/flybrain_drive/frame_persistence_profile_v4.py',\n",
        "    'src/flybrain_drive/frame_persistence_profile_v4.py',\n"
        "    'src/flybrain_drive/ac_shared_texture_host_v4.py',\n"
        "    'src/flybrain_drive/collection_cleanup_v4.py',\n")
    recorder = once(recorder, 'from flybrain_drive.evidence import begin_run, write_json\n',
        'from flybrain_drive.collection_cleanup_v4 import recorder_cleanup\n'
        'from flybrain_drive.evidence import begin_run, write_json\n')
    recorder = once(recorder, '                                   frame_writer_factory=frame_writer_factory,',
        '                                   frame_writer_factory=frame_writer_factory,\n'
        '                                   host_copy_profile=profile,')
    recorder = once(recorder, "        write_json(run/'result.json', result)\n",
        "        result['owned_cleanup'] = recorder_cleanup(child, vision)\n"
        "        write_json(run/'result.json', result)\n")

    runner = checked('scripts/qualify_session_heartbeat_evidence_buffered.py')
    runner = runner.replace('frame_persistence_profile_v3', 'frame_persistence_profile_v4')
    runner = runner.replace('load_buffered_frame_profile', 'load_host_copy_profile')
    runner = runner.replace('scripts.record_ac_engine_ai_probe_buffered',
        'scripts.record_ac_engine_ai_probe_host_v4')
    start = runner.index('        def idle_after_cleanup():')
    end = runner.index("        result['status'] = 'completed_stationary_transport_smoke'", start)
    runner = runner[:start]+(
        "        from flybrain_drive.collection_cleanup_v4 import finish_owned_transport\n\n"
        "        result['handoff'] = finish_owned_transport(\n"
        "            context, evidence, fresh_idle, wait_owned)\n")+runner[end:]
    runner = once(runner, "    except BaseException as error:\n        result.update(error=type(error).__name__, reason=str(error))\n        raise\n",
        "    except BaseException as error:\n"
        "        result.update(error=type(error).__name__, reason=str(error), status='failed')\n"
        "        if isinstance(error, Exception):\n"
        "            from flybrain_drive.collection_cleanup_v4 import (\n"
        "                load_owned_recording_failure,\n"
        "                retire_clean_failed_data,\n"
        "            )\n\n"
        "            retire_clean_failed_data(context, evidence, fresh_idle, wait_owned, result,\n"
        "                load_owned_recording_failure(output/'recording/result.json'),\n"
        "                now_qpc_ns=time.perf_counter_ns())\n"
        "        raise  # Data failure stays exit1 even if healthy transport retired normally.\n")
    auditor = checked('scripts/audit_ac_engine_ai_probe_buffered.py')
    auditor = auditor.replace('frame_persistence_profile_v3', 'frame_persistence_profile_v4')
    auditor = auditor.replace('BufferedFrameProfile', 'HostCopyProfile')
    auditor = auditor.replace('verify_buffered_persistence_archive', 'verify_host_persistence_archive')
    return {Path('src/flybrain_drive/ac_shared_texture_host_v4.py'): shared.encode(),
        Path('src/flybrain_drive/ac_csp_vision_host_v4.py'): vision.encode(),
        Path('scripts/record_ac_engine_ai_probe_host_v4.py'): recorder.encode(),
        Path('scripts/qualify_session_heartbeat_evidence_host_v4.py'): runner.encode(),
        Path('scripts/audit_ac_engine_ai_probe_host_v4.py'): auditor.encode(),
        Path('configs/ac-frame-host-v4-20261030.json'):
            (json.dumps(HostCopyProfile().record(), indent=2)+'\n').encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-unfrozen-new-sources', action='store_true')
    args = parser.parse_args()
    outputs = build()
    if args.write_unfrozen_new_sources:
        for path, content in outputs.items():
            path.write_bytes(content)
    print(json.dumps(dict(base_sha256=BASES,
        outputs={str(p):hashlib.sha256(b).hexdigest() for p,b in outputs.items()},live_actions=0)))


if __name__ == '__main__':
    main()
