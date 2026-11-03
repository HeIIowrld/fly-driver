"""UNSEALED CPU/source-only derivative builder; no role/deploy/capture operation."""
# ruff: noqa: E501 -- Exact immutable source fragments intentionally retain context.
import argparse
import hashlib
import json
from pathlib import Path

BASES = {
    'scripts/record_ac_engine_ai_probe_host_v4.py': 'c80b8715541851283b23bd7b8f0db19a7c51d21c96ddd9809889577eeecc8879',
    'scripts/qualify_session_heartbeat_evidence_host_v4.py': 'b51bef99eb39f29d3644cb5c56953ceaaf66c6b86c84aaa7816d36949cddf608',
    'scripts/audit_ac_engine_ai_probe_host_v4.py': '7e02ef592a15351ecd6a4d0c789b32e6a1533c68206436e17c7c5df5949a0ba6',
}


def checked(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASES[path]:
        raise ValueError('Immutable old source differs: '+path)
    return raw.decode().replace('\r\n', '\n')


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Exact unique derivative edit required: '+old[:90])
    return source.replace(old, new)


def indent_block(block, spaces=4):
    return ''.join(' '*spaces+line if line.strip() else line for line in block.splitlines(True))


def recorder():
    text = checked('scripts/record_ac_engine_ai_probe_host_v4.py')
    text = once(text, 'from flybrain_drive.collection_cleanup_v4 import recorder_cleanup\n',
        'from flybrain_drive.actor_cadence_trace_v1 import ActorTrace, load_actor_profile\n'
        'from flybrain_drive.collection_cleanup_v4 import recorder_cleanup\n')
    text = once(text, "    'src/flybrain_drive/collection_cleanup_v4.py',\n",
        "    'src/flybrain_drive/collection_cleanup_v4.py',\n"
        "    'src/flybrain_drive/actor_cadence_trace_v1.py',\n")
    text = once(text, 'def main(*, heartbeat_start=None):',
        'def main(*, heartbeat_start=None, actor_trace_sink=None):')
    text = once(text, "    parser.add_argument('--output', required=True)",
        "    parser.add_argument('--actor-cadence-profile', required=True)\n"
        "    parser.add_argument('--output', required=True)")
    text = once(text, '    frame_writer_factory = HostFrameWriterFactory(profile)\n',
        "    actor_profile = load_actor_profile(args.actor_cadence_profile)\n"
        "    if actor_trace_sink is None:\n"
        "        raise ValueError('Explicit same-parent RAM actor trace sink required')\n"
        "    actor_trace = ActorTrace()  # fixed capacity allocated before activation\n"
        '    frame_writer_factory = HostFrameWriterFactory(profile)\n')
    text = once(text, "        frame_persistence_profile=profile.record(),\n",
        "        actor_cadence_profile=actor_profile.record(),\n"
        "        actor_cadence_profile_sha256=sha256(args.actor_cadence_profile),\n"
        "        frame_persistence_profile=profile.record(),\n")
    text = once(text, "    result['frame_persistence_profile'] = profile.record()\n",
        "    result['frame_persistence_profile'] = profile.record()\n"
        "    result['actor_cadence_profile'] = actor_profile.record()\n")
    start = text.index('                while child.poll() is None:\n')
    end = text.index('        if child.returncode != 0:\n', start)
    old_loop = text[start:end]
    body = old_loop.split('\n', 1)[1]
    # The SAME existing perf_counter call moves before poll/status. No observation
    # receipt or decision clock moves. Source calls stay once, with trace spans.
    body = once(body, '                    started = time.perf_counter()\n', '')
    old = ("                    held_task = read_ai_phase(session_status, vision.clock.generation,\n"
           "                        task=True, clock=vision.host_clock) if args.pace_task else None\n"
           "                    running = held_task is not None if args.pace_task else ai_is_running()\n")
    body = once(body, old, "                    with actor_trace.span('status'):\n"+indent_block(old))
    body = once(body, '                        time.sleep(.01)\n',
        "                        with actor_trace.span('sleep'):\n"
        "                            actor_trace.sleep_requested(.01)\n"
        '                            time.sleep(.01)\n')
    body = once(body, '                        vision.wait_for_candidate(timeout=.04)\n',
        "                        with actor_trace.span('candidate_wait'):\n"
        '                            vision.wait_for_candidate(timeout=.04)\n')
    old = ('                        body, coupled_authority = receive_recording_native(\n'
           '                            reader, same_copy_native_authority=args.same_copy_native_authority)\n')
    body = once(body, old, "                        with actor_trace.span('native_receive'):\n"+indent_block(old))
    body = once(body, '                        if not ai_is_running():\n                            continue\n',
        "                        with actor_trace.span('status'):\n"
        '                            still_running = ai_is_running()\n'
        '                        if not still_running:\n                            continue\n')
    body = once(body, '                        candidate = vision.candidate_before(body.received_monotonic_seconds)\n',
        "                        with actor_trace.span('candidate_select'):\n"
        '                            candidate = vision.candidate_before(body.received_monotonic_seconds)\n')
    body = once(body, '                    decision = vision.host_clock()\n',
        '                    decision = vision.host_clock()\n'
        '                    actor_trace.decision(len(times), decision)\n')
    old = ("                    packed = retina.pack(vision.samples_for(candidate), decision_time=decision,\n"
           "                        native_scalars={k: body.fields.get(k)\n"
           "                                        for k in retina.required_native_fields},\n"
           "                        native_receipt=body.received_monotonic_seconds)\n")
    body = once(body, old, "                    with actor_trace.span('retinal_pack'):\n"+indent_block(old))
    body = once(body, "                    stream.write(json.dumps(event, allow_nan=False)+'\\n')\n                    stream.flush()\n",
        "                    with actor_trace.span('event_encode'):\n"
        "                        event_json = json.dumps(event, allow_nan=False)+'\\n'\n"
        "                    with actor_trace.span('event_write'):\n"
        "                        stream.write(event_json)\n"
        "                    with actor_trace.span('event_flush'):\n"
        "                        stream.flush()\n")
    body = once(body, '                    time.sleep(max(0., .04-(time.perf_counter()-started)))\n',
        "                    with actor_trace.span('sleep'):\n"
        '                        sleep_seconds = max(0., actor_profile.nominal_seconds-(time.perf_counter()-started))\n'
        '                        actor_trace.sleep_requested(sleep_seconds)\n'
        '                        time.sleep(sleep_seconds)\n')
    loop = ("                while True:\n"
            "                    actor_trace.begin_iteration()\n"
            "                    try:\n"
            "                        started = time.perf_counter()\n"
            "                        actor_trace.budget_started(started)\n"
            "                        with actor_trace.span('child_poll'):\n"
            "                            child_active = child.poll() is None\n"
            "                        if not child_active:\n"
            "                            break\n"+indent_block(body)+
            "                    finally:\n"
            "                        actor_trace.finish_iteration(\n"
            "                            original_error_active=sys.exc_info()[0] is not None)\n")
    text = text[:start]+loop+text[end:]
    text = once(text, "        result['owned_cleanup'] = recorder_cleanup(child, vision)\n",
        "        result['owned_cleanup'] = recorder_cleanup(child, vision)\n"
        "        actor_trace_sink.accept(actor_trace.seal(), result['owned_cleanup'])\n"
        "        result['actor_trace_decisions'] = actor_trace.decisions\n")
    return text


def parent():
    text = checked('scripts/qualify_session_heartbeat_evidence_host_v4.py')
    text = text.replace('scripts.record_ac_engine_ai_probe_host_v4',
                        'scripts.record_ac_engine_ai_probe_actor_trace_v1')
    text = once(text, 'def collection_arguments(mode, output, profile_file):',
        'def collection_arguments(mode, output, profile_file, actor_profile_file):')
    text = once(text, "        '--frame-persistence-profile', str(profile_file),\n",
        "        '--frame-persistence-profile', str(profile_file),\n"
        "        '--actor-cadence-profile', str(actor_profile_file),\n")
    text = once(text, 'def run(args):\n',
        "def run(args):\n"
        "    from flybrain_drive.actor_cadence_trace_v1 import ActorTraceSink, load_actor_profile\n"
        "\n"
        "    actor_profile = load_actor_profile(args.actor_cadence_profile)\n"
        "    actor_sink = ActorTraceSink()\n")
    text = once(text, 'role=\'diagnostic\', fit_allowed=False, source_candidate_sha256=sha256(CANDIDATE)))',
        "role='diagnostic', fit_allowed=False, source_candidate_sha256=sha256(CANDIDATE),\n"
        "        actor_cadence_profile=actor_profile.record(),\n"
        "        actor_cadence_profile_sha256=sha256(args.actor_cadence_profile)))")
    text = once(text, "                    args.mode, output/'recording', args.frame_persistence_profile)]",
        "                    args.mode, output/'recording', args.frame_persistence_profile,\n"
        "                    args.actor_cadence_profile)]")
    text = once(text, '                record_main(heartbeat_start=start_transport)\n',
        '                record_main(heartbeat_start=start_transport, actor_trace_sink=actor_sink)\n')
    text = once(text, "        write_json(output/'result.json', result)\n",
        "        try:\n"
        "            result['actor_trace'] = actor_sink.persist(output/'actor-trace', context=context)\n"
        "        except BaseException as error:\n"
        "            result['actor_trace'] = dict(status='persistence_failed',\n"
        "                error=type(error).__name__, reason=str(error), fit_allowed=False)\n"
        "        if result['status'].startswith('completed') and result['actor_trace']['status'] != 'complete':\n"
        "            result.update(status='failed', error='IncompleteActorTrace',\n"
        "                reason='Actor trace incomplete; retain original candidate without fit')\n"
        "        write_json(output/'result.json', result)\n")
    text = once(text, "    parser.add_argument('--output', required=True)",
        "    parser.add_argument('--actor-cadence-profile', required=True)\n"
        "    parser.add_argument('--output', required=True)")
    return text


def auditor():
    text = checked('scripts/audit_ac_engine_ai_probe_host_v4.py')
    text = once(text, 'from flybrain_drive.ac_shared_texture import verify_shared_frame\n',
        'from flybrain_drive.ac_shared_texture import verify_shared_frame\n'
        'from flybrain_drive.actor_cadence_trace_v1 import ActorCadenceProfile, verify_actor_trace\n')
    text = once(text, "    parser.add_argument('--output', required=True)",
        "    parser.add_argument('--actor-trace', required=True)\n"
        "    parser.add_argument('--output', required=True)")
    text = once(text, "    profile = HostCopyProfile.from_record(\n",
        "    ActorCadenceProfile.from_record(manifest['config']['actor_cadence_profile'])\n"
        "    profile = HostCopyProfile.from_record(\n")
    text = once(text, "    write_json(run/'result.json', dict(status='completed', decisions=len(events),\n",
        "    if Path(args.actor_trace).resolve() != source.resolve().parent/'actor-trace':\n"
        "        raise ValueError('Actor trace must be attached to the same original parent')\n"
        "    recording_result = json.loads((source/'result.json').read_bytes())\n"
        "    ActorCadenceProfile.from_record(recording_result['actor_cadence_profile'])\n"
        "    if recording_result['actor_trace_decisions'] != len(events):\n"
        "        raise ValueError('Every original actor decision trace required')\n"
        "    actor_trace = verify_actor_trace(args.actor_trace, events,\n"
        "        expected_cleanup=recording_result['owned_cleanup'])\n"
        "    write_json(run/'result.json', dict(status='completed', decisions=len(events),\n"
        "        actor_cadence_trace_verified=actor_trace,\n")
    return text


def build():
    from flybrain_drive.actor_cadence_trace_v1 import ActorCadenceProfile

    header = '# ruff: noqa: E501 -- Exact source-derived nested context is intentional.\n'
    return {
        Path('scripts/record_ac_engine_ai_probe_actor_trace_v1.py'): (header+recorder()).encode(),
        Path('scripts/qualify_session_heartbeat_actor_trace_v1.py'): (header+parent()).encode(),
        Path('scripts/audit_ac_engine_ai_probe_actor_trace_v1.py'): (header+auditor()).encode(),
        Path('configs/ac-actor-cadence-trace-30ms-20261030.json'):
            (json.dumps(ActorCadenceProfile().record(), indent=2)+'\n').encode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-unsealed-new-sources', action='store_true')
    args = parser.parse_args()
    outputs = build()
    if args.write_unsealed_new_sources:
        for path, raw in outputs.items():
            path.write_bytes(raw)
    print(json.dumps(dict(status='unsealed_CPU_source_prototype_only',
        outputs={str(p): hashlib.sha256(raw).hexdigest() for p, raw in outputs.items()},
        role_or_capture_started=False, fit_allowed=False)))


if __name__ == '__main__':
    main()
