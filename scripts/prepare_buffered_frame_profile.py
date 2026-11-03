"""Build separate explicitly profiled recorder/runner/auditor; no live action."""

import argparse
import hashlib
import json
from pathlib import Path

BASES = {
    "scripts/record_ac_engine_ai_probe.py": (
        "a034c9f6255cbeb5129342f2b57e9c62246517c10aa390dcf2d389f91f608642"
    ),
    "scripts/qualify_session_heartbeat_evidence.py": (
        "f9edd60992c6d9f238b308e5b19312181823fa582f6006140cc1b2bf040e43fc"
    ),
    "scripts/audit_ac_engine_ai_probe.py": (
        "b1c540beea198a771fb840d9a21c48db4d302f40c1063d455d4aeaf432c91fba"
    ),
}


def checked(name):
    raw = Path(name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASES[name]:
        raise ValueError("Frozen original source differs: " + name)
    return raw.decode("utf8").replace("\r\n", "\n")


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Exactly one source-derived replacement required: " + old[:60])
    return source.replace(old, new)


def build():
    recorder = checked("scripts/record_ac_engine_ai_probe.py")
    recorder = once(
        recorder,
        "from flybrain_drive.evidence import begin_run, write_json\n",
        "from flybrain_drive.evidence import begin_run, write_json\n"
        "from flybrain_drive.frame_persistence_profile_v3 import (\n"
        "    BufferedFrameWriterFactory,\n    load_buffered_frame_profile,\n)\n",
    )
    recorder = once(
        recorder,
        "    'scripts/record_ac_wgc.py',\n",
        "    'scripts/record_ac_wgc.py',\n"
        "    'src/flybrain_drive/frame_persistence_profile_v3.py',\n",
    )
    recorder = once(
        recorder,
        "    parser.add_argument('--output', required=True)\n",
        "    parser.add_argument('--frame-persistence-profile', required=True)\n"
        "    parser.add_argument('--output', required=True)\n",
    )
    recorder = once(
        recorder,
        "    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n"
        "    profile = load_buffered_frame_profile(args.frame_persistence_profile)\n"
        "    if (not all((args.async_frame_persistence, args.mmf_metadata,\n"
        "                 args.shared_transport, args.batched_shared_copy,\n"
        "                 args.same_copy_native_authority, args.source_timing))\n"
        "            or args.image_format != 'bmp' or args.ai_steer_multiplier != .9\n"
        "            or args.flight_shadow_config\n"
        "            or (args.duration_seconds, args.speed_hint_kmh, args.pace_task) not in\n"
        "               ((3, 60, False), (30, 100, True))):\n"
        "        raise ValueError('Explicit3s60/30s pace100 MMF diagnostic required')\n"
        "    frame_writer_factory = BufferedFrameWriterFactory(profile)\n",
    )
    recorder = once(
        recorder,
        "                                     sensory_configuration=config))",
        "                                     sensory_configuration=config,\n"
        "        frame_persistence_profile=profile.record(),\n"
        "        frame_persistence_profile_source_sha256=sha256(args.frame_persistence_profile)))",
    )
    recorder = once(
        recorder,
        "    vision, child, log, shadow = None, None, None, None\n",
        "    result['frame_persistence_profile'] = profile.record()\n"
        "    vision, child, log, shadow = None, None, None, None\n",
    )
    async_line = " " * 35 + "async_frame_persistence=args.async_frame_persistence,\n"
    recorder = once(
        recorder,
        async_line,
        async_line
        + "                                   frame_writer_factory=frame_writer_factory,\n",
    )

    runner = checked("scripts/qualify_session_heartbeat_evidence.py")
    runner = once(
        runner,
        "def collection_arguments(mode, output):",
        "def collection_arguments(mode, output, profile_file):",
    )
    runner = once(
        runner,
        "    arguments = ['--config', 'configs/ac-csp-exterior-front-v3-retina.json',\n",
        "    arguments = ['--config', 'configs/ac-csp-exterior-front-v3-retina.json',\n"
        "        '--frame-persistence-profile', str(profile_file),\n",
    )
    runner = once(
        runner,
        "    from flybrain_drive.session_heartbeat_admission import refuse_leased_transport\n",
        "    from flybrain_drive.frame_persistence_profile_v3 import "
        + "load_buffered_frame_profile\n"
        "    from flybrain_drive.session_heartbeat_admission import refuse_leased_transport\n",
    )
    runner = once(
        runner,
        "    require_local_c_output(args.output)\n",
        "    profile = load_buffered_frame_profile(args.frame_persistence_profile)\n"
        "    require_local_c_output(args.output)\n",
    )
    runner = once(
        runner,
        "    context = None\n",
        "    result['frame_persistence_profile'] = profile.record()\n    context = None\n",
    )
    runner = once(
        runner,
        "from scripts.record_ac_engine_ai_probe import main as record_main",
        "from scripts.record_ac_engine_ai_probe_buffered import main as record_main",
    )
    runner = once(
        runner,
        "                    args.mode, output/'recording')]",
        "                    args.mode, output/'recording', args.frame_persistence_profile)]",
    )
    runner = once(
        runner,
        "    parser.add_argument('--output', required=True)\n",
        "    parser.add_argument('--frame-persistence-profile', required=True)\n"
        "    parser.add_argument('--output', required=True)\n",
    )

    auditor = checked("scripts/audit_ac_engine_ai_probe.py")
    auditor = once(
        auditor,
        "from flybrain_drive.immutable_frame_writer import verify_persistence_archive\n",
        "",
    )
    auditor = once(
        auditor,
        "from flybrain_drive.evidence import begin_run, write_json\n",
        "from flybrain_drive.evidence import begin_run, write_json\n"
        "from flybrain_drive.frame_persistence_profile_v3 import (\n"
        "    BufferedFrameProfile,\n    verify_buffered_persistence_archive,\n)\n",
    )
    auditor = once(
        auditor,
        "        persisted_rows = verify_persistence_archive(camera, expected_mmf=mmf) "
        "if deferred else None",
        "        persisted_rows = (verify_buffered_persistence_archive(\n"
        "            camera, expected_mmf=mmf) if deferred else None)",
    )
    auditor = once(
        auditor,
        "    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n"
        "    if args.repack_config or not args.require_aid_off:\n"
        "        raise ValueError('Profile qualification forbids repack and requires aidOFF')\n",
    )
    auditor = once(
        auditor,
        "    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))\n",
        "    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))\n"
        "    profile = BufferedFrameProfile.from_record(\n"
        "        manifest['config']['frame_persistence_profile'])\n"
        "    if not all(manifest['config']['arguments'].get(k) for k in\n"
        "               ('async_frame_persistence', 'mmf_metadata',\n"
        "                'same_copy_native_authority')):\n"
        "        raise ValueError('Explicit bounded MMF same-copy source required')\n",
    )
    auditor = once(
        auditor,
        "    write_json(run/'result.json', dict(status='completed', decisions=len(events),",
        "    write_json(run/'result.json', dict(status='completed', decisions=len(events),\n"
        "        bounded_frame_profile_verified=profile.record(),\n",
    )
    return {
        Path("scripts/record_ac_engine_ai_probe_buffered.py"): recorder.encode(),
        Path("scripts/qualify_session_heartbeat_evidence_buffered.py"): runner.encode(),
        Path("scripts/audit_ac_engine_ai_probe_buffered.py"): auditor.encode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    outputs = build()
    for path, content in outputs.items():
        path.write_bytes(content)
    receipt = dict(
        base_sha256=BASES,
        outputs={str(p): hashlib.sha256(b).hexdigest() for p, b in outputs.items()},
        live_actions=0,
        old_sources_unchanged=True,
    )
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
