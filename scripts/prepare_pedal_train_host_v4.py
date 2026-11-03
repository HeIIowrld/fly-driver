"""Prepare NEW-only prospective TRAIN hostV4 sources; never seal a role or run capture.

Old1823 sources and diagnostic data remain untouched. The draft's actual pipeline,
plant, nonce and output bindings stay null until separate reviewed qualification.
"""
# ruff: noqa: E501 -- Source-derived fragments intentionally retain exact context.

import argparse
import hashlib
import json
from pathlib import Path

BASES = {
    'src/flybrain_drive/pedal_train_candidate.py': 'b1dc263f51d8e06a5ba2c9754ef1c4f7b5ef9b09b3c70bfa65a70175c9ad877c',
    'src/flybrain_drive/pedal_train_loader.py': 'e4c88026bca421637e19eea278f4a49f3faf3c315360f84e3c3b4a4533b49f4c',
    'scripts/prepare_pedal_train_collector.py': 'e6ed057c27648401d904b68660d20f50678ad1955c4dd94d1f8e62659aec0ef2',
    'scripts/prepare_pedal_train_declaration.py': 'a93cb9c29a0f9c01388fbdbbd9031bddb655c8fac7f2078541fe58a9ecb4c453',
    'scripts/load_pedal_train_candidate.py': '44def7dc34987494456f7e57b14b9a11f248f08ca043e506b81e08c1164d31e7',
    'configs/flight-pedal-new-train-candidate-20261030.json': '44a819dfc21be23de565a26a3951ef050974cdaccb3af853f032f33655abe9f3',
    'scripts/qualify_session_heartbeat_evidence_host_v4.py': 'b51bef99eb39f29d3644cb5c56953ceaaf66c6b86c84aaa7816d36949cddf608',
    'tests/test_pedal_train_candidate.py': 'f32b989e9fb0cc494ead2c9b20c7fb91ce9468f66918e88f26f68f566bd9bfb7',
}
RENAMES = {
    'qualify_session_heartbeat_evidence_buffered': 'qualify_session_heartbeat_evidence_host_v4',
    'record_ac_engine_ai_probe_buffered': 'record_ac_engine_ai_probe_host_v4',
    'frame_persistence_profile_v3': 'frame_persistence_profile_v4',
    'ac_csp_vision.py': 'ac_csp_vision_host_v4.py',
    'pedal_train_candidate': 'pedal_train_candidate_host_v4',
    'pedal_train_loader': 'pedal_train_loader_host_v4',
    'record_pedal_train_candidate': 'record_pedal_train_candidate_host_v4',
    'prepare_pedal_train_collector': 'prepare_pedal_train_collector_host_v4',
    'prepare_pedal_train_declaration': 'prepare_pedal_train_declaration_host_v4',
    'load_pedal_train_candidate': 'load_pedal_train_candidate_host_v4',
    'ac-frame-persistence-buffered32-20261030.json': 'ac-frame-host-v4-20261030.json',
    'flight-pedal-new-train-candidate-20261030.json': 'flight-pedal-new-train-candidate-host-v4-20261030.json',
}


def checked(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASES[path]:
        raise ValueError('Frozen original source differs; no adaptive rewrite: '+path)
    return raw.decode().replace('\r\n','\n')


def renamed(source):
    # Longer names first: a generic candidate substring must not rename a longer
    # record/load CLI twice. Replacement tokens are applied to the ORIGINAL text.
    import re

    pattern = '|'.join(re.escape(k) for k in sorted(RENAMES,key=len,reverse=True))
    return re.sub(pattern, lambda m: RENAMES[m.group()], source)


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Exact unique derivative edit required: '+old[:80])
    return source.replace(old,new)


def build():
    candidate = renamed(checked('src/flybrain_drive/pedal_train_candidate.py'))
    candidate = once(candidate, "SCHEMA = 'flybrain.pedal_train_candidate_host_v4.declaration.v1'",
        "SCHEMA = 'flybrain.pedal_train_candidate.declaration.host_v4.v1'")
    candidate = once(candidate, "    'scripts/prepare_pedal_train_collector_host_v4.py', 'scripts/prepare_pedal_train_declaration_host_v4.py',",
        "    'scripts/prepare_pedal_train_host_v4.py', 'scripts/prepare_frame_host_v4.py',\n"
        "    'scripts/prepare_pedal_train_collector_host_v4.py',\n"
        "    'scripts/prepare_pedal_train_declaration_host_v4.py',")
    candidate = once(candidate,
        "    'src/flybrain_drive/pedal_train_candidate_host_v4.py', 'src/flybrain_drive/pedal_train_loader_host_v4.py',",
        "    'src/flybrain_drive/pedal_train_candidate_host_v4.py',\n"
        "    'src/flybrain_drive/pedal_train_loader_host_v4.py',")
    candidate = once(candidate, "    if value.get('body_contract') != 'configs/ac-virtual-navigation-body-v2.json':",
        "    if (value.get('profile') != 'configs/ac-frame-host-v4-20261030.json'\n"
        "            or value.get('host_copy_profile_schema') != 'flybrain.frame_host_copy_profile.v4'):\n"
        "        raise ValueError('Explicit separately qualified hostV4 profile required')\n"
        "    if value.get('body_contract') != 'configs/ac-virtual-navigation-body-v2.json':")
    candidate = once(candidate, "    if not sealed:\n        return\n    bindings = value.get('bindings', {})",
        "    if not sealed:\n        return\n"
        "    if value.get('actual_pipeline_qualification_completed') is not True:\n"
        "        raise ValueError('Actual reviewed hostV4 pipeline qualification required '\n"
        "                         'before role seal/capture')\n"
        "    bindings = value.get('bindings', {})")
    loader = renamed(checked('src/flybrain_drive/pedal_train_loader.py'))
    loader = once(loader, 'from .ac_teacher_physics import decode_teacher_physics\n',
        'from .ac_teacher_physics import decode_teacher_physics\n'
        'from .frame_persistence_profile_v4 import HostCopyProfile\n')
    loader = once(loader, "    'scripts/record_ac_wgc.py', 'src/flybrain_drive/frame_persistence_profile_v4.py',\n",
        "    'scripts/record_ac_wgc.py', 'src/flybrain_drive/frame_persistence_profile_v4.py',\n"
        "    'src/flybrain_drive/ac_shared_texture_host_v4.py',\n"
        "    'src/flybrain_drive/collection_cleanup_v4.py',\n")
    loader = once(loader, "    args = child_manifest['config']['arguments']\n",
        "    HostCopyProfile.from_record(child_manifest['config'].get('frame_persistence_profile'))\n"
        "    HostCopyProfile.from_record(result.get('frame_persistence_profile'))\n"
        "    args = child_manifest['config']['arguments']\n"
        "    if (not isinstance(args.get('frame_persistence_profile'), str)\n"
        "            or sha_file(args['frame_persistence_profile']) != declaration['source_sha256'][\n"
        "                'configs/ac-frame-host-v4-20261030.json']):\n"
        "        raise ValueError('Actual original child hostV4 profile attachment differs')\n")
    # Actual1845 diagnostic metadata exposed this previously unexecuted old bug.
    # Only the new derivative changes it; no role or data from1845 is promoted.
    loader = once(loader, "if len(events) != result['observations'] or len(events) > 2048:",
        "if len(events) != result['decisions'] or len(events) > 2048:")
    loader = once(loader, "    for path, digest in declaration['source_sha256'].items():\n",
        "    if type(result.get('decisions')) is not int or not 1 <= result['decisions'] <= 2048:\n"
        "        raise ValueError('Original recorder decisions count required; no observations alias')\n"
        "    for path, digest in declaration['source_sha256'].items():\n")
    builder = renamed(checked('scripts/prepare_pedal_train_collector.py'))
    # Bootstrap via the immutable old pure six-edit builder before new AST closure
    # modules exist, then rename only its added candidate import.
    from scripts.prepare_pedal_train_collector import build as train_parent

    parent = train_parent(checked('scripts/qualify_session_heartbeat_evidence_host_v4.py').encode())
    parent = parent.replace(b'from flybrain_drive.pedal_train_candidate import collector_declaration',
        b'from flybrain_drive.pedal_train_candidate_host_v4 import collector_declaration')
    declaration_cli = renamed(checked('scripts/prepare_pedal_train_declaration.py'))
    load_cli = renamed(checked('scripts/load_pedal_train_candidate.py'))
    load_cli = once(load_cli, 'flybrain.pedal_train_candidate_host_v4.whole_loader.v1',
        'flybrain.pedal_train_candidate.whole_loader.host_v4.v1')
    draft = json.loads(checked('configs/flight-pedal-new-train-candidate-20261030.json'))
    draft.update(schema='flybrain.pedal_train_candidate.declaration.host_v4.v1',
        profile='configs/ac-frame-host-v4-20261030.json',
        host_copy_profile_schema='flybrain.frame_host_copy_profile.v4',
        actual_pipeline_qualification_completed=False,
        required_external_gates=['Actual hostV4 camera3+pace30 source-bound qualification',
            *draft['required_external_gates']],
        provenance='NEW-only source derivative; all actual role/identity/calibration/qualification bindings null')
    tests = renamed(checked('tests/test_pedal_train_candidate.py'))
    tests = once(tests, "    value.update(status='sealed_before_capture', nonce='a'*32,",
        "    value.update(status='sealed_before_capture', nonce='a'*32,\n"
        "        actual_pipeline_qualification_completed=True,")
    tests = once(tests, "match='Fresh'", "match='Fresh|Actual reviewed hostV4'")
    tests = once(tests, "    args.update(flight_shadow_config=None, duration_seconds=30)",
        "    from flybrain_drive.frame_persistence_profile_v4 import HostCopyProfile\n"
        "    args.update(flight_shadow_config=None, duration_seconds=30,\n"
        "        frame_persistence_profile='configs/ac-frame-host-v4-20261030.json')")
    tests = once(tests, "json.dumps(dict(config=dict(arguments=args)))",
        "json.dumps(dict(config=dict(arguments=args, frame_persistence_profile=HostCopyProfile().record())))")
    tests = once(tests, "dict(status='completed', observations=3,",
        "dict(status='completed', decisions=3, frame_persistence_profile=HostCopyProfile().record(),")
    tests = once(tests, "'src/flybrain_drive/ac_shared_texture.py'",
        "'src/flybrain_drive/ac_shared_texture_host_v4.py'")
    tests = '# ruff: noqa: E501 -- Exact inherited synthetic fixture context is intentional.\n'+tests
    return {
        Path('src/flybrain_drive/pedal_train_candidate_host_v4.py'):candidate.encode(),
        Path('src/flybrain_drive/pedal_train_loader_host_v4.py'):loader.encode(),
        Path('scripts/record_pedal_train_candidate_host_v4.py'):parent,
        Path('scripts/prepare_pedal_train_collector_host_v4.py'):builder.encode(),
        Path('scripts/prepare_pedal_train_declaration_host_v4.py'):declaration_cli.encode(),
        Path('scripts/load_pedal_train_candidate_host_v4.py'):load_cli.encode(),
        Path('configs/flight-pedal-new-train-candidate-host-v4-20261030.json'):
            (json.dumps(draft,indent=2)+'\n').encode(),
        Path('tests/test_pedal_train_host_v4_inherited.py'):tests.encode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-unfrozen-new-sources',action='store_true')
    args = parser.parse_args()
    outputs = build()
    if args.write_unfrozen_new_sources:
        for path,raw in outputs.items():
            path.write_bytes(raw)
    print(json.dumps(dict(status='source_preparation_only_no_role_seal',
        outputs={str(p):hashlib.sha256(raw).hexdigest()for p,raw in outputs.items()},
        collection_started=False,fit_allowed=False,actual_bindings_unsealed=True)))


if __name__ == '__main__':
    main()
