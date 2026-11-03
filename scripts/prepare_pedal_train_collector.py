"""Generate separately versioned TRAIN parent; qualified diagnostic sources untouched."""

import argparse
from pathlib import Path

from flybrain_drive.pedal_train_candidate import sha_bytes

BASE = Path('scripts/qualify_session_heartbeat_evidence_buffered.py')
TARGET = Path('scripts/record_pedal_train_candidate.py')


def edits():
    return [
        ("from flybrain_drive.graph import sha256\n",
         "from flybrain_drive.graph import sha256\n"
         "from flybrain_drive.pedal_train_candidate import collector_declaration\n"),
        ('def run(args):\n', 'def run(args):\n    sealed_train = collector_declaration(args)\n'),
        ("role='diagnostic', fit_allowed=False, source_candidate_sha256=sha256(CANDIDATE)))",
         "role='train', fit_allowed=False, source_candidate_sha256=sha256(CANDIDATE),\n"
         "        train_nonce=sealed_train['nonce'],\n"
         "        train_declaration_sha256=sealed_train['sha256'],\n"
         "        original_train_declaration=sealed_train['value']))\n"
         "    with (output/'original-train-declaration.json').open('xb') as declaration_file:\n"
         "        declaration_file.write(sealed_train['raw'])\n"
         "    with (output/'executed-program.py').open('xb') as executed_source:\n"
         "        executed_source.write(Path(__file__).read_bytes())"),
        ("result = dict(status='failed', role='diagnostic', fit_allowed=False, "
         "optimizer_updates=0,",
         "result = dict(status='failed', role='train', fit_allowed=False, optimizer_updates=0,\n"
         "        train_nonce=sealed_train['nonce'],\n"
         "        train_declaration_sha256=sealed_train['sha256'],"),
        ("else 'completed_diagnostic_transport_collection'",
         "else 'completed_train_candidate_transport_collection'"),
        ("    parser.add_argument('--mode', choices=['camera3', 'pace30'], required=True)",
         "    parser.add_argument('--train-declaration', required=True)\n"
         "    parser.add_argument('--mode', choices=['camera3', 'pace30'], required=True)"),
    ]


def build(source):
    result = source.decode('utf8')
    for old, new in edits():
        if result.count(old) != 1:
            raise ValueError('Qualified base differs; no adaptive rewrite: '+old)
        result = result.replace(old, new)
    return result.encode('utf8')


def inverse(candidate):
    result = candidate.decode('utf8')
    for old, new in reversed(edits()):
        if result.count(new) != 1:
            raise ValueError('Candidate allowlist differs')
        result = result.replace(new, old)
    return result.encode('utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-base-sha256', required=True)
    args = parser.parse_args()
    source = BASE.read_bytes()
    if sha_bytes(source) != args.expected_base_sha256:
        raise ValueError('Exact reviewed buffered source required')
    candidate = build(source)
    assert inverse(candidate) == source
    with TARGET.open('xb') as stream:
        stream.write(candidate)
    print(sha_bytes(candidate))


if __name__ == '__main__':
    main()
