# ruff: noqa: E501 -- Exact role-edit source fragments intentionally retain bytes.
"""Exact six-edit NEW B1 TRAIN parent derivative; no capture or role seal."""

import argparse
import hashlib
from pathlib import Path

BASE = Path("scripts/qualify_session_heartbeat_actor_trace_v1.py")
TARGET = Path("scripts/record_pedal_b1.py")
BASE_SHA256 = "09f25ce66f619ee52abc5b3a32c7cbb380bbe9c21d6cecc72d1ca4b113f3ef64"


def edits():
    return [
        (
            "from flybrain_drive.graph import sha256\n",
            "from flybrain_drive.graph import sha256\nfrom flybrain_drive.pedal_train_b1 import collector_declaration\n",
        ),
        ("def run(args):\n", "def run(args):\n    sealed_train = collector_declaration(args)\n"),
        (
            "role='diagnostic', fit_allowed=False, source_candidate_sha256=sha256(CANDIDATE),\n"
            "        actor_cadence_profile=actor_profile.record(),\n"
            "        actor_cadence_profile_sha256=sha256(args.actor_cadence_profile)))",
            "role='train', fit_allowed=False, source_candidate_sha256=sha256(CANDIDATE),\n"
            "        actor_cadence_profile=actor_profile.record(),\n"
            "        actor_cadence_profile_sha256=sha256(args.actor_cadence_profile),\n"
            "        train_nonce=sealed_train['nonce'], train_declaration_sha256=sealed_train['sha256'],\n"
            "        original_train_declaration=sealed_train['value']))\n"
            "    with (output/'original-train-declaration.json').open('xb') as stream:\n"
            "        stream.write(sealed_train['raw'])\n"
            "    with (output/'executed-program.py').open('xb') as stream:\n"
            "        stream.write(Path(__file__).read_bytes())",
        ),
        (
            "result = dict(status='failed', role='diagnostic', fit_allowed=False, optimizer_updates=0,",
            "result = dict(status='failed', role='train', fit_allowed=False, optimizer_updates=0,\n"
            "        train_nonce=sealed_train['nonce'], train_declaration_sha256=sealed_train['sha256'],",
        ),
        (
            "else 'completed_diagnostic_transport_collection'",
            "else 'completed_train_candidate_b1_transport_collection'",
        ),
        (
            "    parser.add_argument('--mode', choices=['camera3', 'pace30'], required=True)",
            "    parser.add_argument('--train-declaration', required=True)\n"
            "    parser.add_argument('--mode', choices=['camera3', 'pace30'], required=True)",
        ),
    ]


def build(source):
    if hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise ValueError("Exact frozen R2 actor parent required")
    text = source.decode("utf8")
    for old, new in edits():
        if text.count(old) != 1:
            raise ValueError("Exact six-edit fragment differs")
        text = text.replace(old, new)
    return text.encode("utf8")


def inverse(source):
    text = source.decode("utf8")
    for old, new in reversed(edits()):
        if text.count(new) != 1:
            raise ValueError("B1 role edit allowlist differs")
        text = text.replace(new, old)
    return text.encode("utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-new-parent", action="store_true")
    args = parser.parse_args()
    raw = build(BASE.read_bytes())
    assert inverse(raw) == BASE.read_bytes()
    if args.write_new_parent:
        with TARGET.open("xb") as stream:
            stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
