"""Inventory or seal an explicit future role declaration; never launch collection."""

import argparse
import json
from pathlib import Path

from flybrain_drive.pedal_train_candidate import (
    SOURCE_FILES,
    seal_declaration,
    sha_file,
    validate_declaration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration',
                        default='configs/flight-pedal-new-train-candidate-20261030.json')
    parser.add_argument('--seal-output')
    args = parser.parse_args()
    value = json.loads(Path(args.declaration).read_bytes())
    validate_declaration(value, sealed=False)
    if args.seal_output:
        print(json.dumps(seal_declaration(value, args.seal_output), indent=2))
    else:
        print(json.dumps(dict(status='draft_inventory_no_live_actions',
            bindings=value['bindings'], fit_allowed=False, collection_started=False,
            required_source_sha256={p: sha_file(p) for p in SOURCE_FILES}), indent=2))


if __name__ == '__main__':
    main()
