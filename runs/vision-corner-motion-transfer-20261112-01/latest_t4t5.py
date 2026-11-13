"""Evaluate latest T4/T5 features on the same held-out boundary proposals."""
from pathlib import Path

import supported
import transfer_eval as protocol


def proposals(hashes):
    hashes.add(Path(__file__))
    return supported.proposals(hashes)


if __name__ == "__main__":
    protocol.OUT = Path(__file__).parent / "latest-t4t5"
    protocol.OUT.mkdir(exist_ok=False)
    protocol.MODES = ("boundaries", "boundaries_latest_t4t5")
    protocol.load_proposals = proposals
    protocol.main()
