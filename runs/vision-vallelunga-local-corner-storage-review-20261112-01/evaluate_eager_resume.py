
"""Verify the completed eager rehearsal and run the locked evaluation in a fresh process."""
from pathlib import Path
from unittest.mock import patch

import evaluate_eager as eager
from train_vision_teacher_line import digest, read

proof = read(eager.parity_receipt)
assert proof["status"] == "verified"
assert proof["wrapper_sha256"] == digest(Path(eager.__file__))
assert proof["lock_sha256"] == digest(eager.protocol / "lock.json")
for mode, values in proof["modes"].items():
    assert values["eager_sha256"] == digest(eager.directory / "eager-silverstone-rehearsal" / mode / "features.npz")
    assert values["frozen_graph_sha256"] == digest(eager.rehearsal / mode / "features.npz")
    assert values["max_abs_error"] <= 1e-6

with (patch.object(eager.extraction, "setup", eager.eager_setup),
      patch.object(eager.extraction, "extract_pair", eager.report_pair),
      patch.object(eager.storage.admission, "verify_buffered_persistence_archive",
                   eager.storage.verify_actual_writer)):
    eager.storage.entry.main()
    if len(eager.storage.used) != 1:
        raise RuntimeError("Original MMF archive was not independently verified once")
