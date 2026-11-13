"""Replay the entry-aware checkpoint with the identical frozen downstream controls."""
from pathlib import Path

import evaluate_shape as evaluator
from train_entries import OUT


if __name__ == "__main__":
    evaluator.OUT = OUT
    evaluator.main()
    path = OUT / "result.json"
    result = evaluator.old.read(path)
    result["source_sha256"][str(Path(__file__).resolve())] = evaluator.old.digest(Path(__file__))
    result["teacher_policy"] = "physical_candidate_with_offline_entry_agreement"
    evaluator.old.write(path, result)
