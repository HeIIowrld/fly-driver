"""Prepare an explicitly UNSEALED pedal training template; no Torch/data imports."""

import argparse
import hashlib
import json
from pathlib import Path

from flybrain_drive.flight_pedal_train_plan_v1 import (
    DRAFT_PATH,
    NEW_SOURCES,
    preparation_template,
    sha,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", default=DRAFT_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = Path(args.draft).read_bytes()
    prepared = preparation_template(raw)
    output = Path(args.output)
    if output.drive.upper() != "C:" or output.exists():
        raise ValueError("New unique localC preparation output only")
    output.mkdir(parents=True, exist_ok=False)
    (output / "unsealed-execution-template.json").write_text(
        json.dumps(prepared, indent=2) + "\n", encoding="utf8"
    )
    sources = {path: sha(path) for path in NEW_SOURCES}
    result = dict(
        schema="flybrain.first_pedal_training_preparation.v1",
        status="source_prepared_UNSEALED_no_model_or_data_execution",
        fit_allowed=False,
        model_execution_authorized=False,
        optimizer_updates=0,
        data_payloads_opened=0,
        model_loaded=False,
        device_queries=0,
        live_actions=0,
        protocol_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=sources,
        unsealed_template_sha256=sha(output / "unsealed-execution-template.json"),
        future_V4_loader_binding_known=False,
        actual_new_TRAIN_admission_known=False,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    (output / "executed-preparer.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
