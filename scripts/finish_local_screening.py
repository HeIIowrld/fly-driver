"""Wait for the local pilot, replicate passing changes, prepare additional HPC trials."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from scripts.run_campaign import run_campaign
from scripts.select_local_candidates import select_candidates


def finish(config, stage_one, output, stage_one_pid):
    run = begin_run(output, {"config": config, "stage_one": str(stage_one),
                             "stage_one_pid": stage_one_pid,
                             "existing_remote_campaign": "265295/265296 unchanged"})
    deadline = time.monotonic() + 8 * 3600

    def state(value, **extra):
        write_json(run / "lifecycle.json", {"state": value, "pid": os.getpid(), **extra})

    try:
        missing = 0
        while not (stage_one / "result.json").exists():
            state("waiting_for_stage_one")
            try:
                os.kill(stage_one_pid, 0)
                missing = 0
            except ProcessLookupError:
                missing += 1
            if missing >= 3 or time.monotonic() >= deadline:
                raise RuntimeError("Stage one exited without a result or exceeded the wait limit")
            time.sleep(20)
        summary = json.loads((stage_one / "summary.json").read_text())
        first = select_candidates(summary)
        write_json(run / "stage-one-selection.json", first)
        candidates = first["replication_candidates"]
        if candidates:
            replica = copy.deepcopy(config)
            replica["campaign"]["seeds"] = [config["local_screening"]["replication_seed"]]
            replica["campaign"]["arms"] = [a for a in config["campaign"]["arms"]
                                            if a["name"] in ["full_mixed", *candidates]]
            write_json(run / "replication-config.json", replica)
            state("replicating", candidates=candidates)
            run_campaign(replica, run / "replication")
            second = json.loads((run / "replication/summary.json").read_text())
            summary["runs"].extend(second["runs"])
        final = select_candidates(summary)
        write_json(run / "combined-summary.json", summary)
        write_json(run / "selection.json", final)
        confirmed = final["remote_addition_candidates"]
        if confirmed:
            remote = json.loads(Path("configs/campaign-vision-tv-gpu.json").read_text())
            remote["purpose"] = "Replicated local candidates; existing remote array retained"
            remote["campaign"]["seeds"] = [4101, 4102]
            remote["campaign"]["arms"] = [a for a in config["campaign"]["arms"]
                                           if a["name"] in ["full_mixed", *confirmed]]
            # Keep the longer remote PPO budget and original independent test suite.
            remote["local_selection"] = final
            write_json(run / "remote-addition-config.json", remote)
        state("completed", candidates=confirmed, replicated=bool(candidates))
        write_json(run / "result.json", {"status": "completed", "candidates": confirmed})
    except Exception as error:
        state("failed", reason=type(error).__name__, message=str(error))
        write_json(run / "result.json", {"status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-one", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-one-pid", type=int, required=True)
    args = parser.parse_args()
    finish(json.loads(args.config.read_text()), args.stage_one, args.output, args.stage_one_pid)
