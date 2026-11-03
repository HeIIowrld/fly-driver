"""Development-only checkpoint selection; locked tests never enter this module."""

import shutil
import time
from pathlib import Path

from .evidence import write_json
from .graph import sha256


def validation_score(evaluation):
    aggregate = evaluation["aggregate"]
    # Completion outranks partial progress; failure reduction outranks distance.
    return (
        aggregate["completion_rate"], -aggregate["failure_rate"],
        aggregate["mean_remaining_course_fraction"],
        -aggregate["mean_lateral_error_rms"],
    )


def validate_training_settings(config):
    settings = config.get("validation")
    if not settings:
        return
    if settings.get("role") != "development":
        raise ValueError("Training validation must be explicitly development-only")
    if int(settings.get("interval", 8)) < 1:
        raise ValueError("Validation interval must be positive")
    if not settings.get("suite", {}).get("cases"):
        raise ValueError("Validation needs declared development cases")
    if int(settings.get("regression_patience", 0)) < 0:
        raise ValueError("Regression patience cannot be negative")
    ratio = float(settings.get("regression_ratio", 0.75))
    if not 0 < ratio < 1:
        raise ValueError("Regression ratio must be in (0, 1)")
    if not 0 < float(settings.get("minimum_progress_ratio", .9)) <= 1:
        raise ValueError("Selected checkpoint progress ratio must be in (0, 1]")


def ensure_test_separation(validation, test):
    if not validation:
        return
    dev_seeds = {int(c.get("seed", 8001)) for c in validation["suite"]["cases"]}
    test_seeds = {int(c.get("seed", 8001)) for c in test["cases"]}
    if dev_seeds & test_seeds:
        raise ValueError("Development validation and end evaluation seeds overlap")


class TrainingValidator:
    def __init__(self, config, run):
        validate_training_settings(config)
        self.settings = config.get("validation")
        self.run = Path(run)
        self.history = []
        self.best_score = None
        self.regressions = 0
        self.seconds = 0.0

    def due(self, update, final=False):
        return bool(self.settings) and (
            update == 0 or final or update % int(self.settings.get("interval", 8)) == 0
        )

    def evaluate(self, checkpoint, update):
        from .evaluation import evaluate_suite

        if not self.settings:
            return False
        started = time.perf_counter()
        evaluation = evaluate_suite(
            checkpoint, self.run / "development" / f"update-{update:06d}",
            self.settings["suite"], record=bool(self.settings.get("record", True)),
            record_every=int(self.settings.get("record_every", 16)),
        )
        self.seconds += time.perf_counter() - started
        score = validation_score(evaluation)
        initial = self.history[0]["aggregate"] if self.history else evaluation["aggregate"]
        current = evaluation["aggregate"]
        # An idle policy must not win merely by surviving the evaluation budget.
        eligible = (
            current["completion_rate"] > initial["completion_rate"]
            or current["mean_progress_delta"] >= max(0, initial["mean_progress_delta"])
            * float(self.settings.get("minimum_progress_ratio", .9))
        )
        selected = self.best_score is None or (eligible and score > self.best_score)
        if selected:
            shutil.copy2(checkpoint, self.run / "best-checkpoint.pt")
            self.best_score = score
        row = {"completed_updates": update, "checkpoint_sha256": sha256(checkpoint),
               "aggregate": evaluation["aggregate"], "score": list(score),
               "selected": selected, "progress_floor_passed": eligible,
               "cases": evaluation["cases"]}
        self.history.append(row)
        regressed = (
            current["completion_rate"] < initial["completion_rate"]
            or current["failure_rate"] > initial["failure_rate"]
            or (initial["mean_progress_delta"] > 0 and current["mean_progress_delta"]
                < float(self.settings.get("regression_ratio", .75))
                * initial["mean_progress_delta"])
        )
        self.regressions = self.regressions + 1 if regressed else 0
        patience = int(self.settings.get("regression_patience", 0))
        stop = patience > 0 and self.regressions >= patience
        selected_row = next(r for r in reversed(self.history) if r["selected"])
        write_json(self.run / "validation.json", {
            "role": "development", "history": self.history,
            "selected_completed_updates": selected_row["completed_updates"],
            "selected_checkpoint_sha256": sha256(self.run / "best-checkpoint.pt"),
            "stop_requested": stop, "consecutive_regressions": self.regressions,
            "validation_seconds": self.seconds,
        })
        print(f"development_update={update} progress={current['mean_progress_delta']:.2f} "
              f"selected={selected} regression_stop={stop}", flush=True)
        return stop
