"""Replay the rolling body-frame line with and without the new fixed corner forecasts."""
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
LINE = OUT.parent / "vision-corner-onset-through-line-20261113-01"
sys.path.insert(0, str(LINE))
from rolling_lines import LIMITS, classic, corridor, rolling_line, sequence, sensor_rows, alt
from left_policy import dependencies
from fixed_left import CLI, EVAL, METHOD, digests, frames, keys, load_sensor, old


def main():
    _, settings, _, _ = dependencies()
    comparison_path = OUT / "comparison.json"
    comparison = old.read(comparison_path)
    line_receipt = LINE / "line-result.json"
    previous_lines = old.read(line_receipt)
    reports = {}
    for track, name in EVAL:
        directory = CLI / ("vallelunga-retry" if name == "vallelunga" else name)
        receipt = directory / "result.json"
        sensor_path = directory / "recovered-sensors.npz"
        data = load_sensor(sensor_path, old.read(receipt)["recovered_input_sha256"])
        metadata = sensor_rows(data, settings)
        current = OUT / f"{track}-forecasts.jsonl"
        forecasts = frames(current, comparison["tracks"][track]["forecasts_sha256"])
        previous = LINE / f"{track}-lines.jsonl"
        baseline = frames(previous, previous_lines["tracks"][track]["lines_sha256"])
        if keys(metadata) != keys(forecasts) or keys(metadata) != keys(baseline):
            raise ValueError("Line and forecast timelines differ")
        continuity = [False, *(alt.adjacent(a, b) for a, b in zip(metadata, metadata[1:]))]
        body = data["motion"]
        stations = data["stations"]
        aligned = sequence(stations, continuity, data["sections"], body)
        path = None
        changes = []
        coverage = 0
        for i, (row, turn, control) in enumerate(zip(metadata, forecasts, baseline, strict=True)):
            moving = np.isfinite(body[i]).all() and body[i, 1] >= old.MIN_SPEED_MPS
            section = corridor(stations, aligned[i]) if moving else None
            step = alt.transform(body[i], body[i-1]) if i and moving and continuity[i] and np.isfinite(body[i-1]).all() else None
            prior = alt.advance(path, *step) if step is not None and path is not None else None
            forecast = turn["forecasts"][METHOD]
            classic_line = control["lines"]["classic"]
            line = rolling_line(section, prior, forecast, classic_line)
            coverage += int(line is not None)
            path = np.column_stack((line["x_m"], line["y_m"])) if line is not None else None
            if line != control["lines"]["preview"]:
                changes.append(dict(run=row["run"], frame=row["frame"], forecast=forecast,
                                    before=control["lines"]["preview"], after=line))
        witness = OUT / f"{track}-line-changes.jsonl"
        old.jsonl(witness, changes)
        reports[track] = dict(frames=len(metadata), rolling_coverage=coverage, changed_lines=len(changes),
                              rows_sha256=old.digest(witness),
                              source_sha256=digests((receipt, sensor_path, current, previous)))
        print(track, coverage, len(changes), flush=True)
    old.write(OUT / "rolling-line-comparison.json", dict(
        tracks=reports, limits=LIMITS, seed=None, source_revision=old.revision(),
        source_sha256=digests((Path(__file__), Path(rolling_line.__code__.co_filename), line_receipt, comparison_path)),
        termination="archive_end"))


if __name__ == "__main__":
    main()
