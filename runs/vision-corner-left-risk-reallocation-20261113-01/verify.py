"""Verify fixed corner artifacts and independently recount baseline and revised metrics."""
from pathlib import Path

OUT = Path(__file__).resolve().parent
from replay_independent import check_extraction, load_fixed_policy, validate_transfer
from left_policy import select
from fixed_left import CLI, EVAL, METHOD, PER_RECORDING, digests, frames, keys, old, reference, score


def main():
    receipt, policy, _, _, _, _ = load_fixed_policy()
    check_extraction(policy)
    comparison_path = OUT / "comparison.json"
    comparison = old.read(comparison_path)
    if old.digest(receipt) != comparison["freeze_sha256"]:
        raise ValueError("Comparison used a different frozen policy")
    line_path = OUT / "rolling-line-comparison.json"
    lines = old.read(line_path)
    for path, digest in lines["source_sha256"].items():
        assert old.digest(Path(path)) == digest
    revised_results = {}
    for track, name in EVAL:
        saved = comparison["tracks"][track]
        for path, digest in saved["source_sha256"].items():
            assert old.digest(Path(path)) == digest
        directory = CLI / ("vallelunga-retry" if name == "vallelunga" else name)
        prior = directory / "predictions/forecasts.jsonl"
        baseline = frames(prior, old.read(prior.with_name("result.json"))["forecasts_sha256"])
        revised = frames(OUT / f"{track}-forecasts.jsonl", saved["forecasts_sha256"])
        assert keys(baseline) == keys(revised)
        assert all(a == b for a, b in zip(baseline, revised, strict=True)
                   if a["forecasts"][METHOD]["state"] == "right")
        assert all(a["forecasts"]["temporal"] == b["forecasts"]["temporal"]
                   for a, b in zip(baseline, revised, strict=True))
        truth, _ = reference(track)
        before, after = score(baseline, truth), score(revised, truth)
        assert (before, after) == (saved["before"], saved["after"])
        assert after["approaches"]["1"]["joint_hit"] > before["approaches"]["1"]["joint_hit"]
        assert after["approaches"]["2"]["joint_hit"] == before["approaches"]["2"]["joint_hit"]
        assert all(after["counts"].get(k, 0)-before["counts"].get(k, 0) <= cap
                   for k, cap in PER_RECORDING.items())
        assert sum(after["counts"].values()) == len(truth)
        assert after["states"].get("no_output", 0) == 0
        for side in ("1", "2"):
            data = after["approaches"][side]
            assert data["joint_hit"] + data["miss"] == data["frames"]
        line = lines["tracks"][track]
        assert line["changed_lines"] == 0
        assert old.digest(OUT / f"{track}-line-changes.jsonl") == line["rows_sha256"]
        revised_results[track] = dict(frames=len(revised), labeled=len(truth), missed_frames=sum(
            item["miss"] for item in after["approaches"].values()), right_forecasts_preserved=True,
            caps_pass=True, rolling_line_changed_frames=line["changed_lines"])

    # Removing one real approach forecast must count as a missed and unreported frame.
    truth = [dict(run="check", frame=0, truth="approach", label="1", truth_endpoints_m=[10.])]
    report = score([], truth)
    assert report["approaches"]["1"]["miss"] == 1 and report["states"]["no_output"] == 1
    no_alert = [dict(run="check", frame=0, forecasts={METHOD: dict(state="abstain", distance_m=None)})]
    report = score(no_alert, truth)
    assert report["approaches"]["1"]["miss"] == 1 and report["states"]["abstain"] == 1
    # Check that a cut at a tied score accepts the tied positive and negative together.
    negatives = [dict(run="check", frame=1, truth="straight", label="0", truth_endpoints_m=[])]
    trial = select([({}, {0:(.8,10.), 1:(.8,10.)}, [*truth, *negatives]) for _ in range(4)])
    assert trial["add"] == .8 and trial["added_hits"] == 4 and trial["net_risk"] == 4
    for track in ("spa", "magione", "monza", "ks_silverstone1967", "ks_vallelunga"):
        try:
            validate_transfer(track, "")
        except ValueError:
            continue
        raise AssertionError(track)
    spline = validate_transfer("ks_highlands", "layout_short")
    old.write(OUT / "verification.json", dict(
        status="verified", tracks=revised_results, absent_and_abstaining_frames_counted=True,
        protected_right_forecasts=True, independent_replay_preflight_only=True,
        reserved_spline_sha256=old.digest(spline),
        source_sha256=digests((Path(__file__), receipt, comparison_path, line_path)),
        seed=None, termination="verification_complete"))
    print(revised_results, flush=True)


if __name__ == "__main__":
    main()
