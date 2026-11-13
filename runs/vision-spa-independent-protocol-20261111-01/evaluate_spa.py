"""Run the two predeclared Spa comparisons after archive admission."""
import argparse
from pathlib import Path

from audit_corner_capture_admission import review
from evaluate_fused_corner import load_score
from evaluate_recovered_corner import run as evaluate
from train_vision_teacher_line import digest, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture_dir", "track_dir", "cache", "output"):
        parser.add_argument("--"+name.replace("_","-"), type=Path, required=True)
    args = parser.parse_args()
    settings = Path(__file__).parent
    frozen = read(settings/"admission-review.json")
    candidate = settings/"candidate.json"
    if digest(Path(__file__)) != frozen["evaluation_program_sha256"]:
        raise ValueError("Spa evaluation runner differs from the frozen source")
    if args.track_dir.name != "spa" or digest(candidate) != frozen["immutable_candidate_sha256"]:
        raise ValueError("Expected the frozen Spa evaluation")
    if digest(Path("scripts/audit_corner_capture_admission.py")) != frozen["audit_program_sha256"]:
        raise ValueError("Archive reviewer differs from the frozen source")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    check = review(args.capture_dir, settings/"capture.json", settings/"admission-review.json",
                   output/"admission")
    if not check["ready_for_scoring"]:
        write(output/"result.json",dict(status="not_admitted",admission=check,termination="metadata_admission"))
        return

    full = output/"comparison"
    evaluate(argparse.Namespace(protocol=candidate, archives=args.capture_dir/"archive-map.json",
               track_dir=args.track_dir, cache=args.cache, output=full,base_replay=None))
    proof = read(full/"result.json")
    base = read(full/"base/result.json")
    if (proof["status"] != "evaluated" or proof["development_rehearsal"]
            or base["admission_rows_sha256"] != check["rows_sha256"]):
        raise ValueError("Scored timeline differs from the admitted independent video")
    mixed_receipt = base["score_receipt_sha256"]["candidate"]
    load_score(full/"base/candidate/score",mixed_receipt)
    write(output/"result.json",dict(track="spa",status="evaluated",
          primary=proof["comparison"], secondary_mixed=base["comparison"],
          admission_sha256=digest(output/"admission/result.json"),
          primary_receipt_sha256=digest(full/"result.json"),
          secondary_receipt_sha256=digest(full/"base/result.json"),
          protocol_sha256=digest(candidate), termination="independent_spa_comparisons_complete"))


if __name__ == "__main__":
    main()
