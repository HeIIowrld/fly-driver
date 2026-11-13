"""Re-score cached bend proposals with accumulated body motion."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from audit import INPUTS, RECOVERY, RUNS, MUGELLO
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import BASES, compare_tracks, score

OUT = Path(__file__).parent

@dataclass(frozen=True)
class Bend:
    side: int
    observed: float
    anchor: float
    traveled: float

def continue_bends(prior, options, move, limit):
    """Preserve an estimate only while local and cumulative positions agree."""
    bends = []
    for side, x in options:
        possible = [b for b in prior if b.side==side
                    and abs(x-(b.observed-move))<=limit
                    and abs(x-(b.anchor-move))<=limit]
        if possible:
            match = min(possible,key=lambda b:(-b.traveled,abs(x-(b.anchor-move))))
            bends.append(Bend(side,x,match.anchor-move,match.traveled+move))
        else:
            bends.append(Bend(side,x,x,0.))
    return bends

def choose(bends, minimum):
    supported = [b for b in bends if b.traveled>=minimum and 0<=b.anchor<=20]
    if not supported:
        return None
    best = min((-b.traveled,abs(b.anchor-b.observed),b.anchor) for b in supported)
    sides = {b.side for b in supported if (-b.traveled,abs(b.anchor-b.observed),b.anchor)==best}
    return (sides.pop(),best[2]) if len(sides)==1 else None

def run():
    summary, all_rows = {}, []
    inputs = {Path(__file__)}
    for track,(protocol_name,_) in INPUTS.items():
        config_path = RECOVERY/protocol_name
        config = load_frozen(config_path)
        settings_path = Path(config["boundary_fusion_pins"])
        settings = load_frozen(settings_path)
        rules_path = Path(config["base_protocol"])
        rules = read(rules_path)["comparison"]
        eval_dir = MUGELLO if track=="mugello" else RECOVERY/track
        eval_path = eval_dir/"result.json"
        proof = read(eval_path)
        assert digest(config_path)==proof["protocol_sha256"]
        base = BASES[track]
        base_path = base/"result.json"
        assert digest(base_path)==proof["base_receipt_sha256"]
        original = load_score(base/"parent/score",read(base_path)["score_receipt_sha256"]["parent"])
        baseline = load_score(eval_dir/"score",proof["score_receipt_sha256"])
        proposal_path = OUT/track/"rows.jsonl"
        proposal_proof = read(proposal_path.with_name("result.json"))
        assert digest(proposal_path)==proposal_proof["rows_sha256"]
        stage = (RUNS/"vision-mugello-readout-stage-audit-20261111-01" if track=="mugello"
                 else RUNS/"vision-corner-relative-chroma-gates-20261111-01"/track)
        visual_path = stage/("rows.jsonl" if track=="mugello" else "all-gates.jsonl")
        assert digest(visual_path)==proposal_proof["source_sha256"][str(visual_path)]
        inputs.update((config_path,settings_path,rules_path,eval_path,base_path,proposal_path,visual_path))
        previous, bends = None, []
        rows = []
        for record, evidence, current in zip(read_rows(proposal_path),read_rows(visual_path),baseline,strict=True):
            assert frame_key(record)==frame_key(current)==frame_key(evidence)
            move = record["travel_m"]
            continuing = previous is not None and adjacent(previous,record) and move is not None and move>=.10
            bends = continue_bends(bends if continuing else [],record["pooled_all"] or (),move if continuing else 0.,
                                   settings["maximum_position_innovation_m"])
            choice = choose(bends,settings["minimum_local_travel_m"])
            visual = evidence["fusion_visual_pass"] if track=="mugello" else evidence["presence"] and evidence["ordinal_range"]
            rows.append((current,choice,visual,move))
            previous = record
        before = compare_tracks(original,baseline,rules)
        assert before["metrics"]==proof["comparison"]["metrics"]
        cases = {}
        for name,require_visual in (("accumulated",False),("accumulated_visual",True)):
            emitted = []
            counts = Counter()
            for current,choice,visual,move in rows:
                allow = choice is not None and (visual or not require_visual) and current["state"]=="straight"
                row = score(current,choice) if allow else current
                if allow:
                    counts["added"]+=1
                    counts["added_after_step_below_0.75_m"]+=move<.75
                    counts["correct_direction"]+=row["direction_hit"]
                    counts["joint"]+=row["joint_hit"]
                    all_rows.append(dict(row,track=track,method=name))
                emitted.append(row)
            after = compare_tracks(original,emitted,rules)
            cases[name] = dict(comparison=after,added=counts)
        summary[track] = dict(revision=proof["revision"],before=before,cases=cases)
        print(track,{n:(c["comparison"]["metrics"]["candidate"],dict(c["added"])) for n,c in cases.items()})
    changes=OUT/"accumulated-changes.jsonl"
    jsonl(changes,all_rows)
    write(OUT/"accumulated-result.json",dict(tracks=summary,seed=None,changes_sha256=digest(changes),
          source_sha256={str(p):digest(p) for p in sorted(inputs)},termination="development_comparison_completed"))

if __name__=="__main__":
    run()
