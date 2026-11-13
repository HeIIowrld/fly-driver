"""Leave one development circuit out when calibrating boundary onset distance."""
from pathlib import Path
from collections import Counter
from statistics import mean

from audit import INPUTS, MUGELLO, RECOVERY, RUNS
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows
from evaluate_independent_corner import compare
from score_corner_captures import adjacent, summarize
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
BASES = {
    "magione": RUNS/"vision-corner-independent-runner-rehearsal-20261111-01/replay",
    "ks_laguna_seca": RUNS/"vision-laguna-independent-evaluation-20261111-01",
    "ks_zandvoort": RUNS/"vision-zandvoort-independent-evaluation-20261111-01/fusion/base",
    "mugello": MUGELLO/"base",
}

def dist(row, fraction):
    strong = row["strong_m"]
    support = row["support_start_m"]
    return strong if support is None else strong + fraction*(support-strong)

def correct_geometry(rows):
    return [r for r in rows if r["truth"]=="approach" and str(r["side"])==r["label"]]

def calibrate(train):
    def error(weight):
        return mean(mean(abs(dist(r,weight)-r["truth_distance_m"]) for r in group) for group in train)
    weights = [i/20 for i in range(21)]
    return min(weights,key=error)

def extra_forecasts(geometry, baseline, weight, settings):
    previous = None
    for observation, row in zip(geometry, baseline, strict=True):
        assert frame_key(row) == frame_key(observation)
        if not observation["side"]:
            previous = None
            yield None
            continue
        side = observation["side"]
        move = observation["travel_m"]
        continuous = (previous is not None and adjacent(previous,observation) and previous["side"]==side
                      and move is not None and move >= settings["minimum_local_travel_m"]
                      and abs(observation["strong_m"]-(previous["strong_m"]-move))
                             <= settings["maximum_position_innovation_m"])
        previous = observation
        if continuous and row["state"] == "straight":
            distance = dist(observation,weight)
            if 0 <= distance <= 20:
                yield side,distance
                continue
        yield None

def score(original, prediction):
    side, distance = prediction
    state = "left" if side==1 else "right"
    hit = original["truth"]=="approach" and original["label"]==str(side)
    errors = [distance-v for v in original["truth_endpoints_m"]] if hit else []
    return dict(original, state=state, predicted_direction=side, predicted_distance_m=distance,
                direction_hit=hit, joint_hit=bool(errors) and max(map(abs,errors)) <= 5,
                signed_distance_error_m=mean(errors) if errors else None,
                absolute_distance_error_m=mean(map(abs,errors)) if errors else None)

def compare_tracks(parent, rows, rules):
    def group(values):
        return {run:[r for r in values if r["run"]==run] for run in dict.fromkeys(r["run"] for r in values)}
    candidate = group(rows)
    receipt = compare({"parent":group(parent), "candidate":candidate},rules)
    receipt["stable_leads_m"] = [r["first_stable_lead_m"] for group in candidate.values()
                                 for r in summarize(group)["events"] if r["first_stable_frame"] is not None]
    return receipt

def run():
    geometry_paths = {track:OUT/track/"rows.jsonl" for track in INPUTS}
    geometry = {}
    for track,path in geometry_paths.items():
        assert digest(path) == read(path.with_name("result.json"))["rows_sha256"]
        geometry[track] = read_rows(path)
    checked = {track:correct_geometry(rows) for track,rows in geometry.items()}
    details = {}
    changed = []
    inputs = set(geometry_paths.values())
    for track, data in geometry.items():
        configuration_path = RECOVERY/INPUTS[track][0]
        configuration = load_frozen(configuration_path)
        rules_path = Path(configuration["base_protocol"])
        rules = read(rules_path)["comparison"]
        settings_path = Path(configuration["boundary_fusion_pins"])
        settings = load_frozen(settings_path)
        base = BASES[track]
        base_path = base/"result.json"
        base_proof = read(base_path)
        evaluation = MUGELLO if track=="mugello" else RECOVERY/track
        result_path = evaluation/"result.json"
        result = read(result_path)
        assert result["base_receipt_sha256"] == digest(base_path)
        parent = load_score(base/"parent/score", base_proof["score_receipt_sha256"]["parent"])
        current = load_score(evaluation/"score", result["score_receipt_sha256"])
        inputs.update((configuration_path,rules_path,settings_path,base_path,result_path))
        train = [group for name,group in checked.items() if name!=track]
        weight = calibrate(train)
        comparisons = {}
        for name,fraction in (("strong",0.),("half",.5),("leave_circuit_out",weight)):
            new = []
            for original,prediction in zip(current,extra_forecasts(data,current,fraction,settings),strict=True):
                row = score(original,prediction) if prediction else original
                if prediction:
                    changed.append(dict(row, track=track, method=name))
                new.append(row)
            before = compare_tracks(parent,current,rules)
            assert before["metrics"] == result["comparison"]["metrics"]
            after = compare_tracks(parent,new,rules)
            comparisons[name] = dict(fraction=fraction,
                geometry_distance_mae_m=mean(abs(dist(r,fraction)-r["truth_distance_m"]) for r in checked[track]),
                geometry_within_five_m=sum(max(abs(dist(r,fraction)-v) for v in r["truth_endpoints_m"]) <= 5
                                           for r in checked[track]), comparison=after)
        details[track] = dict(geometry_distance_samples=len(checked[track]), before=before, alternatives=comparisons)
        print(track, weight, before["metrics"]["candidate"],
              {name:(r["geometry_within_five_m"],r["comparison"]["metrics"]["candidate"])
               for name,r in comparisons.items()})
    jsonl(OUT/"counterfactual-changes.jsonl", changed)
    write(OUT/"transfer-result.json",dict(by_track=details, seed=None, source_sha256={str(p):digest(p)
          for p in sorted(inputs | {Path(__file__)})}, changes_sha256=digest(OUT/"counterfactual-changes.jsonl"),
          termination="development_replays_complete"))
if __name__=="__main__":
    run()
