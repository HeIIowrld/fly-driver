"""Evidence boundary for learned braking: only native response before cleanup counts."""
import math


def evaluate_braking(events, feedback, *, entry_speed_kmh=100., brake_threshold=.1,
                     minimum_drop_kmh=5., minimum_brake_requests=3):
    if not (math.isfinite(entry_speed_kmh) and 0 < entry_speed_kmh
            and 0 < brake_threshold <= 1 and 0 < minimum_drop_kmh < entry_speed_kmh
            and type(minimum_brake_requests) is int and minimum_brake_requests >= 3):
        raise ValueError("Explicit high-speed braking acceptance thresholds required")
    stops = [e for e in events if e["kind"]=="stop_request"]
    if len(stops)!=1:
        raise ValueError("Exactly one independent cleanup boundary required")
    end = stops[0]["when"]
    accepted=[]
    for e in events:
        if e["kind"]!="student_request" or not (e.get("receipt") or {}).get("driver_accepted"):
            continue
        conversion=e["conversion"]
        receipt=e["receipt"]
        stamp=receipt["driver_call_finished_perf_counter_seconds"]
        if (not math.isfinite(stamp) or stamp>=end or (accepted and stamp<=accepted[-1][0])
                or conversion["pedal_override"] or conversion["clipping_applied"]
                or conversion["requested_control"]!=receipt["requested_control"]
                or list(conversion["raw_prediction"][1:])!=[
                    receipt["requested_control"]["throttle"],receipt["requested_control"]["brake"]]):
            raise ValueError("Unmodified, ordered and pre-cleanup student requests required")
        accepted.append((stamp, receipt["requested_control"]["brake"]))
    output=dict(passed=False,accepted_student_requests=len(accepted),cleanup_excluded=True,
        high_speed_student_brake_requests=0,reason="no_student_authority_interval")
    if not accepted:
        return output
    samples=[];seen=set();initial_damage=None
    for f in feedback:
        t=f["first_receipt"]
        if not accepted[0][0]<=t<end or f["packet"] in seen:
            continue
        authority=f["authority"]
        damage=f["guard"]["carDamage"]
        if (authority.get("builtin_ai_controlled") is not False or f["guard"]["numberOfTyresOut"]!=0
                or not math.isfinite(f["fields"]["speedKmh"])
                or len(damage)!=5 or not all(math.isfinite(x) for x in damage)
                or (initial_damage is not None and any(x>base+.01 for x,base in zip(damage,initial_damage)))):
            output["reason"]="native_authority_or_track_limit_failed"
            return output
        if initial_damage is None:initial_damage=damage
        if samples and t<=samples[-1]["first_receipt"]:
            raise ValueError("Original native receipt sequence must increase")
        seen.add(f["packet"]); samples.append(f)
    for j,(t,brake) in enumerate(accepted):
        if brake<brake_threshold:
            continue
        preceding=[f for f in samples if 0<=t-f["first_receipt"]<=.05]
        if not preceding or preceding[-1]["fields"]["speedKmh"]<entry_speed_kmh:
            continue
        cluster=[]
        for when,value in accepted[j:]:
            if value<brake_threshold or (cluster and when-cluster[-1]>.05):
                break
            cluster.append(when)
        output["high_speed_student_brake_requests"]=max(len(cluster),output["high_speed_student_brake_requests"])
        if len(cluster)<minimum_brake_requests:
            continue
        following=[f for f in samples if t<=f["first_receipt"]<=min(cluster[-1]+.05,end)
                   and f["fields"]["brake"]>=brake_threshold-.02]
        if len(following)<3:
            continue
        start=preceding[-1]["fields"]["speedKmh"]
        drop=start-min(f["fields"]["speedKmh"] for f in following)
        if drop>=minimum_drop_kmh:
            output.update(passed=True,reason="verified_learned_high_speed_braking",braking_entry_kmh=start,
                          speed_drop_kmh=drop,student_brake_requests=len(cluster),
                          native_braking_packets=len(following),first_brake_sdk_finish=t)
            return output
    output["reason"]="no_qualifying_pre_cleanup_high_speed_braking"
    return output
