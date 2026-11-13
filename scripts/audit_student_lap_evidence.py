"""Audit recorded student lap timing and validity without accessing the simulator."""
import argparse
import json
from pathlib import Path

from flybrain_drive.ac_native_clock import decode_graphics_timing
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX
from flybrain_drive.teacher_curriculum import LapEvidence
from scripts.audit_teacher_recovery_laps import decode_payload
from train_vision_teacher_line import digest, read, write


def sdk_sample(receipt):
    packet = json.loads(decode_payload(receipt,"wire_base64","wire_sha256"))
    namespace = packet["privileged_teacher"]
    if (packet["steps"] != receipt["sequence"] or not packet["producer_session_id"]
            or namespace["actor_input"] is not False
            or namespace["protocol"] != "flybrain.ac.privileged_teacher.v1"):
        raise ValueError("SDK sequence, producer or diagnostic scope differs")
    truth = namespace["raw"]
    sample = (packet["steps"],truth.get("LapInvalidated"),packet["LapCount"],packet["timestamp_ac"])
    return packet["producer_session_id"], sample, truth["NormalizedSplinePosition"]


def load_sdk_source(path, proof):
    if proof.get("receiver_terminated") is not True or digest(path) != proof["sha256"]:
        raise ValueError("SDK source hash or receiver termination differs")
    producer = None
    samples = {}
    with path.open(encoding="utf8") as stream:
        for row in map(json.loads,stream):
            session, sample, _ = sdk_sample(row)
            sequence = sample[0]
            if samples and (session != producer or sequence <= next(reversed(samples))):
                raise ValueError("SDK source producer or ordering differs")
            producer = session
            samples[sequence] = sample
    if not samples or len(samples) != proof["packets"]:
        raise ValueError("SDK source packet count differs")
    return producer, samples


def audit(recording):
    receipt, events = recording/"result.json", recording/"events.jsonl"
    source = read(receipt)["lap_evidence"]
    if source["actor_input"] is not False or digest(events) != source["events_sha256"]:
        raise ValueError("Student diagnostic event hash or scope differs")
    source_stream = source.get("sdk_source")
    sdk_path = recording/"tcp-source-packets.jsonl"
    inputs = [Path(__file__),receipt,events]
    if source_stream is not None:
        source_producer, source_samples = load_sdk_source(sdk_path,source_stream)
        inputs.append(sdk_path)
    tracker = LapEvidence()
    rows = graphics_count = 0
    laps = []
    lap_packets = []
    start_time = None
    previous_sequence = None
    previous_graphics = None
    producer = None
    tyres_index = next(sum(n for _,_,n in PHYSICS_LAYOUT[:i])
                       for i,(name,_,_) in enumerate(PHYSICS_LAYOUT) if name=="numberOfTyresOut")
    with events.open(encoding="utf8") as stream:
        for event in map(json.loads,stream):
            if event["kind"] != "raw_pair":
                continue
            rows += 1
            session, sample, progress = sdk_sample(event["sdk"])
            sequence, _, _, now = sample
            native = event["native"]
            tyres = PHYSICS_PREFIX.unpack(decode_payload(native,"prefix_base64","prefix_sha256"))[tyres_index]
            if (previous_sequence is not None and sequence < previous_sequence
                    or producer is not None and producer != session):
                raise ValueError("SDK sequence or producer changed")
            previous_sequence, producer = sequence, session
            if "graphics" not in event:
                tracker = LapEvidence()
                lap_packets = []
                start_time = None
                continue
            graphics_count += 1
            g = event["graphics"]
            timing = decode_graphics_timing(decode_payload(g,"prefix_base64","prefix_sha256"),
                                            g["received_monotonic_seconds"])
            if timing.metadata() != g or (previous_graphics is not None and timing.packet_id < previous_graphics):
                raise ValueError("Graphics timing metadata or packet order differs")
            previous_graphics = timing.packet_id
            if timing.status != "live":
                tracker = LapEvidence()
                lap_packets = []
                start_time = None
                continue
            previous_crossings = tracker.crossings
            previous_laps = len(tracker.laps)
            tracker.update(progress,timing,tyres)
            lap_packets.append(sample)
            if len(tracker.laps) > previous_laps:
                lap = tracker.laps[-1].copy()
                unique = {row[0]:row[1:] for row in lap_packets}
                stable_duplicates = all(row[1:] == unique[row[0]] for row in lap_packets)
                source_agrees = None
                if source_stream is not None:
                    source_agrees = (source_producer == producer and source_stream.get("error") is None
                        and all(source_samples.get(row[0]) == row for row in lap_packets))
                    unique = {seq:sample[1:] for seq,sample in source_samples.items()
                              if lap_packets[0][0] <= seq <= sequence}
                values = list(unique.values())
                contiguous = len(unique) == sequence-lap_packets[0][0]+1
                sdk_lap_agrees = (
                    all(v[1] in (lap["lap"]-1,lap["lap"]) for v in values)
                    and sum(v[1] == lap["lap"]-1 for v in values) >= .95*len(values))
                source_times = [v[2] for v in values]
                elapsed = now-start_time
                clock_matches = (all(a<b for a,b in zip(source_times,source_times[1:]))
                                 and abs(elapsed-lap["lap_time_ms"]/1000) <= .15)
                official = (contiguous and stable_duplicates and source_agrees is not False
                            and sdk_lap_agrees and clock_matches
                            and all(type(v[0]) is int and v[0]==0 for v in values))
                lap.update(sdk_contiguous=contiguous, sdk_duplicates_agree=stable_duplicates,
                    sdk_source_stream_agrees=source_agrees,
                    sdk_counter_agreement=sdk_lap_agrees, sdk_source_time_matches=clock_matches,
                    sdk_source_packets=len(unique), official_validity_verified=official,
                    verified_valid_lap=official and lap["clean_all_tyres"])
                laps.append(lap)
            if tracker.crossings != previous_crossings:
                lap_packets = lap_packets[-1:]
                start_time = now
    if (rows,graphics_count,rows-graphics_count) != (
            source["raw_pairs"],source["graphics_records"],source["missing_graphics"]):
        raise ValueError("Student lap evidence counts differ from receipt")
    return dict(seed=None, laps=laps, verified_valid_laps=sum(lap["verified_valid_lap"] for lap in laps),
        raw_pairs=rows, graphics_records=graphics_count, missing_graphics=rows-graphics_count,
        inputs_sha256={str(p):digest(p) for p in inputs},
        actor_input=False, actuator_messages=0, termination="offline student lap audit completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.recording)
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/"result.json",result)


if __name__=="__main__":
    main()
