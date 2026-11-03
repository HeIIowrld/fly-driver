"""Join reported transmitters to graph IDs; signs remain an explicit model hypothesis."""

from collections import Counter

import numpy as np


def join_transmitters(graph_ids, records, sign_hypothesis):
    """Use the dataset's consensus label without inventing consensus confidence.

    Individual prediction confidence is retained as a separate diagnostic. Missing
    and unassigned labels need explicit signs (typically zero) in the hypothesis.
    This does not infer postsynaptic receptors or physiological synaptic strength.
    """
    if (not sign_hypothesis or "missing" not in sign_hypothesis
            or any(type(value) is not int or value not in (-1, 0, 1)
                   for value in sign_hypothesis.values())):
        raise ValueError("An explicit -1/0/+1 sign hypothesis including missing is required")
    ids = list(map(int, graph_ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Unique nonempty graph body IDs required")
    by_id = {}
    for record in records:
        body = int(record["body"])
        if body in by_id:
            raise ValueError("Duplicate transmitter record")
        by_id[body] = dict(record)
    joined, signs, counts, missing, disagreements, low = [], [], Counter(), [], [], []
    for body in ids:
        record = by_id.get(body)
        label = record.get("consensus_nt") if record is not None else "missing"
        if label not in sign_hypothesis:
            raise ValueError("Unspecified transmitter label: " + str(label))
        counts[label] += 1
        sign = sign_hypothesis[label]
        signs.append(sign)
        if record is None:
            missing.append(str(body))
        else:
            confidence = record.get("predicted_nt_confidence")
            if confidence is not None and (not np.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Individual prediction confidence must be finite in [0,1]")
            if confidence is not None and confidence < .8:
                low.append(str(body))
            if label != record.get("predicted_nt"):
                disagreements.append(str(body))
        joined.append({**(record or {"body": body}), "missing": record is None,
                       "model_sign": sign, "model_sign_label_source": "consensus_nt",
                       "model_sign_is_physiologically_verified": False})
    return np.array(signs, dtype=np.float32), joined, {
        "nodes": len(ids), "matching_records": len(ids) - len(missing),
        "consensus_labels": dict(counts), "missing_body_ids": missing,
        "consensus_differs_from_individual_body_ids": disagreements,
        "individual_confidence_below_0_8_body_ids": low,
        "sign_counts": {str(sign): int(np.sum(np.asarray(signs) == sign)) for sign in (-1, 0, 1)},
        "sign_hypothesis": dict(sign_hypothesis), "physiological_signs_verified": False,
        "individual_confidence_is_not_consensus_confidence": True,
    }
