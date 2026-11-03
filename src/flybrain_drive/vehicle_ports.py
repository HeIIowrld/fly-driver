"""Declared vehicle-to-organ port hypothesis, using feature names rather than offsets."""


def extend_organ_ports(graph, annotations, old_constructor, old_names, new_names, candidate_ports):
    """Preserve old input order and add side-specific temperature/load projections.

    Front/rear wheels map to fore/hind leg ports; middle leg ports receive both
    same-side wheel feature sets. This is an engineering mapping, not a measured
    six-legged animal with four wheels. No synthetic middle-leg measurement is made.
    """
    old_names, new_names = tuple(old_names), tuple(new_names)
    if (len(old_names) != len(set(old_names)) or len(new_names) != len(set(new_names))
            or len(old_names) != old_constructor["organ_size"]
            or not set(old_names) <= set(new_names)):
        raise ValueError("Unique complete old/new organ feature names required")
    old_to_new = [new_names.index(name) for name in old_names]
    groups = {name: {"body_ids": list(map(str, group["body_ids"])),
                     "feature_indices": [old_to_new[i] for i in group["feature_indices"]]}
              for name, group in old_constructor["organ_groups"].items()}
    metadata = {}
    for row in annotations:
        body = str(int(row["bodyId"]))
        if body in metadata:
            raise ValueError("Duplicate annotated body ID")
        metadata[body] = row
    for family, sensor in (("thermal_receptor_candidates", "tyre_core_temperature"),
                           ("leg_load_receptor_candidates", "wheel_normal_load")):
        for body in candidate_ports[family]["body_ids"]:
            body = str(body)
            row = metadata[body]
            side = row.get("rootSide")
            if side not in ("L", "R"):
                raise ValueError("Declared left/right receptor side required")
            front, rear = (0, 2) if side == "L" else (1, 3)
            if family == "thermal_receptor_candidates":
                if (row.get("class") != "thermosensory"
                        or row.get("type") not in ("TRN_VP2", "TRN_VP3a", "TRN_VP3b")):
                    raise ValueError("Exact reviewed thermal receptor candidate required")
                name = "thermal_" + row["type"] + "_" + side
                corners = (front, rear)
            else:
                nerve = row.get("entryNerve")
                if (row.get("subclass") != "campaniform sensilla"
                        or nerve not in ("ProLN", "MesoLN", "MetaLN")):
                    raise ValueError("Exact reviewed leg load candidate required")
                name = "load_" + nerve + "_" + side
                corners = {"ProLN": (front,), "MesoLN": (front, rear), "MetaLN": (rear,)}[nerve]
            prefixes = tuple(f"{sensor}.{c}." for c in corners)
            features = [i for i, label in enumerate(new_names) if label.startswith(prefixes)]
            if not features:
                raise ValueError("Scalar feature names missing from organ calibration")
            group = groups.setdefault(name, {"body_ids": [], "feature_indices": features})
            if group["feature_indices"] != features:
                raise ValueError("Conflicting port group calibration")
            group["body_ids"].append(body)
    assigned = list(map(str, old_constructor["visual_ids"])) + [
        body for group in groups.values() for body in group["body_ids"]]
    inputs = {str(body) for body, role in zip(graph["ids"], graph["roles"], strict=True)
              if role == "input"}
    if len(assigned) != len(set(assigned)) or set(assigned) != inputs:
        raise ValueError("Port groups must cover exactly all disjoint graph inputs")
    return groups, old_to_new
