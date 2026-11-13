"""Save native-pixel annotations from the original Silverstone front images."""
from hashlib import file_digest
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "vision-silverstone-visible-edges-20261112-01"

# Mark the midpoint of the visible transition using raw images and native row
# intensities. Two sides of the same dark strip are different image features.
COLUMNS = (215, 235, 255, 275, 295, 315)
GRASS = {
    488: (96.5, 98.5, 101., 103.5, 105.5, 107.5),
    489: (97.5, 99.5, 101.5, 104.5, 106.5, 108.5),
    490: (97.5, 99.5, 102.5, 104.5, 106.5, 109.5),
}
STRIP = {
    620: ((93.5, 95.5, 97.5, 99.5, 101.5, 103.5), (91.5, 92.5, 94.5, 95.5, 96.5, 98.5)),
    621: ((93.5, 96.5, 98.5, 100.5, 102.5, 104.5), (91.5, 93.5, 94.5, 96.5, 97.5, 98.5)),
    622: ((94.5, 97.5, 99.5, 101.5, 103.5, 105.5), (92.5, 93.5, 95.5, 96.5, 98.5, 99.5)),
    623: ((95.5, 98.5, 100.5, 102.5, 105.5, 107.5), (92.5, 94.5, 96.5, 97.5, 99.5, 101.5)),
}
# These short visible features locate the shoulder or a dark object. They
# do not label which surface marks the edge of the usable road.
AMBIGUOUS = {
    807: ("dark_object_road_facing_edge", [[235, 95.5], [255, 96.5], [275, 99.5]]),
    808: ("dark_object_road_facing_edge", [[235, 96.5], [255, 98.5], [275, 100.5]]),
    809: ("dark_object_road_facing_edge", [[235, 96.5], [255, 98.5], [275, 100.5]]),
    1048: ("bright_paved_seam", [[215, 94.5], [235, 96.5]]),
    1049: ("bright_paved_seam", [[215, 93.5], [235, 96.5], [255, 98.5]]),
    1050: ("bright_paved_seam", [[215, 93.5], [235, 96.5], [255, 98.5], [275, 100.5], [295, 103.5]]),
}


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def make_labels():
    inventory = SOURCE / "inventory.json"
    original = {row["frame"]: row for row in json.loads(inventory.read_text(encoding="utf8"))["records"]}
    result = {}
    for frame in sorted(GRASS.keys() | STRIP.keys() | AMBIGUOUS.keys()):
        path = SOURCE / f"front-{frame}.png"
        assert digest(path) == original[frame]["png_sha256"]
        row = dict(source_path=str(path), sha256=digest(path), group=original[frame]["corner"])
        if frame in GRASS:
            row.update(kind="grass_asphalt", target=list(zip(COLUMNS, GRASS[frame], strict=True)))
        elif frame in STRIP:
            lower, upper = STRIP[frame]
            row.update(kind="dark_strip", target=list(zip(COLUMNS, lower, strict=True)),
                       competitor=list(zip(COLUMNS, upper, strict=True)))
        else:
            kind, points = AMBIGUOUS[frame]
            row.update(kind="ambiguous", target=None, visible_feature=dict(kind=kind, points=points))
        result[frame] = row
    data = dict(
        status="single_review_image_diagnostic", source_inventory_sha256=digest(inventory),
        annotation_source_sha256=digest(Path(__file__)), image_size=[320, 180],
        nominal_row_uncertainty_px=1., coordinates="native pixel centers, approximate transition midpoints",
        semantic_scope="Visible transitions only. The dark strip's road-facing edge is an image cue. "
                       "Paved shoulder extent and official track limits are unverified.",
        frames=result,
    )
    (OUT / "labels.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    make_labels()
