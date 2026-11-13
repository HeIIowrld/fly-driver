"""Read installed KN5 surface meshes for offline labels.

KN5 layout reference: AcTools/Kn5File/Kn5Basic.cs (gro-ove/actools).
Texture payloads, material properties, and unneeded geometry are skipped.
"""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import configparser
import json
import re
import struct
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
AC = Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
SOURCE_FORMAT = "https://github.com/gro-ove/actools/blob/master/AcTools/Kn5File/Kn5Basic.cs"
PHYSICS_NAME = re.compile(r"^0*[1-9][0-9]*([A-Z].*)$", re.IGNORECASE)


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


class Reader:
    def __init__(self, file, length):
        self.file, self.length = file, length

    def read(self, size):
        data = self.file.read(size)
        if len(data) != size:
            raise ValueError("Incomplete KN5 record")
        return data

    def skip(self, size):
        target = self.file.tell() + size
        if size < 0 or target > self.length:
            raise ValueError("Invalid KN5 record length")
        self.file.seek(target)

    def uint(self):
        return struct.unpack("<I", self.read(4))[0]

    def text(self):
        size = self.uint()
        if size > 1_000_000:
            raise ValueError("Invalid KN5 name length")
        return self.read(size).decode("utf-8", errors="strict")


def read_header(reader):
    if reader.read(6) != b"sc6969":
        raise ValueError("Unknown KN5 header")
    version = reader.uint()
    if version not in (5, 6):
        raise ValueError(f"Unsupported KN5 version: {version}")
    if version == 6:
        reader.skip(4)
    for _ in range(reader.uint()):
        reader.skip(4)
        reader.text()
        reader.skip(reader.uint())
    materials = []
    for _ in range(reader.uint()):
        materials.append(reader.text())
        reader.text()
        reader.skip(6)
        for _ in range(reader.uint()):
            reader.text()
            reader.skip(40)
        for _ in range(reader.uint()):
            reader.text()
            reader.skip(4)
            reader.text()
    return version, materials


def nodes(reader, parent, active, materials, inventory, collect):
    kind = reader.uint()
    name = reader.text()
    children = reader.uint()
    enabled = bool(reader.read(1)[0]) and active
    if kind == 1:
        local = np.frombuffer(reader.read(64), dtype="<f4").astype(float).reshape(4, 4)
        if not np.allclose(local[:, 3], [0., 0., 0., 1.], atol=1e-5):
            raise ValueError(f"Unexpected KN5 matrix layout: {name}")
        parent = local @ parent
    elif kind in (2, 3):
        flags = tuple(reader.read(3))
        if kind == 3:
            for _ in range(reader.uint()):
                reader.text()
                reader.skip(64)
        vertices = reader.uint()
        stride = 44 if kind == 2 else 76
        take = bool(collect and kind == 2 and enabled and PHYSICS_NAME.match(name))
        if take:
            raw = np.frombuffer(reader.read(vertices * stride), dtype="<f4").reshape(vertices, stride // 4)
            world = raw[:, :3] @ parent[:3, :3] + parent[3, :3]
        else:
            reader.skip(vertices * stride)
        indices = reader.uint()
        if take:
            faces = np.frombuffer(reader.read(indices * 2), dtype="<u2").reshape(-1, 3)
            if len(faces) and faces.max() >= vertices:
                raise ValueError(f"Invalid KN5 face index: {name}")
        else:
            reader.skip(indices * 2)
        material = reader.uint()
        reader.skip(29 if kind == 2 else 12)
        if PHYSICS_NAME.match(name):
            metadata = dict(name=name, material=materials[material] if material < len(materials) else None,
                            triangles=indices//3, active=enabled, visible=bool(flags[1]), kind=kind)
            inventory.append(metadata)
            if take and len(faces):
                yield metadata, world[faces]
    else:
        raise ValueError(f"Unsupported KN5 node type {kind}: {name}")
    for _ in range(children):
        yield from nodes(reader, parent, enabled, materials, inventory, collect)


def load_model(path, collect=True):
    inventory = []
    length = path.stat().st_size
    with path.open("rb") as file:
        reader = Reader(file, length)
        version, materials = read_header(reader)
        meshes = list(nodes(reader, np.eye(4), True, materials, inventory, collect))
        if file.tell() != length:
            raise ValueError(f"Trailing bytes in KN5: {path.name} ({length - file.tell()})")
    return dict(version=version, bytes=length, sha256=digest(path), nodes=inventory), meshes


def config(path):
    result = configparser.ConfigParser(interpolation=None, strict=True)
    with path.open(encoding="utf-8-sig") as source:
        result.read_file(source)
    return result


def track_models(track):
    path = track / "models.ini"
    if not path.is_file():
        return [track / f"{track.name}.kn5"], None
    models = config(path)
    files = []
    for section in models.sections():
        if not section.startswith("MODEL_"):
            continue
        data = models[section]
        if any(float(a) for key in ("POSITION", "ROTATION") for a in data[key].split(",")):
            raise ValueError("External KN5 model transform not supported in this audit")
        files.append(track / data["FILE"])
    if not files:
        raise ValueError("No loaded models")
    return files, path


def surface_config(track):
    paths = [AC / "system" / "data" / "surfaces.ini", track / "data" / "surfaces.ini"]
    known = {}
    for path in paths:
        for row in config(path).values():
            if "KEY" not in row:
                continue
            key = row["KEY"].strip().upper()
            known[key] = dict(valid=row.getint("IS_VALID_TRACK") == 1,
                              pitlane=row.getint("IS_PITLANE", fallback=0) == 1,
                              friction=row.getfloat("FRICTION"), source=str(path))
    return known, paths


def surface_key(name, known):
    match = PHYSICS_NAME.match(name)
    if not match:
        return None
    base = match[1].upper()
    keys = [key for key in known if base.startswith(key)]
    return max(keys, key=len) if keys else None


def main():
    track = AC / "content" / "tracks" / (sys.argv[1] if len(sys.argv) > 1 else "ks_silverstone1967")
    files, model_config = track_models(track)
    known, sources = surface_config(track)
    result = {}
    counts = Counter()
    for path in files:
        result[path.name], _ = load_model(path, collect=False)
        for node in result[path.name]["nodes"]:
            node["surface"] = surface_key(node["name"], known)
            counts[node["surface"] or "unmatched"] += 1
        print(path.name, len(result[path.name]["nodes"]), flush=True)
    if model_config is not None:
        sources.append(model_config)
    receipt = dict(track=track.name, parser_sha256=digest(Path(__file__)), format_reference=SOURCE_FORMAT,
                   models=result, surfaces=known, config_sha256={str(p): digest(p) for p in sources},
                   matched_names=dict(counts), actor_input=False, source_asset_copies=0)
    (OUT / f"{track.name}-mesh-inventory.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(counts)


if __name__ == "__main__":
    main()
