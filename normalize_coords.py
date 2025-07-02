#!/usr/bin/env python3
"""
normalize_coordinates.py
========================

Traverse an outputs/ directory, load every *.json result, rescale all
(x, y, z) coordinates so that *that run's own* bounding box is mapped
to the canonical cube [-128, 128] on each axis, and write the updated
files into a mirror directory tree rooted at ./normalized.

Example
-------
$ python normalize_coordinates.py               # defaults
$ python normalize_coordinates.py /data/outs my_norm

Directory layout preserved:
outputs/<model>/<category>/<file>.json  -->  normalized/<model>/<category>/<file>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

CANON_MIN, CANON_MAX = -128.0, 128.0
CANON_RANGE = CANON_MAX - CANON_MIN


Coord = Tuple[float, float, float]


# ────────────────────────── helpers ──────────────────────────
def load_json(fp: Path) -> Dict:
    with fp.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_positions(obj_json: Dict) -> Dict[str, Coord]:
    """Return {object_name: (x,y,z)}; missing z → 0.0."""
    out: Dict[str, Coord] = {}
    for name, entry in obj_json.get("objects", {}).items():
        p = entry.get("position", {})
        out[name] = (
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
            float(p.get("z", 0.0)),
        )
    return out


def normalize_coords(coords: List[Coord]) -> List[Coord]:
    """Affine-map each axis to [-128, 128]; if span==0 keep axis at 0."""
    xs, ys, zs = zip(*coords)
    mins = [min(xs), min(ys), min(zs)]
    maxs = [max(xs), max(ys), max(zs)]
    spans = [mx - mn if mx != mn else 1.0 for mn, mx in zip(mins, maxs)]

    normed: List[Coord] = []
    for x, y, z in coords:
        nx = CANON_MIN + (x - mins[0]) / spans[0] * CANON_RANGE
        ny = CANON_MIN + (y - mins[1]) / spans[1] * CANON_RANGE
        nz = CANON_MIN + (z - mins[2]) / spans[2] * CANON_RANGE
        normed.append((nx, ny, nz))
    return normed


def replace_positions(obj_json: Dict, normed_positions: Dict[str, Coord]) -> None:
    """In-place update of the 'position' dicts."""
    for name, (x, y, z) in normed_positions.items():
        pos = obj_json["objects"][name]["position"]
        pos.update({"x": x, "y": y, "z": z})


# ─────────────────────────── main logic ───────────────────────
def process_file(in_fp: Path, out_fp: Path) -> None:
    outer = load_json(in_fp)

    # parse the nested raw_output JSON string
    raw = json.loads(outer["raw_output"])
    positions = extract_positions(raw)

    # normalise
    names = list(positions.keys())
    coords = list(positions.values())
    normed_coords = normalize_coords(coords)
    normed_map = dict(zip(names, normed_coords))

    replace_positions(raw, normed_map)

    # put the updated string back into the outer JSON
    outer["raw_output"] = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))

    # OPTIONAL: flag that this file is normalised
    outer["normalised"] = True

    save_json(outer, out_fp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalise spatial JSON coordinates to ±128 cube")
    ap.add_argument("in_root", nargs="?", default="outputs",
                    help="root folder to read (default: ./outputs)")
    ap.add_argument("out_root", nargs="?", default="normalized",
                    help="root folder to write (default: ./normalized)")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    json_files = list(in_root.rglob("*.json"))
    if not json_files:
        print("No JSON files found under", in_root)
        return

    for in_fp in json_files:
        rel = in_fp.relative_to(in_root)
        out_fp = out_root / rel
        try:
            process_file(in_fp, out_fp)
        except Exception as exc:
            print(f"⚠️  Skipped {in_fp}: {exc}")

    print(f"✓ Done. Normalised files written under {out_root}")


if __name__ == "__main__":
    main()
