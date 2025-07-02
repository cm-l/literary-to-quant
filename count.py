#!/usr/bin/env python3
"""
Spatial-reasoning summary v3

Outputs (per model, scenario_id):
- dominant_object_count   (mode of object counts)
- bbox_volume             (tightest axis-aligned cuboid)
- max_pairwise_distance   (largest inter-object distance)

Directory layout assumed:
outputs/<model>/<category>/<file>.json
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = "outputs"
Coord = Tuple[float, float, float]           # (x, y, z)
StatsKey = Tuple[str, str]                  # (model, scenario)

# ────────────────────────── helpers ──────────────────────────
def load(fp: Path) -> Dict:
    with fp.open(encoding="utf-8") as f:
        return json.load(f)

def extract_positions(raw_json: Dict) -> Dict[str, Coord]:
    objs = raw_json.get("objects", {})
    pos_dict: Dict[str, Coord] = {}
    for name, info in objs.items():
        p = info.get("position", {})
        pos_dict[name] = (float(p.get("x", 0.0)),
                          float(p.get("y", 0.0)),
                          float(p.get("z", 0.0)))      # default z=0 for 2-D
    return pos_dict

# ─────────────────── aggregation object ─────────────────────
class Agg:
    """Aggregate stats for one (model, scenario)."""
    __slots__ = ("count_ctr", "min_xyz", "max_xyz", "points")

    def __init__(self) -> None:
        self.count_ctr: Counter[int] = Counter()
        self.min_xyz: List[float] = [math.inf, math.inf, math.inf]
        self.max_xyz: List[float] = [-math.inf, -math.inf, -math.inf]
        self.points: List[Coord] = []      # collect all points (for max-dist)

    def update(self, positions: Dict[str, Coord]) -> None:
        n = len(positions)
        self.count_ctr[n] += 1

        for x, y, z in positions.values():
            self.points.append((x, y, z))
            self.min_xyz[0] = min(self.min_xyz[0], x)
            self.min_xyz[1] = min(self.min_xyz[1], y)
            self.min_xyz[2] = min(self.min_xyz[2], z)
            self.max_xyz[0] = max(self.max_xyz[0], x)
            self.max_xyz[1] = max(self.max_xyz[1], y)
            self.max_xyz[2] = max(self.max_xyz[2], z)

    # derived metrics ------------------------------------------------------
    def dominant_count(self) -> int:
        return self.count_ctr.most_common(1)[0][0]

    def bbox_spans(self) -> Tuple[float, float, float]:
        dx = self.max_xyz[0] - self.min_xyz[0]
        dy = self.max_xyz[1] - self.min_xyz[1]
        dz = self.max_xyz[2] - self.min_xyz[2]
        return dx, dy, dz

    def bbox_volume(self) -> float:
        dx, dy, dz = self.bbox_spans()
        return dx * dy * dz

    def max_pairwise(self) -> float:
        if len(self.points) < 2:
            return 0.0
        max_d2 = 0.0
        for (x1, y1, z1), (x2, y2, z2) in combinations(self.points, 2):
            d2 = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
            if d2 > max_d2:
                max_d2 = d2
        return math.sqrt(max_d2)

# ────────────────────────── main logic ───────────────────────
def summarise(root: str) -> Dict[str, Dict[str, Dict]]:
    aggs: Dict[StatsKey, Agg] = defaultdict(Agg)

    for fp in Path(root).rglob("*.json"):
        try:
            outer = load(fp)
            model = outer["model"]
            scenario = outer["scenario_id"]
            raw = json.loads(outer["raw_output"])
            positions = extract_positions(raw)

            aggs[(model, scenario)].update(positions)
        except Exception as exc:
            print(f"⚠️  Skipping {fp}: {exc}")

    results: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for (model, scenario), agg in aggs.items():
        results[model][scenario] = {
            "dominant_object_count": agg.dominant_count(),
            "bbox_volume": agg.bbox_volume(),
            "max_pairwise_distance": agg.max_pairwise(),
        }
    return results

def main() -> None:
    ap = argparse.ArgumentParser(description="Spatial-reasoning summary v3")
    ap.add_argument("root", nargs="?", default=ROOT,
                    help=f"root folder (default ./{ROOT})")
    args = ap.parse_args()

    summary = summarise(args.root)
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
