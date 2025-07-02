#!/usr/bin/env python3
"""
visualize_spatial.py
====================

• One 3-D scatter per scenario
• Marker shape ⇒ model
• Marker colour ⇒ canonical object name (synonyms collapsed)
• Semi-transparent bounding box per object category
• Centroid of each box shown as an enlarged point

PNG files land in ./plots (default).  Use --show for interactive view.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ───────────────────────── configuration ─────────────────────────
ROOT_DEFAULT = "normalized"
PLOTS_DEFAULT = "normalizedplots"

Coord = Tuple[float, float, float]
Row   = Tuple[str, str, Coord]              # (model, canon_object, coord)
SceneDict = Dict[str, List[Row]]            # scenario → [Row, …]

# ─────────────────────── canonicalisation ───────────────────────
_SYN_MAP = {
    # base set + user extensions
    "fridge": "refrigerator",
    "refrigerator": "refrigerator",
    "half empty mug": "mug",
    "half full mug": "mug",
    "mug": "mug",
    "stack of books": "books",
    "pile of books": "books",
    "books": "books",
    "wooden ladder": "ladder",
    "ladder": "ladder",
    "trash can": "trashcan",
    "trashcan": "trashcan",
    "dining table": "table",
    "table": "table",
    "light fixure": "light",
    "light fixture": "light",
    "light": "light",
    "lego bricks": "lego",
    "lego pile": "lego",
    "lego": "lego",
    "pile of dishes": "dishes",
    "dishes": "dishes",
    "business cards": "business cards",
    "stack of business cards": "business cards",
    "electronic typewriter": "typewriter",
    "typewriter": "typewriter",
    "desk": "desk",
    "tiny desk": "desk",
}


def canonicalise(name: str) -> str:
    n = name.lower().replace("_", " ").replace("-", " ")
    n = re.sub(r"\d+$", "", n).strip()
    n = re.sub(r"\s+", " ", n)
    return _SYN_MAP.get(n, n)

# ───────────────────────── JSON helpers ─────────────────────────
def load_json(fp: Path) -> Dict:
    with fp.open(encoding="utf-8") as f:
        return json.load(f)


def extract_positions(raw: Dict) -> Dict[str, Coord]:
    objs = raw.get("objects", {})
    out: Dict[str, Coord] = {}
    for name, data in objs.items():
        p = data.get("position", {})
        out[name] = (
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
            float(p.get("z", 0.0)),
        )
    return out

# ─────────────────────── data collection ───────────────────────
def gather_positions(root: Path) -> SceneDict:
    scenes: SceneDict = defaultdict(list)
    for fp in root.rglob("*.json"):
        try:
            outer = load_json(fp)
            model     = outer["model"]
            scenario  = outer["scenario_id"]
            raw       = json.loads(outer["raw_output"])
            for obj_name, coord in extract_positions(raw).items():
                canon = canonicalise(obj_name)
                scenes[scenario].append((model, canon, coord))
        except Exception as exc:
            print(f"⚠️  Skipping {fp}: {exc}")
    return scenes

# ───────────────────────── plotting utils ──────────────────────
def cuboid_vertices(xmin, xmax, ymin, ymax, zmin, zmax):
    return [
        (xmin, ymin, zmin), (xmax, ymin, zmin),
        (xmax, ymax, zmin), (xmin, ymax, zmin),
        (xmin, ymin, zmax), (xmax, ymin, zmax),
        (xmax, ymax, zmax), (xmin, ymax, zmax),
    ]


def cuboid_faces(v):
    return [
        [v[i] for i in [0, 1, 2, 3]],
        [v[i] for i in [4, 5, 6, 7]],
        [v[i] for i in [0, 1, 5, 4]],
        [v[i] for i in [2, 3, 7, 6]],
        [v[i] for i in [1, 2, 6, 5]],
        [v[i] for i in [0, 3, 7, 4]],
    ]

# ───────────────────────── plotting ────────────────────────────
def make_plot(scenario: str, rows: List[Row], outdir: Path) -> None:
    models  = sorted({m for m, _, _ in rows})
    objects = sorted({o for _, o, _ in rows})

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8"]
    model_to_marker = {m: marker_cycle[i % len(marker_cycle)]
                       for i, m in enumerate(models)}

    cmap = cm.get_cmap("tab20", len(objects))
    obj_to_color = {o: cmap(i) for i, o in enumerate(objects)}

    # bounding-box extents per object
    bounds = {o: [float("inf"), -float("inf"),
                  float("inf"), -float("inf"),
                  float("inf"), -float("inf")] for o in objects}

    for _, obj, (x, y, z) in rows:
        b = bounds[obj]
        b[0] = min(b[0], x); b[1] = max(b[1], x)
        b[2] = min(b[2], y); b[3] = max(b[3], y)
        b[4] = min(b[4], z); b[5] = max(b[5], z)

    # ── figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # scatter points
    for model, obj, (x, y, z) in rows:
        ax.scatter(x, y, z,
                   marker=model_to_marker[model],
                   color=obj_to_color[obj],
                   s=40, alpha=0.50)               # more transparent

    # bounding boxes + centroids
    for obj, (xmin, xmax, ymin, ymax, zmin, zmax) in bounds.items():
        if xmin == xmax and ymin == ymax and zmin == zmax:
            # single point – still drop centroid but no box
            cx, cy, cz = xmin, ymin, zmin
        else:
            verts = cuboid_vertices(xmin, xmax, ymin, ymax, zmin, zmax)
            faces = cuboid_faces(verts)
            box = Poly3DCollection(faces,
                                   facecolors=obj_to_color[obj],
                                   edgecolors=obj_to_color[obj],
                                   linewidths=0.5,
                                   alpha=0.08)       # more transparent
            ax.add_collection3d(box)
            cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2

        # centroid marker (always a large filled circle)
        ax.scatter(cx, cy, cz,
                   marker="o",
                   edgecolor="black",
                   linewidth=0.5,
                   color=obj_to_color[obj],
                   s=120, alpha=0.90)

    ax.set_title(f"{scenario} – shape=model, colour=object")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)

    # legends
    shape_handles = [mlines.Line2D([], [], linestyle="",
                                   marker=model_to_marker[m],
                                   markersize=8,
                                   markerfacecolor="grey",
                                   markeredgecolor="black",
                                   label=m)
                     for m in models]
    colour_handles = [mpatches.Patch(facecolor=obj_to_color[o], label=o)
                      for o in objects]

    first = ax.legend(handles=shape_handles, title="Model", loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=colour_handles, title="Object", loc="lower left",
              ncol=2, fontsize="small")

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{scenario}_3d.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"✓ saved {outfile}")
    plt.close(fig)

# ────────────────────────── main ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="3-D spatial visualisations with bounding boxes & centroids")
    ap.add_argument("root", nargs="?", default=ROOT_DEFAULT,
                    help="root of outputs/ tree (default: ./outputs)")
    ap.add_argument("-o", "--out", default=PLOTS_DEFAULT,
                    help="folder to write PNG files (default: ./plots)")
    ap.add_argument("--show", action="store_true",
                    help="also display figures interactively")
    args = ap.parse_args()

    scenes = gather_positions(Path(args.root))
    if not scenes:
        print("No data found.")
        return

    outdir = Path(args.out)
    for scenario, rows in scenes.items():
        make_plot(scenario, rows, outdir)
        if args.show:
            plt.show()

if __name__ == "__main__":
    main()
