#!/usr/bin/env python3
"""Apply 1D Kalman smoothing to existing Apex race JSON.gz packs."""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


def kalman_1d(values: list[float], process_var: float = 8e-4, measure_var: float = 3.5) -> list[float]:
    if not values:
        return values
    x = float(values[0])
    p = 1.0
    out = []
    for z in values:
        # predict
        p = p + process_var
        # update
        k = p / (p + measure_var)
        x = x + k * (float(z) - x)
        p = (1 - k) * p
        out.append(x)
    return out


def jitter(samples: list[dict], key: str) -> float:
    if len(samples) < 3:
        return 0.0
    vals = [s[key] for s in samples]
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    if not diffs:
        return 0.0
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return math.sqrt(var)


def smooth_driver(samples: list[dict]) -> tuple[list[dict], float]:
    if len(samples) < 4:
        return samples, 0.0
    before = (jitter(samples, "x") + jitter(samples, "y")) / 2
    xs = kalman_1d([s["x"] for s in samples])
    ys = kalman_1d([s["y"] for s in samples])
    speeds = kalman_1d([s.get("speed", 0.0) for s in samples], process_var=2e-2, measure_var=6.0)
    out = []
    for i, s in enumerate(samples):
        ns = dict(s)
        ns["x"] = xs[i]
        ns["y"] = ys[i]
        ns["speed"] = max(0.0, speeds[i])
        out.append(ns)
    after = (jitter(out, "x") + jitter(out, "y")) / 2
    reduction = 0.0 if before <= 1e-9 else max(0.0, (before - after) / before)
    return out, reduction


def process_file(path: Path) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    reductions = []
    for driver in data.get("drivers", []):
        samples = driver.get("samples") or []
        smoothed, reduction = smooth_driver(samples)
        driver["samples"] = smoothed
        reductions.append(reduction)
    data.setdefault("processing", {})
    data["processing"]["kalman"] = True
    data["processing"]["spline"] = "catmull-rom-runtime"
    if reductions:
        data["processing"]["jitterReduction"] = round(sum(reductions) / len(reductions), 4)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(path)
    avg = sum(reductions) / len(reductions) if reductions else 0
    print(f"{path.name}: jitter reduction ~{avg*100:.1f}% over {len(reductions)} drivers")


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "data"
    files = sorted((root / "races").glob("*.json.gz"))
    orphan = root / "race.json.gz"
    if orphan.exists():
        files.append(orphan)
    if not files:
        print("No race files found", file=sys.stderr)
        return 1
    for path in files:
        process_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
