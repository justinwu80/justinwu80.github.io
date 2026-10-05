#!/usr/bin/env python3
"""Downsample heavy Apex race packs so browsers can load one race safely."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


def downsample(samples: list[dict], hz: float) -> list[dict]:
    if not samples or hz <= 0:
        return samples
    step = 1.0 / hz
    out = [samples[0]]
    last_t = samples[0]["t"]
    for s in samples[1:]:
        if s["t"] - last_t >= step - 1e-9:
            out.append(s)
            last_t = s["t"]
    if out[-1] is not samples[-1]:
        out.append(samples[-1])
    return out


def round_sample(s: dict) -> dict:
    ns = dict(s)
    for key, digits in (
        ("t", 3),
        ("x", 3),
        ("y", 3),
        ("speed", 2),
        ("throttle", 3),
        ("brake", 3),
        ("distance", 2),
        ("relativeDistance", 5),
        ("position", 2),
    ):
        if key in ns and isinstance(ns[key], (int, float)):
            ns[key] = round(float(ns[key]), digits)
    return ns


def process(path: Path, hz: float, max_gz_mb: float) -> bool:
    size_mb = path.stat().st_size / 1e6
    if size_mb <= max_gz_mb and "2025-" not in path.name and path.name != "race.json.gz":
        # Keep already-lite seasonal packs unless oversized.
        if size_mb <= max_gz_mb:
            print(f"skip {path.name} ({size_mb:.1f}MB)")
            return False

    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    before = sum(len(d.get("samples") or []) for d in data.get("drivers", []))
    for driver in data.get("drivers", []):
        samples = driver.get("samples") or []
        samples = downsample(samples, hz)
        driver["samples"] = [round_sample(s) for s in samples]

    # Slim centerline
    center = data.get("track", {}).get("centerline") or []
    if len(center) > 600:
        step = max(1, len(center) // 500)
        data["track"]["centerline"] = [
            [round(float(p[0]), 3), round(float(p[1]), 3)] for p in center[::step]
        ]

    proc = data.setdefault("processing", {})
    proc["sampleHz"] = hz
    proc["lite"] = True

    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    tmp.replace(path)
    after = sum(len(d.get("samples") or []) for d in data.get("drivers", []))
    new_mb = path.stat().st_size / 1e6
    print(f"{path.name}: {size_mb:.1f}MB -> {new_mb:.1f}MB samples {before} -> {after}")
    return True


def main() -> int:
    hz = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    max_gz_mb = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    root = Path(__file__).resolve().parents[1] / "data"
    files = sorted((root / "races").glob("*.json.gz"))
    orphan = root / "race.json.gz"
    if orphan.exists():
        files.append(orphan)
    changed = 0
    for path in files:
        if process(path, hz=hz, max_gz_mb=max_gz_mb):
            changed += 1
    print(f"updated {changed}/{len(files)} packs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
