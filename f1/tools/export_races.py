#!/usr/bin/env python3
"""Export FastF1 races into Apex JSON.gz packs with Kalman smoothing.

Produces lite packs suitable for GitHub Pages (~1-3MB each).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import fastf1
except ImportError:
    print("fastf1 required", file=sys.stderr)
    raise


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def kalman_1d(values, process_var=8e-4, measure_var=3.5):
    if len(values) == 0:
        return values
    x = float(values[0])
    p = 1.0
    out = []
    for z in values:
        p = p + process_var
        k = p / (p + measure_var)
        x = x + k * (float(z) - x)
        p = (1 - k) * p
        out.append(x)
    return out


def downsample(samples, hz=1.0):
    if not samples:
        return samples
    out = [samples[0]]
    last_t = samples[0]["t"]
    step = 1.0 / hz
    for s in samples[1:]:
        if s["t"] - last_t >= step - 1e-9:
            out.append(s)
            last_t = s["t"]
    if out[-1] is not samples[-1]:
        out.append(samples[-1])
    return out


def export_session(year: int, round_number: int, out_dir: Path, hz: float = 1.0) -> dict | None:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    row = schedule.loc[schedule["RoundNumber"] == round_number]
    if row.empty:
        return None
    event_name = str(row.iloc[0]["EventName"])
    location = str(row.iloc[0].get("Location", event_name))
    country = str(row.iloc[0].get("Country", ""))
    print(f"Loading {year} R{round_number}: {event_name}", flush=True)

    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=True, weather=False, messages=False)

    laps = session.laps
    if laps is None or laps.empty:
        print("  no laps")
        return None

    # Track outline from a representative lap
    centerline = []
    width = 5.5
    try:
        ref = laps.pick_fastest()
        tel = ref.get_telemetry()
        xs = tel["X"].to_numpy(dtype=float)
        ys = tel["Y"].to_numpy(dtype=float)
        # scale like existing packs (~meters -> scene units)
        scale = 0.1
        cx, cy = xs.mean(), ys.mean()
        centerline = [[float((x - cx) * scale), float((y - cy) * scale)] for x, y in zip(xs[::max(1, len(xs)//800)], ys[::max(1, len(xs)//800)])]
        # reuse scale for cars
    except Exception as e:
        print("  track failed", e)
        scale = 0.1
        cx = cy = 0.0

    drivers = []
    for drv in sorted(laps["Driver"].unique()):
        d_laps = laps.pick_drivers(drv)
        if d_laps.empty:
            continue
        try:
            tel = d_laps.get_telemetry()
        except Exception:
            continue
        if tel is None or tel.empty:
            continue

        # relative time from race start
        t0 = tel["SessionTime"].iloc[0]
        samples = []
        for _, row_t in tel.iterrows():
            st = row_t["SessionTime"]
            t = float((st - t0).total_seconds()) if hasattr(st - t0, "total_seconds") else float(st)
            # prefer Date-based if SessionTime weird
            samples.append({
                "t": t,
                "lap": int(row_t["LapNumber"]) if not math.isnan(row_t.get("LapNumber", float("nan"))) else 1,
                "position": float(row_t["Position"]) if "Position" in row_t and not math.isnan(row_t["Position"]) else 0.0,
                "x": float((row_t["X"] - cx) * scale),
                "y": float((row_t["Y"] - cy) * scale),
                "speed": float(row_t["Speed"]) if not math.isnan(row_t["Speed"]) else 0.0,
                "throttle": float(row_t["Throttle"]) / 100.0 if not math.isnan(row_t["Throttle"]) else 0.0,
                "brake": float(row_t["Brake"]) if isinstance(row_t["Brake"], (int, float)) else (1.0 if row_t["Brake"] else 0.0),
                "distance": float(row_t["Distance"]) if "Distance" in row_t and not math.isnan(row_t["Distance"]) else 0.0,
                "relativeDistance": float(row_t["RelativeDistance"]) if "RelativeDistance" in row_t and not math.isnan(row_t["RelativeDistance"]) else 0.0,
            })

        # Fix timestamps using Distance/SessionTime properly
        try:
            times = tel["SessionTime"]
            base = times.iloc[0]
            for i, st in enumerate(times):
                samples[i]["t"] = float((st - base).total_seconds())
        except Exception:
            pass

        samples = downsample(samples, hz=hz)
        if len(samples) < 10:
            continue

        xs = kalman_1d([s["x"] for s in samples])
        ys = kalman_1d([s["y"] for s in samples])
        speeds = kalman_1d([s["speed"] for s in samples], process_var=2e-2, measure_var=6.0)
        for i, s in enumerate(samples):
            s["x"] = round(xs[i], 3)
            s["y"] = round(ys[i], 3)
            s["speed"] = round(max(0.0, speeds[i]), 2)
            s["throttle"] = round(float(s.get("throttle", 0)), 3)
            s["brake"] = round(float(s.get("brake", 0)), 3)
            s["distance"] = round(float(s.get("distance", 0)), 2)
            s["relativeDistance"] = round(float(s.get("relativeDistance", 0)), 5)

        info = session.get_driver(drv)
        team_color = "#" + str(getattr(info, "TeamColor", "888888")).lstrip("#")
        drivers.append({
            "code": str(drv),
            "number": str(getattr(info, "DriverNumber", "")),
            "teamColor": team_color if len(team_color) == 7 else "#888888",
            "samples": samples,
        })

    if not drivers or not centerline:
        print("  insufficient data")
        return None

    # Normalize time so race starts at 0 using min t across drivers
    min_t = min(d["samples"][0]["t"] for d in drivers)
    for d in drivers:
        for s in d["samples"]:
            s["t"] = round(s["t"] - min_t, 3)

    slug = slugify(event_name.replace("Grand Prix", "").strip() or location)
    race_id = f"{year}-{round_number:02d}-{slug}"
    payload = {
        "event": {"year": year, "name": event_name, "circuit": location, "country": country},
        "track": {"centerline": centerline, "width": width},
        "drivers": drivers,
        "processing": {"kalman": True, "spline": "catmull-rom-runtime", "sampleHz": hz},
        "source": "fastf1-cache",
    }
    out_path = out_dir / f"{race_id}.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path.name} ({size_mb:.2f} MB, {len(drivers)} drivers)")
    return {
        "id": race_id,
        "label": f"{year} {event_name.replace('Grand Prix', 'GP').strip()}",
        "url": f"data/races/{out_path.name}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2023, 2022, 2021, 2020])
    parser.add_argument("--hz", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Max races to export (0=all)")
    args = parser.parse_args()

    cache = Path("f1/.fastf1-cache")
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))

    out_dir = Path("f1/data/races")
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = Path("f1/data/races.json")
    catalog = json.loads(catalog_path.read_text())
    existing_ids = {r["id"] for r in catalog["races"]}

    added = []
    count = 0
    for year in args.years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for rnd in schedule["RoundNumber"].tolist():
            if int(rnd) <= 0:
                continue
            # Skip rounds that already have an exported pack on disk.
            schedule_row = schedule.loc[schedule["RoundNumber"] == int(rnd)]
            if schedule_row.empty:
                continue
            event_name = str(schedule_row.iloc[0]["EventName"])
            slug = slugify(event_name.replace("Grand Prix", "").strip() or str(schedule_row.iloc[0].get("Location", "")))
            race_id = f"{year}-{int(rnd):02d}-{slug}"
            out_path = out_dir / f"{race_id}.json.gz"
            if out_path.exists() or race_id in existing_ids:
                print(f"Skipping existing {race_id}")
                if race_id not in existing_ids:
                    entry = {
                        "id": race_id,
                        "label": f"{year} {event_name.replace('Grand Prix', 'GP').strip()}",
                        "url": f"data/races/{out_path.name}",
                    }
                    catalog["races"].append(entry)
                    existing_ids.add(race_id)
                continue

            tentative = None
            try:
                tentative = export_session(year, int(rnd), out_dir, hz=args.hz)
            except Exception as e:
                print("  failed", e)
                continue
            if not tentative:
                continue
            catalog["races"].append(tentative)
            existing_ids.add(tentative["id"])
            added.append(tentative)
            count += 1
            if args.limit and count >= args.limit:
                break
        if args.limit and count >= args.limit:
            break

    # Keep sample demo at end; put historical before 2025? Sort by id
    sample = [r for r in catalog["races"] if r.get("sample")]
    real = [r for r in catalog["races"] if not r.get("sample")]
    real.sort(key=lambda r: r["id"])
    # Prefer adding 2021 britain orphan if present
    orphan = Path("f1/data/race.json.gz")
    if orphan.exists() and not any(r.get("url") == "data/race.json.gz" for r in real):
        real.insert(0, {"id": "2021-10-britain", "label": "2021 British GP", "url": "data/race.json.gz"})
    catalog["races"] = real + sample
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Added {len(added)} races. Catalog now {len(catalog['races'])} entries.")


if __name__ == "__main__":
    main()
