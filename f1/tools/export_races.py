#!/usr/bin/env python3
"""Export FastF1 races into Apex JSON.gz packs.

Raw FastF1 telemetry is normalised into scene units, then passed through
``track_align`` so every car is expressed in track coordinates (arc length +
lateral offset), RTS-Kalman smoothed, and snapped onto the rendered road.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_align import align_race, normalize_scene  # noqa: E402

try:
    import fastf1
except ImportError:
    print("fastf1 required", file=sys.stderr)
    raise

RAW_STEP_S = 0.5


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def seconds(td_series) -> np.ndarray:
    return td_series.dt.total_seconds().to_numpy(dtype=float)


def thin(t: np.ndarray, step: float) -> np.ndarray:
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= step:
            keep.append(i)
    return np.array(keep, dtype=int)


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

    ref_tel = laps.pick_fastest().get_telemetry()
    centerline = ref_tel[["X", "Y"]].to_numpy(dtype=float)
    centerline = centerline[np.isfinite(centerline).all(1)]

    race_start = None
    drivers = []
    for drv in sorted(laps["Driver"].unique()):
        d_laps = laps.pick_drivers(drv).sort_values("LapNumber")
        if d_laps.empty:
            continue
        try:
            tel = d_laps.get_telemetry()
        except Exception:
            continue
        if tel is None or tel.empty:
            continue
        t = seconds(tel["SessionTime"])
        idx = thin(t, RAW_STEP_S)
        tel = tel.iloc[idx]
        t = t[idx]
        lap_start = seconds(d_laps["LapStartTime"])
        lap_no = d_laps["LapNumber"].to_numpy(dtype=float)
        lap_pos = d_laps["Position"].to_numpy(dtype=float)
        li = np.clip(np.searchsorted(lap_start, t, side="right") - 1, 0, len(lap_no) - 1)
        speed = tel["Speed"].to_numpy(dtype=float)
        throttle = tel["Throttle"].to_numpy(dtype=float) / 100.0
        brake = tel["Brake"].astype(float).to_numpy()
        dist = tel["Distance"].to_numpy(dtype=float)
        rel = tel["RelativeDistance"].to_numpy(dtype=float)
        xs = tel["X"].to_numpy(dtype=float)
        ys = tel["Y"].to_numpy(dtype=float)
        samples = []
        for k in range(len(t)):
            if not (np.isfinite(xs[k]) and np.isfinite(ys[k])):
                continue
            samples.append({
                "t": float(t[k]),
                "lap": int(lap_no[li[k]]) if np.isfinite(lap_no[li[k]]) else 1,
                "position": float(lap_pos[li[k]]) if np.isfinite(lap_pos[li[k]]) else 0.0,
                "x": float(xs[k]),
                "y": float(ys[k]),
                "speed": float(np.nan_to_num(speed[k])),
                "throttle": float(np.clip(np.nan_to_num(throttle[k]), 0, 1)),
                "brake": float(np.nan_to_num(brake[k])),
                "distance": float(np.nan_to_num(dist[k])),
                "relativeDistance": float(np.nan_to_num(rel[k])),
            })
        if len(samples) < 10:
            continue
        race_start = samples[0]["t"] if race_start is None else min(race_start, samples[0]["t"])
        info = session.get_driver(drv)
        team_color = "#" + str(getattr(info, "TeamColor", "888888") or "888888").lstrip("#")
        drivers.append({
            "code": str(drv),
            "number": str(getattr(info, "DriverNumber", "")),
            "teamColor": team_color if len(team_color) == 7 else "#888888",
            "samples": samples,
        })

    if not drivers or len(centerline) < 50:
        print("  insufficient data")
        return None
    for d in drivers:
        for s in d["samples"]:
            s["t"] -= race_start

    raw = {
        "event": {"year": year, "name": event_name, "circuit": location, "country": country},
        "track": {"centerline": centerline.tolist(), "width": 5.5},
        "drivers": drivers,
        "source": "fastf1-cache",
    }
    payload, summary = align_race(normalize_scene(raw), hz=hz, rebuild_centerline=True)

    slug = slugify(event_name.replace("Grand Prix", "").strip() or location)
    race_id = f"{year}-{round_number:02d}-{slug}"
    out_path = out_dir / f"{race_id}.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  wrote {out_path.name} ({out_path.stat().st_size / 1e6:.2f} MB) {summary}", flush=True)
    return {
        "id": race_id,
        "label": f"{year} {event_name.replace('Grand Prix', 'GP').strip()}",
        "url": f"data/races/{out_path.name}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--rounds", nargs="+", type=int, default=None)
    parser.add_argument("--hz", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Re-export packs that already exist")
    args = parser.parse_args()

    cache = Path("f1/.fastf1-cache")
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    fastf1.set_log_level("ERROR")

    out_dir = Path("f1/data/races")
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = Path("f1/data/races.json")
    catalog = json.loads(catalog_path.read_text())
    by_id = {r["id"]: r for r in catalog["races"]}

    for year in args.years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rounds = args.rounds or [int(r) for r in schedule["RoundNumber"].tolist() if int(r) > 0]
        for rnd in rounds:
            event_name = str(schedule.loc[schedule["RoundNumber"] == rnd].iloc[0]["EventName"])
            slug = slugify(event_name.replace("Grand Prix", "").strip())
            race_id = f"{year}-{rnd:02d}-{slug}"
            if (out_dir / f"{race_id}.json.gz").exists() and not args.force:
                print(f"Skipping existing {race_id}")
                continue
            try:
                entry = export_session(year, rnd, out_dir, hz=args.hz)
            except Exception as e:
                print(f"  failed {race_id}: {e}")
                continue
            if entry:
                by_id[entry["id"]] = entry

    sample = [r for r in by_id.values() if r.get("sample")]
    real = sorted((r for r in by_id.values() if not r.get("sample")), key=lambda r: r["id"])
    catalog["races"] = sample + real
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Catalog now {len(catalog['races'])} entries.")


if __name__ == "__main__":
    main()
