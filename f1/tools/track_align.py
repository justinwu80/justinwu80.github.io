#!/usr/bin/env python3
"""Snap FastF1 car telemetry onto the rendered track centerline.

Every sample is converted to track coordinates: ``s`` (unwrapped arc length
along the centerline, in scene units) and ``d`` (signed lateral offset, using
the left normal ``(-ty, tx)`` exactly like the runtime in ``index-*.js``).
Both series are smoothed with a forward-backward (Rauch-Tung-Striebel) Kalman
smoother, which removes GPS jitter without the lag of a forward-only filter,
then resampled to a uniform clock. ``x``/``y`` are rebuilt from ``(s, d)`` so
every stored position lies on the asphalt the viewer draws.

Usage:
    python3 f1/tools/track_align.py RAW.json.gz [RAW2.json.gz ...] --out-dir f1/data/races
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

SCENE_EXTENT = 720.0
ROAD_SCALE = 2.4  # K0() draws asphalt at track.width * 2.4
CAR_HALF_WIDTH = 1.6


class Track:
    def __init__(self, centerline: np.ndarray):
        self.pts = np.asarray(centerline, dtype=float)
        nxt = np.roll(self.pts, -1, axis=0)
        self.seg = nxt - self.pts
        self.seg_len = np.hypot(self.seg[:, 0], self.seg[:, 1])
        self.seg_len[self.seg_len < 1e-9] = 1e-9
        self.tan = self.seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])

    def project_window(self, p: np.ndarray, idx: np.ndarray):
        a = self.pts[idx]
        ab = self.seg[idx]
        t = np.clip(((p - a) * ab).sum(1) / (self.seg_len[idx] ** 2), 0.0, 1.0)
        proj = a + ab * t[:, None]
        dist = np.hypot(*(p - proj).T)
        k = int(np.argmin(dist))
        i = int(idx[k])
        s = self.cum[i] + self.seg_len[i] * t[k]
        normal = np.array([-self.tan[i, 1], self.tan[i, 0]])
        d = float(np.dot(p - proj[k], normal))
        return s, d, float(dist[k])

    def project_global(self, p):
        return self.project_window(np.asarray(p, float), np.arange(len(self.pts)))

    def segment_at(self, s: float) -> int:
        u = s % self.length
        return int(np.clip(np.searchsorted(self.cum, u, side="right") - 1, 0, len(self.pts) - 1))

    def window(self, s: float, back: float, ahead: float) -> np.ndarray:
        start = self.segment_at(s - back)
        out = []
        covered = 0.0
        i = start
        span = back + ahead
        while covered <= span + self.seg_len[i] and len(out) < len(self.pts):
            out.append(i)
            covered += self.seg_len[i]
            i = (i + 1) % len(self.pts)
        return np.array(out, dtype=int)

    def point(self, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        u = np.mod(s, self.length)
        i = np.clip(np.searchsorted(self.cum, u, side="right") - 1, 0, len(self.pts) - 1)
        f = u - self.cum[i]
        base = self.pts[i] + self.tan[i] * f[:, None]
        normal = np.stack([-self.tan[i, 1], self.tan[i, 0]], axis=1)
        return base + normal * d[:, None]


def catmull_rom_closed(points: np.ndarray, max_seg: float) -> np.ndarray:
    pts = np.asarray(points, float)
    n = len(pts)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        steps = max(1, int(np.ceil(np.hypot(*(p2 - p1)) / max_seg)))
        for k in range(steps):
            t = k / steps
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    tr = Track(points)
    n = max(16, int(round(tr.length / spacing)))
    s = np.linspace(0.0, tr.length, n, endpoint=False)
    return tr.point(s, np.zeros_like(s))


def smooth_closed(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points
    k = np.ones(window) / window
    pad = window
    ext = np.vstack([points[-pad:], points, points[:pad]])
    sm = np.stack([np.convolve(ext[:, j], k, mode="same") for j in range(2)], axis=1)
    return sm[pad:-pad]


def rts_constant_velocity(t: np.ndarray, z: np.ndarray, v: np.ndarray, r: float, r_v: float,
                          q: float) -> np.ndarray:
    """RTS smoother on [s, ds/dt] observing both projected position and the speed channel."""
    n = len(z)
    if n < 3:
        return z.copy()
    xs = np.zeros((n, 2))
    ps = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2))
    pp = np.zeros((n, 2, 2))
    fs = np.zeros((n, 2, 2))
    x = np.array([z[0], v[0]])
    p = np.diag([r, r_v])
    rm = np.diag([r, r_v])
    for k in range(n):
        dt = t[k] - t[k - 1] if k else 0.0
        f = np.array([[1.0, dt], [0.0, 1.0]])
        qm = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        x = f @ x
        p = f @ p @ f.T + qm
        xp[k], pp[k], fs[k] = x, p, f
        if np.isfinite(v[k]):
            gain = p @ np.linalg.inv(p + rm)
            x = x + gain @ (np.array([z[k], v[k]]) - x)
            p = (np.eye(2) - gain) @ p
        else:
            gain = p[:, 0] / (p[0, 0] + r)
            x = x + gain * (z[k] - x[0])
            p = p - np.outer(gain, p[0])
        xs[k], ps[k] = x, p
    for k in range(n - 2, -1, -1):
        c = ps[k] @ fs[k + 1].T @ np.linalg.inv(pp[k + 1])
        xs[k] = xs[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = ps[k] + c @ (ps[k + 1] - pp[k + 1]) @ c.T
    return xs[:, 0]


def rts_random_walk(t: np.ndarray, z: np.ndarray, r: float, q: float) -> np.ndarray:
    n = len(z)
    if n < 3:
        return z.copy()
    xf = np.zeros(n)
    pf = np.zeros(n)
    ppred = np.zeros(n)
    x, p = z[0], r
    for k in range(n):
        dt = t[k] - t[k - 1] if k else 0.0
        p = p + q * dt
        ppred[k] = p
        g = p / (p + r)
        x = x + g * (z[k] - x)
        p = (1 - g) * p
        xf[k], pf[k] = x, p
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        c = pf[k] / ppred[k + 1]
        xs[k] = xf[k] + c * (xs[k + 1] - xf[k])
    return xs


def project_driver(track: Track, t: np.ndarray, xy: np.ndarray, odo: np.ndarray):
    """Sequentially project samples; ``odo`` (FastF1 Distance, metres) resolves lap counts across gaps."""
    n = len(t)
    s_out = np.zeros(n)
    d_out = np.zeros(n)
    dist_out = np.zeros(n)
    s0, d0, dist0 = track.project_global(xy[0])
    prev = s0
    s_out[0], d_out[0], dist_out[0] = s0, d0, dist0
    for k in range(1, n):
        dt = max(0.0, t[k] - t[k - 1])
        if dt > 10.0:
            s_loc, d, dist = track.project_global(xy[k])
            progressed = s_out[k - 1] - s_out[0]
            metres = odo[k - 1] - odo[0]
            if progressed > track.length and np.isfinite(metres) and np.isfinite(odo[k]) and metres > 0:
                expected = prev + (odo[k] - odo[k - 1]) * progressed / metres
            else:
                expected = prev
            laps = np.round((expected - s_loc) / track.length)
        else:
            idx = track.window(prev, 40.0, 40.0 + 120.0 * dt)
            s_loc, d, dist = track.project_window(xy[k], idx)
            laps = np.round((prev - s_loc) / track.length)
        s = max(prev, s_loc + laps * track.length)
        s_out[k], d_out[k], dist_out[k] = s, d, dist
        prev = s
    return s_out, d_out, dist_out


def long_runs(mask: np.ndarray, min_len: int, lead: bool = True) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[max(0, i - 1) if lead else i:j] = True
            i = j
        else:
            i += 1
    return out


def bridge_dropouts(track: Track, raw, t, frozen, t_pos, s_pos, d_pos, v_pos):
    """Anchor progress at lap-counter transitions that fall inside telemetry dropouts."""
    laps = np.array([int(s.get("lap", 1) or 1) for s in raw])
    trans = np.where(np.diff(laps) > 0)[0] + 1
    offsets = []
    for k in trans:
        if not frozen[k] and t_pos[0] <= t[k] <= t_pos[-1]:
            offsets.append(np.interp(t[k], t_pos, s_pos) - (laps[k] - 1) * track.length)
    if not offsets:
        return t_pos, s_pos, d_pos, v_pos
    offset = float(np.median(offsets))
    extra_t, extra_s = [], []
    for k in trans:
        if frozen[k]:
            extra_t.append(t[k])
            extra_s.append(offset + (laps[k] - 1) * track.length)
    if not extra_t:
        return t_pos, s_pos, d_pos, v_pos
    t_all = np.concatenate([t_pos, extra_t])
    s_all = np.concatenate([s_pos, extra_s])
    d_all = np.concatenate([d_pos, np.zeros(len(extra_t))])
    v_all = np.concatenate([v_pos, np.full(len(extra_t), np.nan)])
    order = np.argsort(t_all, kind="stable")
    t_all, s_all, d_all, v_all = t_all[order], s_all[order], d_all[order], v_all[order]
    s_all = np.maximum.accumulate(s_all)
    return t_all, s_all, d_all, v_all


def run_length(values) -> list[int]:
    """[value, count, value, count, ...]"""
    out: list[int] = []
    for v in np.asarray(values).astype(int).tolist():
        if out and out[-2] == v:
            out[-1] += 1
        else:
            out += [v, 1]
    return out


def lerp_series(t_src, values, t_dst):
    return np.interp(t_dst, t_src, values)


def step_series(t_src, values, t_dst):
    i = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 1)
    return np.asarray(values)[i]


def align_race(race: dict, hz: float, rebuild_centerline: bool) -> tuple[dict, dict]:
    centerline = np.asarray(race["track"]["centerline"], float)
    width = float(race["track"].get("width", 5.5))
    if rebuild_centerline:
        centerline = smooth_closed(resample_closed(centerline, 2.0), 3)
    centerline = resample_closed(catmull_rom_closed(centerline, 1.0), 4.0)
    track = Track(centerline)
    max_lat = width * ROAD_SCALE / 2 - CAR_HALF_WIDTH

    report = {"raw_offset_p50": [], "raw_offset_p99": []}
    projected = []
    for drv in race["drivers"]:
        raw = [s for s in drv["samples"] if np.isfinite(s.get("x", np.nan)) and np.isfinite(s.get("y", np.nan))]
        if len(raw) < 10:
            continue
        raw.sort(key=lambda s: s["t"])
        t = np.array([s["t"] for s in raw], float)
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        raw = [s for s, k in zip(raw, keep) if k]
        t = t[keep]
        xy = np.array([[s["x"], s["y"]] for s in raw], float)
        speed = np.array([float(s.get("speed", 0.0) or 0.0) for s in raw])
        step = np.concatenate([[1.0], np.hypot(*np.diff(xy, axis=0).T)])
        # FastF1 sometimes repeats a stale GPS fix while the car keeps moving, and
        # during longer dropouts every channel (speed included) freezes.
        frozen = ((step < 0.01) & (speed > 40.0)) | (long_runs(np.r_[False, np.diff(speed) == 0], 20) & (speed > 40.0))
        frozen = ~long_runs(~frozen, 4, lead=False) if frozen.any() else frozen
        if (~frozen).sum() < 10:
            continue
        odo = np.array([float(s.get("distance", np.nan) or np.nan) for s in raw])
        s_raw, d_raw, dist = project_driver(track, t[~frozen], xy[~frozen], odo[~frozen])
        report["raw_offset_p50"].append(float(np.percentile(dist, 50)))
        report["raw_offset_p99"].append(float(np.percentile(dist, 99)))
        t_pos = t[~frozen]
        v_meas = speed[~frozen]
        if frozen.any():
            t_pos, s_raw, d_raw, v_meas = bridge_dropouts(track, raw, t, frozen, t_pos, s_raw, d_raw, v_meas)
        projected.append((drv, raw, t, t_pos, s_raw, d_raw, v_meas))

    ratios = []
    for _, raw, t, t_pos, s_raw, _, _ in projected:
        dist_m = np.interp(t_pos, t, np.array([float(s.get("distance", 0.0) or 0.0) for s in raw]))
        if (s_raw[-1] - s_raw[0]) > track.length:
            ratios.append(float((dist_m[-1] - dist_m[0]) / (s_raw[-1] - s_raw[0])))
    mpu = float(np.median(ratios)) if ratios else None

    drivers_out = []
    for drv, raw, t, t_pos, s_raw, d_raw, v_meas in projected:
        cols = {k: np.array([float(s.get(k, 0.0) or 0.0) for s in raw]) for k in
                ("speed", "throttle", "brake", "distance", "relativeDistance")}
        if mpu:
            s_sm = rts_constant_velocity(t_pos, s_raw, v_meas / 3.6 / mpu, r=6.0, r_v=25.0, q=80.0)
        else:
            s_sm = rts_constant_velocity(t_pos, s_raw, np.gradient(s_raw, t_pos), r=6.0, r_v=1e4, q=300.0)
        s_sm = np.maximum.accumulate(s_sm)
        if s_sm[0] > track.length / 2:
            # Grid slots behind the start line belong to lap 0, not to the end of lap 1.
            s_sm -= track.length
        d_sm = rts_random_walk(t_pos, np.clip(d_raw, -max_lat * 1.5, max_lat * 1.5), r=4.0, q=0.3)
        d_sm = np.clip(d_sm, -max_lat, max_lat)

        grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * hz)) + 1) / hz
        gs = lerp_series(t_pos, s_sm, grid)
        gd = lerp_series(t_pos, d_sm, grid)
        laps = step_series(t, [int(s.get("lap", 1) or 1) for s in raw], grid)
        positions = step_series(t, [int(s.get("position", 0) or 0) for s in raw], grid)
        s10 = np.round(gs * 10).astype(np.int64)
        drivers_out.append({k: v for k, v in drv.items() if k != "samples"} | {
            "t0": round(float(grid[0]), 2),
            "s0": int(s10[0]),
            "ds": np.diff(s10).tolist(),
            "d": np.round(gd * 100).astype(int).tolist(),
            "v": np.round(np.maximum(0.0, lerp_series(t, cols["speed"], grid))).astype(int).tolist(),
            "thr": np.round(np.clip(lerp_series(t, cols["throttle"], grid), 0, 1) * 100).astype(int).tolist(),
            "brk": np.round(np.clip(lerp_series(t, cols["brake"], grid), 0, 1) * 100).astype(int).tolist(),
            "lap": run_length(laps),
            "pos": run_length(positions),
        })
    out = {
        "event": race["event"],
        "track": {
            "centerline": [[round(float(x), 2), round(float(y), 2)] for x, y in centerline],
            "width": width,
            "length": round(track.length, 2),
            **({"metersPerUnit": round(mpu, 4)} if mpu else {}),
        },
        "drivers": drivers_out,
        "format": "apex-columnar-1",
        "sampleHz": hz,
        "processing": {
            "trackAligned": True,
            "smoother": "rts-kalman",
            "sampleHz": hz,
            "maxLateral": round(max_lat, 2),
        },
        "source": race.get("source", "fastf1-cache"),
    }
    summary = {
        "track_length": round(track.length, 1),
        "centerline_points": len(centerline),
        "raw_offset_p50": round(float(np.median(report["raw_offset_p50"])), 2),
        "raw_offset_p99": round(float(np.max(report["raw_offset_p99"])), 2),
        "metersPerUnit": mpu,
        "drivers": len(drivers_out),
    }
    return out, summary


def normalize_scene(race: dict) -> dict:
    """Center and scale raw FastF1 coordinates so the circuit spans SCENE_EXTENT units."""
    c = np.asarray(race["track"]["centerline"], float)
    lo, hi = c.min(0), c.max(0)
    center = (lo + hi) / 2
    scale = SCENE_EXTENT / float((hi - lo).max())
    race["track"]["centerline"] = ((c - center) * scale).tolist()
    for drv in race["drivers"]:
        for s in drv["samples"]:
            s["x"] = (s["x"] - center[0]) * scale
            s["y"] = (s["y"] - center[1]) * scale
    return race


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--rebuild-centerline", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        race = json.load(gzip.open(path, "rt"))
        out, summary = align_race(race, args.hz, args.rebuild_centerline)
        dest = args.out_dir / path.name
        with gzip.open(dest, "wt", encoding="utf-8", compresslevel=9) as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"{path.name}: {dest.stat().st_size / 1e6:.2f} MB {summary}", flush=True)


if __name__ == "__main__":
    main()
