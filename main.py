"""gta6-trophy-rarity-forecast-01 entry: a saturating-curve forecaster (Python 3, stdlib only).

Reads every <id>.in case in the input directory (schema makefun/gta6-trophies-case/v1)
and writes <id>.out: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
five quantiles of each trophy's Exophase earn rate (percent) at the case's target.

Method, per trophy, on the logit scale (rates clamped to 0.01..99.99 first, as the
scorer clamps them):

  - The read series is collapsed to the moments the page actually changed (a read
    whose rate and players_tracked both repeat the previous kept read adds no
    information, only elapsed time); the last read is always kept.
  - With at least 3 distinct points, two saturating curves are fitted by least
    squares and the better kept (the exponential only on a clear win):
      (a) a line of logit(rate) on ln(1 + hours since release) — growth that
          decelerates in time, the house trend baseline's shape, but fitted on
          the whole history of distinct states rather than the last 24 hours;
      (b) an exponential approach to a per-trophy asymptote on the logit scale,
          L(t) = L_inf - c * exp(-k * t), k chosen by a fixed grid search and
          (L_inf, c) solved linearly at each k, with c >= 0.
    The median is the fitted curve read at the target, its move from the last
    observed logit clamped to +/-4 and shrunk toward the last value when the
    distinct points are few. Anything sparser, flat or degenerate falls back to
    persistence: the last rate carried forward.
  - The spread is an honest combination, in logit units:
      sigma^2 = (0.5 * sqrt(horizon in days))^2        the random-walk term
              + (fit residual sd * sqrt(leverage))^2  curve uncertainty
              + (1 / sqrt(N * p * (1 - p)))^2         the last read's own
                                                         sampling wobble
    widened further for the sparse fallback, floored at 0.2 and capped at 3.

Deterministic: a pure function of the case file — no randomness, no clock, no
network. Input from $MAKEFUN_INPUT_DIR else /input; output to
$MAKEFUN_OUTPUT_DIR else /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantiles at the contest's levels (the scorer's own z values).
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
LO_PCT, HI_PCT = 0.01, 99.99  # the scorer's clamp, in percent
RW_PER_SQRT_DAY = 0.5         # the house random-walk spread rate, logit units
MAX_DELTA = 4.0               # largest median move from the last logit we will make
SIGMA_FLOOR, SIGMA_CAP = 0.2, 3.0


def clamp_pct(x: float) -> float:
    return min(HI_PCT, max(LO_PCT, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def norm_ppf(p: float) -> float:
    """Standard normal quantile: exact table at the contest's levels, Acklam's
    rational approximation (max error ~1e-9) anywhere else."""
    if p in Z_TABLE:
        return Z_TABLE[p]
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def round4(x: float) -> float | int:
    """Round half-up to 4 decimals; whole numbers come back as int so the JSON
    writes 100, not 100.0 (the starter's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def release_anchor(case: dict, history: list) -> datetime:
    """Midnight UTC on the game's release date; falls back to the first read's day."""
    raw = (case.get("game") or {}).get("release_date")
    if isinstance(raw, str):
        try:
            d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        except ValueError:
            pass
    for read in history:
        t = parse_time(read.get("at"))
        if t is not None:
            return datetime(t.year, t.month, t.day, tzinfo=timezone.utc)
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def lin_fit(xs: list, ys: list):
    """Least-squares y ~ a + b*x. Returns (a, b, sse, xbar, sxx) or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse, xbar, sxx


def exp_fit(ts: list, ys: list):
    """Least-squares y ~ L_inf - c*exp(-k*t) over a fixed grid of k.

    Returns (predict, sse, leverage_factor_at) as closures/values via a dict, or
    None when no admissible fit (c >= 0, finite asymptote) beats nothing at all.
    leverage is returned for a given target time through the dict's function.
    """
    n = len(ts)
    best = None
    # Time constants from 6 hours to 200 days, 61 log-spaced rates k = 1/tau.
    for i in range(61):
        tau = 6.0 * (4800.0 / 6.0) ** (i / 60.0)  # hours
        k = 1.0 / tau
        us = [math.exp(-k * t) for t in ts]
        # Regress y on [1, -u]: coefficients (L_inf, c).
        fit = lin_fit([-u for u in us], ys)
        if fit is None:
            continue
        linf, c, sse, ubar, suu = fit
        if c < 0 or not math.isfinite(linf) or linf > logit_pct(HI_PCT) + 1.0:
            continue
        if best is None or sse < best["sse"]:
            best = {"k": k, "linf": linf, "c": c, "sse": sse,
                    "ubar": ubar, "suu": suu}
    if best is None:
        return None
    k, linf, c = best["k"], best["linf"], best["c"]
    best["predict"] = lambda t: linf - c * math.exp(-k * t)
    best["leverage"] = lambda t: 1.0 + 1.0 / n + \
        ((-math.exp(-k * t)) - best["ubar"]) ** 2 / best["suu"]
    return best


def forecast_trophy(points: list, target_hours: float, horizon_days: float,
                    last_players) -> tuple:
    """points: [(hours_since_release, rate_pct)] distinct states, time-ordered,
    last point the case's last read. Returns (median_logit, sigma)."""
    last_rate = points[-1][1]
    last_logit = logit_pct(last_rate)

    # The printed rate's own sampling wobble, from the tracked-player count.
    p = clamp_pct(last_rate) / 100.0
    if last_players and last_players > 0:
        sigma_samp = min(1.0, 1.0 / math.sqrt(last_players * p * (1.0 - p)))
    else:
        sigma_samp = 0.25
    sigma_rw = RW_PER_SQRT_DAY * math.sqrt(horizon_days)

    def sparse() -> tuple:
        sigma = math.sqrt(sigma_rw ** 2 + sigma_samp ** 2 + 0.3 ** 2)
        return last_logit, min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    if len(points) < 2:
        return sparse()

    if len(points) == 2:
        # One observed move (typically a single refresh): extend its direction
        # on the ln-time line, heavily shrunk, with a fixed honest spread —
        # two points carry no residual information of their own.
        (t1, r1), (t2, r2) = points
        x1, x2 = math.log1p(t1), math.log1p(t2)
        if x2 - x1 < 1e-9:
            return sparse()
        slope = (logit_pct(r2) - logit_pct(r1)) / (x2 - x1)
        delta = slope * (math.log1p(target_hours) - x2) * 0.4
        delta = max(-MAX_DELTA, min(MAX_DELTA, delta))
        median_logit = max(logit_pct(LO_PCT),
                           min(logit_pct(HI_PCT), last_logit + delta))
        sigma = math.sqrt(sigma_rw ** 2 + sigma_samp ** 2 + 0.45 ** 2)
        return median_logit, min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    ts = [pt[0] for pt in points]
    ys = [logit_pct(pt[1]) for pt in points]
    if max(ys) - min(ys) < 1e-9:
        return sparse()  # the page never moved for this trophy

    n = len(points)
    xs = [math.log1p(t) for t in ts]
    x_target = math.log1p(target_hours)
    line = lin_fit(xs, ys)

    chosen = None  # (predict_logit, sse, leverage)
    if line is not None:
        a, b, sse, xbar, sxx = line
        lev = 1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx
        chosen = (a + b * x_target, sse, lev)
    expf = exp_fit(ts, ys)
    if expf is not None and (chosen is None or expf["sse"] < 0.92 * chosen[1]):
        chosen = (expf["predict"](target_hours), expf["sse"], expf["leverage"](target_hours))
    if chosen is None:
        return sparse()

    pred_logit, sse, leverage = chosen
    resid_sd = math.sqrt(sse / max(1, n - 2))
    sigma_fit = resid_sd * math.sqrt(max(1.0, leverage))

    delta = pred_logit - last_logit
    if n < 8:
        delta *= n / (n + 2.0)  # thin evidence: stay closer to the last read
    delta = max(-MAX_DELTA, min(MAX_DELTA, delta))
    median_logit = last_logit + delta
    median_logit = max(logit_pct(LO_PCT), min(logit_pct(HI_PCT), median_logit))

    sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_samp ** 2)
    return median_logit, min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_case(case: dict) -> dict:
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [norm_ppf(float(tau)) for tau in levels]
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    anchor = release_anchor(case, history)

    target_dt = parse_time(case.get("target"))
    if target_dt is not None:
        target_hours = (target_dt - anchor).total_seconds() / 3600.0
        horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    else:
        horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
        last_dt = parse_time(history[-1].get("at")) if history else None
        base = (last_dt - anchor).total_seconds() / 3600.0 if last_dt else 0.0
        target_hours = base + horizon_days * 24.0
    if horizon_days <= 0:
        horizon_days = 1.0

    # Distinct states per trophy: a read counts when its rate or the tracked
    # player count moved since the previous kept read; the last read always counts.
    series = [[] for _ in trophies]
    last_players = None
    for read in history:
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        t = parse_time(read.get("at"))
        hours = (t - anchor).total_seconds() / 3600.0 if t is not None else 0.0
        for i in range(len(trophies)):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            pts = series[i]
            if pts and pts[-1][1] == rate and pts[-1][2] == players:
                pts[-1] = (hours, rate, players)  # same state, later clock
            else:
                pts.append((hours, rate, players))
        last_players = players

    rows = []
    for i, trophy in enumerate(trophies):
        pts = [(h, r) for h, r, _ in series[i]]
        if pts:
            median_logit, sigma = forecast_trophy(pts, target_hours, horizon_days,
                                                  last_players)
        else:  # never seen in any read: a rare trophy, wide and low
            median_logit, sigma = logit_pct(0.5), 2.0
        q = []
        prev = 0.0
        for z in zs:
            value = float(round4(inv_logit_pct(median_logit + z * sigma)))
            value = min(100.0, max(0.0, value))
            value = max(value, prev)  # quantiles never decrease
            q.append(value)
            prev = value
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
