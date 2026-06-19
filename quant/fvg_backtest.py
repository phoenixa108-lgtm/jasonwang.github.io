#!/usr/bin/env python3
"""QQQ 5-minute bullish-FVG backtest on public real OHLCV data.

The script deliberately runs several defensible interpretations of ambiguous rules,
rather than silently choosing the prettiest result.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

SYMBOL = "QQQ"
CAPITAL = 3000.0
NY_TZ = "America/New_York"


@dataclass(frozen=True)
class Variant:
    name: str
    max_age_bars: int
    impulse_mode: str = "range"
    allow_low_below_gap: bool = False
    entry_mode: str = "next_open"
    eod_flat: bool = True
    target_mode: str = "prior_high_or_1R"


VARIANTS = [
    Variant("exact_age2_next_open_eod", 2),
    Variant("age6_next_open_eod", 6),
    Variant("session_next_open_eod", 78),
    Variant("session_allow_sweep_eod", 78, allow_low_below_gap=True),
    Variant("session_next_open_carry", 78, eod_flat=False),
    Variant("session_close_impulse_eod", 78, impulse_mode="close"),
    Variant("session_1R_only_eod", 78, target_mode="1R_only"),
]


@dataclass
class Candidate:
    formed_i: int
    session: object
    lower: float
    upper: float
    touches: int = 0
    touching: bool = False
    invalid: bool = False


@dataclass
class Position:
    entry_i: int
    entry_time: str
    entry: float
    stop: float
    target: float
    source_formed_i: int
    touch_no: int


def clean_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    rename = {"datetime": "timestamp", "date": "timestamp"}
    df = df.rename(columns=rename)
    required = ["open", "high", "low", "close"]
    if not all(c in df for c in required):
        raise ValueError(f"{source}: missing OHLC columns: {df.columns.tolist()}")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df:
            raise ValueError(f"{source}: missing timestamp")
        df.index = pd.to_datetime(df.pop("timestamp"), errors="coerce", utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df:
        df["volume"] = 0.0
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=required)
    t = df.index.time
    df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    df = df[(df.high >= df.low) & (df.high >= df.open) & (df.high >= df.close)
            & (df.low <= df.open) & (df.low <= df.close)]
    return df[["open", "high", "low", "close", "volume"]]


def load_yahoo_direct() -> pd.DataFrame:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/QQQ"
    params = {
        "range": "60d",
        "interval": "5m",
        "includePrePost": "false",
        "events": "div,splits",
    }
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=40)
    r.raise_for_status()
    obj = r.json()["chart"]["result"][0]
    q = obj["indicators"]["quote"][0]
    idx = pd.to_datetime(obj["timestamp"], unit="s", utc=True)
    df = pd.DataFrame(q, index=idx)
    return clean_frame(df, "Yahoo direct")


def load_yfinance() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download("QQQ", period="60d", interval="5m", auto_adjust=False,
                     prepost=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return clean_frame(df, "yfinance")


def load_yahoo() -> tuple[pd.DataFrame, str]:
    errors = []
    for label, fn in [("Yahoo chart API", load_yahoo_direct), ("yfinance", load_yfinance)]:
        try:
            df = fn()
            if len(df) >= 500:
                return df, label
            errors.append(f"{label}: only {len(df)} bars")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def load_public_github_minute_repo() -> tuple[pd.DataFrame, str]:
    tmp = Path(tempfile.mkdtemp(prefix="qqq_minute_"))
    repo = tmp / "equity-minute-data"
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/erikmattheis/equity-minute-data.git", str(repo)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    files = sorted(repo.glob("qqq-*.csv"))
    if not files:
        raise RuntimeError("no qqq-*.csv files found")
    frames = []
    for p in files:
        raw = pd.read_csv(p, skipinitialspace=True)
        cols = {c.strip().lower(): c for c in raw.columns}
        dt = pd.to_datetime(
            raw[cols["date"]].astype(str).str.strip() + " " +
            raw[cols["time"]].astype(str).str.strip(), errors="coerce"
        )
        # The public files label the US cash open as 08:30, i.e. America/Chicago.
        idx = dt.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert(NY_TZ)
        f = pd.DataFrame(index=pd.DatetimeIndex(idx))
        for c in ["open", "high", "low", "close", "volume"]:
            f[c] = pd.to_numeric(raw[cols[c]], errors="coerce").to_numpy()
        frames.append(f)
    one = pd.concat(frames).sort_index()
    one = one[~one.index.duplicated(keep="last")]
    one = clean_frame(one, "GitHub public minute CSV")
    # Resample each cash session independently so bins are 09:30, 09:35, ...
    out = []
    for _, day in one.groupby(one.index.date):
        day = day.sort_index()
        bars = day.resample("5min", origin="start_day", offset="30min").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum")
        ).dropna(subset=["open", "high", "low", "close"])
        out.append(bars)
    df = pd.concat(out).sort_index() if out else pd.DataFrame()
    return clean_frame(df, "GitHub 1m->5m"), f"GitHub erikmattheis/equity-minute-data ({len(files)} CSV files)"


def try_load_stooq() -> tuple[pd.DataFrame, str] | None:
    # Stooq does not always expose intraday history for ETFs; validate rather than assume.
    urls = [
        "https://stooq.com/q/d/l/?s=qqq.us&i=5",
        "https://stooq.com/q/d/l/?s=qqq.us&i=5m",
    ]
    for url in urls:
        try:
            raw = pd.read_csv(url)
            if len(raw) < 500:
                continue
            dt_col = next((c for c in raw.columns if c.lower() in {"datetime", "date"}), None)
            if not dt_col:
                continue
            idx = pd.to_datetime(raw.pop(dt_col), errors="coerce", utc=True)
            raw.index = idx
            df = clean_frame(raw, "Stooq")
            diffs = df.index.to_series().diff().dropna().dt.total_seconds()
            if len(df) >= 500 and diffs.median() <= 600:
                return df, "Stooq intraday CSV"
        except Exception:
            pass
    return None


def dataset_stats(df: pd.DataFrame) -> dict:
    sessions = pd.Index(df.index.date).nunique()
    return {
        "bars": int(len(df)),
        "sessions": int(sessions),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "median_bars_per_session": float(pd.Series(df.index.date).value_counts().median()),
    }


def is_last_bar_of_session(index: pd.DatetimeIndex, i: int) -> bool:
    return i == len(index) - 1 or index[i + 1].date() != index[i].date()


def backtest(df: pd.DataFrame, variant: Variant, all_in_cost: float = 0.0) -> tuple[pd.DataFrame, dict]:
    o = df.open.to_numpy(float)
    h = df.high.to_numpy(float)
    l = df.low.to_numpy(float)
    c = df.close.to_numpy(float)
    idx = df.index
    n = len(df)
    candidates: list[Candidate] = []
    position: Position | None = None
    pending: dict | None = None
    trades: list[dict] = []
    daily_orders: dict[object, int] = {}
    last_entry_i = -10_000

    def close_trade(exit_i: int, exit_px: float, reason: str) -> None:
        nonlocal position
        assert position is not None
        gross = exit_px - position.entry
        net = gross - all_in_cost
        risk = position.entry - position.stop
        trades.append({
            "entry_time": position.entry_time,
            "exit_time": str(idx[exit_i]),
            "entry": position.entry,
            "stop": position.stop,
            "target": position.target,
            "exit": float(exit_px),
            "gross_pnl": float(gross),
            "net_pnl": float(net),
            "r_multiple_gross": float(gross / risk) if risk > 0 else np.nan,
            "r_multiple_net": float(net / risk) if risk > 0 else np.nan,
            "reason": reason,
            "bars_held": int(exit_i - position.entry_i + 1),
            "touch_no": position.touch_no,
        })
        position = None

    for i in range(n):
        session = idx[i].date()

        # Execute a signal only after its bar has closed, at the next bar's open.
        if pending is not None and pending["entry_i"] == i and position is None:
            if session == pending["session"] and daily_orders.get(session, 0) < 3 and i - last_entry_i >= 3:
                entry = float(o[i]) if variant.entry_mode == "next_open" else float(c[i - 1])
                stop = pending["lower"] * (1.0 - 0.002)
                risk = entry - stop
                if risk > 0 and risk / entry <= 0.012:
                    prior_high = float(np.max(h[pending["formed_i"]:i]))
                    if variant.target_mode == "1R_only":
                        target = entry + risk
                    else:
                        target = max(entry + risk, prior_high)
                    if (target - entry) / risk >= 1.0:
                        position = Position(
                            entry_i=i, entry_time=str(idx[i]), entry=entry, stop=stop,
                            target=target, source_formed_i=pending["formed_i"],
                            touch_no=pending["touch_no"],
                        )
                        daily_orders[session] = daily_orders.get(session, 0) + 1
                        last_entry_i = i
            pending = None

        # Price-path convention: opening gaps first, then conservative stop-before-target.
        if position is not None:
            if o[i] <= position.stop:
                close_trade(i, float(o[i]), "gap_stop")
            elif o[i] >= position.target:
                close_trade(i, float(position.target), "gap_target")
            elif l[i] <= position.stop and h[i] >= position.target:
                close_trade(i, float(position.stop), "both_hit_stop_first")
            elif l[i] <= position.stop:
                close_trade(i, float(position.stop), "stop")
            elif h[i] >= position.target:
                close_trade(i, float(position.target), "target")
            elif variant.eod_flat and is_last_bar_of_session(idx, i):
                close_trade(i, float(c[i]), "eod")

        # Reset/expire candidates across sessions. The setup is intraday.
        for cand in candidates:
            if cand.session != session:
                cand.invalid = True

        # Evaluate already formed FVGs; a FVG cannot be touched on its own formation bar.
        if pending is None and position is None:
            for cand in reversed(candidates):
                if cand.invalid or cand.session != session:
                    continue
                age = i - cand.formed_i
                if age < 1:
                    continue
                if age > variant.max_age_bars:
                    cand.invalid = True
                    continue
                intersects = l[i] <= cand.upper and h[i] >= cand.lower
                if intersects and not cand.touching:
                    cand.touches += 1
                cand.touching = intersects
                if cand.touches > 2:
                    cand.invalid = True
                    continue
                if not variant.allow_low_below_gap and l[i] < cand.lower:
                    cand.invalid = True
                    continue
                if c[i] < cand.lower:
                    cand.invalid = True
                    continue
                if intersects:
                    bar_range = h[i] - l[i]
                    close_loc = (c[i] - l[i]) / bar_range if bar_range > 0 else 0.0
                    if close_loc >= 0.5 and cand.touches in (1, 2) and i + 1 < n and idx[i + 1].date() == session:
                        pending = {
                            "entry_i": i + 1,
                            "session": session,
                            "formed_i": cand.formed_i,
                            "lower": cand.lower,
                            "upper": cand.upper,
                            "touch_no": cand.touches,
                        }
                        cand.invalid = True
                        break

        # Detect a new bullish FVG at the close of bar i.
        if i >= 2 and idx[i - 2].date() == session and idx[i - 1].date() == session:
            lower = float(h[i - 2])
            upper = float(l[i])
            if lower < upper:
                gap_pct = (upper - lower) / lower
                bullish = int(c[i - 2] > o[i - 2]) + int(c[i - 1] > o[i - 1]) + int(c[i] > o[i])
                mid_range = h[i - 1] - l[i - 1]
                mid_body_ratio = abs(c[i - 1] - o[i - 1]) / mid_range if mid_range > 0 else 0.0
                if variant.impulse_mode == "close":
                    impulse_pct = (c[i] - o[i - 2]) / o[i - 2]
                else:
                    impulse_pct = (h[i] - l[i - 2]) / l[i - 2]
                if gap_pct >= 0.0005 and bullish >= 2 and mid_body_ratio >= 0.4 and impulse_pct >= 0.006:
                    candidates.append(Candidate(i, session, lower, upper))

    if position is not None:
        close_trade(n - 1, float(c[-1]), "dataset_end")

    tdf = pd.DataFrame(trades)
    sessions = pd.Index(idx.date).nunique()
    if tdf.empty:
        metrics = {
            "trades": 0, "wins": 0, "win_rate": np.nan, "net_pnl": 0.0,
            "gross_pnl": 0.0, "simple_annualized_pct": 0.0, "profit_factor": np.nan,
            "max_drawdown_dollars": 0.0, "avg_r": np.nan, "sessions": int(sessions),
        }
        return tdf, metrics
    pnl = tdf.net_pnl.to_numpy(float)
    equity = CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[CAPITAL, equity])
    eq2 = np.r_[CAPITAL, equity]
    dd = peak - eq2
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    metrics = {
        "trades": int(len(tdf)),
        "wins": int((pnl > 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "net_pnl": float(pnl.sum()),
        "gross_pnl": float(tdf.gross_pnl.sum()),
        "simple_annualized_pct": float(pnl.sum() / CAPITAL * 252 / sessions * 100) if sessions else np.nan,
        "profit_factor": float(gains / losses) if losses > 0 else math.inf,
        "max_drawdown_dollars": float(dd.max()),
        "max_drawdown_pct_of_capital": float(dd.max() / CAPITAL * 100),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(np.median(pnl)),
        "avg_r": float(tdf.r_multiple_net.mean()),
        "sessions": int(sessions),
        "trades_per_252_sessions": float(len(tdf) / sessions * 252) if sessions else np.nan,
    }
    return tdf, metrics


def monte_carlo_daily(trades: pd.DataFrame, sessions: int, seed: int = 20260619, paths: int = 100_000) -> dict:
    if trades.empty or sessions <= 0:
        return {}
    daily = trades.copy()
    daily["day"] = pd.to_datetime(daily.entry_time).dt.date
    observed = daily.groupby("day").net_pnl.sum()
    # Include zero-trade days. This preserves observed signal frequency.
    sample = np.zeros(sessions, dtype=float)
    sample[:min(len(observed), sessions)] = observed.to_numpy()[:sessions]
    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(paths, 252), replace=True)
    annual_pnl = draws.sum(axis=1)
    annual_ret = annual_pnl / CAPITAL
    # Path maximum drawdown in dollars.
    eq = CAPITAL + np.cumsum(draws, axis=1)
    eq = np.concatenate([np.full((paths, 1), CAPITAL), eq], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    maxdd = np.max(peak - eq, axis=1)
    return {
        "paths": paths,
        "mean_annual_return_pct": float(np.mean(annual_ret) * 100),
        "median_annual_return_pct": float(np.median(annual_ret) * 100),
        "p05_annual_return_pct": float(np.quantile(annual_ret, 0.05) * 100),
        "p95_annual_return_pct": float(np.quantile(annual_ret, 0.95) * 100),
        "prob_positive_pct": float(np.mean(annual_ret > 0) * 100),
        "median_max_drawdown_dollars": float(np.median(maxdd)),
        "p95_max_drawdown_dollars": float(np.quantile(maxdd, 0.95)),
    }


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if v == math.inf:
        return "∞"
    return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)


def main() -> int:
    outdir = Path(os.environ.get("OUTDIR", "artifacts"))
    outdir.mkdir(parents=True, exist_ok=True)
    data_sources: list[tuple[str, pd.DataFrame]] = []
    acquisition_log = []

    for loader_name, loader in [
        ("recent_yahoo", load_yahoo),
        ("public_github_2015", load_public_github_minute_repo),
    ]:
        try:
            df, label = loader()
            data_sources.append((loader_name, df))
            acquisition_log.append({"dataset": loader_name, "status": "ok", "label": label, **dataset_stats(df)})
            df.to_csv(outdir / f"{loader_name}_qqq_5m.csv")
        except Exception as exc:
            acquisition_log.append({"dataset": loader_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    stooq = try_load_stooq()
    if stooq is not None:
        df, label = stooq
        data_sources.append(("stooq", df))
        acquisition_log.append({"dataset": "stooq", "status": "ok", "label": label, **dataset_stats(df)})
        df.to_csv(outdir / "stooq_qqq_5m.csv")
    else:
        acquisition_log.append({"dataset": "stooq", "status": "unavailable_or_not_intraday"})

    if not data_sources:
        raise RuntimeError("No real QQQ intraday dataset could be acquired")

    rows = []
    trade_files = []
    costs = [("paper_gross", 0.0), ("slippage_0.05", 0.05), ("all_in_0.75", 0.75)]
    for ds_name, df in data_sources:
        for variant in VARIANTS:
            for cost_name, cost in costs:
                trades, metrics = backtest(df, variant, all_in_cost=cost)
                row = {"dataset": ds_name, "variant": variant.name, "cost_model": cost_name, **metrics}
                rows.append(row)
                if cost_name == "all_in_0.75":
                    p = outdir / f"trades_{ds_name}_{variant.name}.csv"
                    trades.to_csv(p, index=False)
                    trade_files.append(str(p))

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "backtest_summary.csv", index=False)

    # Primary answer: next-bar-open, session-valid FVG, intraday flat, conservative bar ordering.
    primary_name = "session_next_open_eod"
    primary_rows = summary[(summary.variant == primary_name) & (summary.cost_model == "all_in_0.75")].copy()
    pooled_trades = []
    pooled_sessions = 0
    for ds_name, df in data_sources:
        tr, _ = backtest(df, next(v for v in VARIANTS if v.name == primary_name), all_in_cost=0.75)
        if not tr.empty:
            tr = tr.copy(); tr["dataset"] = ds_name; pooled_trades.append(tr)
        pooled_sessions += pd.Index(df.index.date).nunique()
    pooled = pd.concat(pooled_trades, ignore_index=True) if pooled_trades else pd.DataFrame()
    mc = monte_carlo_daily(pooled, pooled_sessions)
    pooled.to_csv(outdir / "primary_pooled_trades.csv", index=False)

    report = []
    report.append("# QQQ 5-minute bullish-FVG real-data backtest\n")
    report.append("## Data acquisition\n")
    for x in acquisition_log:
        if x.get("status") == "ok":
            report.append(f"- **{x['dataset']}**: {x['label']}; {x['sessions']} sessions, {x['bars']} 5m bars, {x['start']} to {x['end']}")
        else:
            report.append(f"- **{x['dataset']}**: {x['status']} {x.get('error','')}")
    report.append("\n## Rule implementation\n")
    report.append("- Bullish FVG: bar1 high < bar3 low")
    report.append("- 3-bar impulse >= 0.6%; gap >= 0.05%; >=2 bullish candles; middle body/range >= 0.4")
    report.append("- First/second retest, close location >= 0.5, reclaim above lower edge")
    report.append("- Entry at next 5m open; stop = FVG lower edge -0.2%; max stop distance 1.2%")
    report.append("- Target = max(1R, post-formation prior high); stop wins same-bar ambiguity")
    report.append("- Long only, one share, max 3 entries/day, 15m cooldown, RTH only")
    report.append("- Primary implementation expires the FVG at session end and forces flat at session close")
    report.append("\n## Primary results, with $0.75 assumed all-in cost per completed 1-share trade\n")
    report.append("| Dataset | Sessions | Trades | Win rate | Net P&L | Annualized on $3,000 | PF | Max DD |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in primary_rows.iterrows():
        report.append(f"| {r.dataset} | {int(r.sessions)} | {int(r.trades)} | {fmt(r.win_rate*100 if pd.notna(r.win_rate) else np.nan)}% | ${fmt(r.net_pnl)} | {fmt(r.simple_annualized_pct)}% | {fmt(r.profit_factor)} | ${fmt(r.max_drawdown_dollars)} |")
    report.append("\n## Interpretation sensitivity\n")
    sensitivity = summary[summary.cost_model == "all_in_0.75"].pivot_table(
        index="variant", columns="dataset", values="simple_annualized_pct", aggfunc="first"
    )
    report.append(sensitivity.to_markdown(floatfmt=".2f"))
    report.append("\n## Real-trade daily bootstrap Monte Carlo for the primary interpretation\n")
    if mc:
        report.append(f"- Mean annual return: **{mc['mean_annual_return_pct']:.2f}%**")
        report.append(f"- Median annual return: **{mc['median_annual_return_pct']:.2f}%**")
        report.append(f"- 5th–95th percentile: **{mc['p05_annual_return_pct']:.2f}% to {mc['p95_annual_return_pct']:.2f}%**")
        report.append(f"- Probability of a positive year: **{mc['prob_positive_pct']:.2f}%**")
        report.append(f"- Median / 95th-percentile max drawdown: **${mc['median_max_drawdown_dollars']:.2f} / ${mc['p95_max_drawdown_dollars']:.2f}**")
    else:
        report.append("- Insufficient realized trades for Monte Carlo.")
    report.append("\n## Important caveat\n")
    report.append("The PDF's `valid_signal_bars=2` is ambiguous. The report therefore shows both a literal two-bar expiry and wider session-valid interpretations. The primary estimate is not selected by maximizing return; it is selected because next-bar execution and same-session expiry are operationally conservative and reproducible.")

    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {
        "acquisition": acquisition_log,
        "primary_variant": primary_name,
        "primary_rows": primary_rows.replace({np.nan: None, np.inf: "inf"}).to_dict("records"),
        "monte_carlo": mc,
    }
    (outdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print((outdir / "report.md").read_text(encoding="utf-8"))
    print("\nRESULT_JSON_BEGIN")
    print(json.dumps(result, ensure_ascii=False))
    print("RESULT_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
