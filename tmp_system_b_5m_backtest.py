from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tmp_fvg_engine as eng

NY_TZ = "America/New_York"
CAPITAL = 3000.0
SEED = 20260620


@dataclass(frozen=True)
class Params:
    name: str
    entry_mode: str                 # limit50, limit618, confirm
    pivot_strength: int = 2
    min_leg_pct: float = 0.006
    max_leg_pct: float = 0.035
    min_leg_bars: int = 2
    max_leg_bars: int = 30
    max_wait_after_high_bars: int = 12
    confirm_close_location: float = 0.50
    trend_filter: bool = False
    max_stop_distance_pct: float = 0.020
    min_reward_r: float = 0.95
    first_entry_time: str = "09:45"
    last_entry_time: str = "15:30"
    force_flat_time: str = "15:55"
    max_daily_trades: int = 3
    cooldown_minutes: int = 15
    round_trip_cost_per_share: float = 0.02


VARIANTS = [
    Params("SYSTEM_B_EXACT_50", "limit50"),
    Params("SYSTEM_B_EXACT_618", "limit618", min_reward_r=1.20),
    Params("SYSTEM_B_CONFIRM", "confirm"),
    Params("SYSTEM_B_TREND_50", "limit50", trend_filter=True),
    Params("SYSTEM_B_TREND_618", "limit618", trend_filter=True, min_reward_r=1.20),
    Params("SYSTEM_B_TREND_CONFIRM", "confirm", trend_filter=True),
    Params("SYSTEM_B_TREND_618_STRICT", "limit618", pivot_strength=3,
           min_leg_pct=0.008, trend_filter=True, max_stop_distance_pct=0.015,
           min_reward_r=1.30, max_wait_after_high_bars=10),
    Params("SYSTEM_B_TREND_50_FAST", "limit50", min_leg_pct=0.004,
           trend_filter=True, max_stop_distance_pct=0.015),
]


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def session_resample(df: pd.DataFrame, rule: str, required: int) -> pd.DataFrame:
    pieces = []
    for _, day in df.groupby(df.index.date):
        bars = day.resample(rule, origin="start_day", offset="30min", label="left", closed="left").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), count=("close", "count")
        )
        bars = bars[bars["count"] >= required].dropna(subset=["open", "high", "low", "close"])
        pieces.append(bars)
    return pd.concat(pieces).sort_index() if pieces else pd.DataFrame()


def prepare(data: pd.DataFrame, input_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = eng.rth_filter(eng.ensure_ohlcv(data))
    bars5 = eng.to_5m(raw, input_minutes).copy()
    bars5 = bars5[["open", "high", "low", "close", "volume"]]

    typical = (bars5.high + bars5.low + bars5.close) / 3.0
    bars5["vwap"] = (typical * bars5.volume).groupby(bars5.index.date).cumsum() / bars5.volume.groupby(bars5.index.date).cumsum().replace(0.0, np.nan)

    bars15 = session_resample(bars5, "15min", 3)
    bars15["ema20"] = ema(bars15.close, 20)
    bars15["ema50"] = ema(bars15.close, 50)
    bars15["ema50_prev"] = bars15.ema50.shift(4)
    typical15 = (bars15.high + bars15.low + bars15.close) / 3.0
    bars15["vwap"] = (typical15 * bars15.volume).groupby(bars15.index.date).cumsum() / bars15.volume.groupby(bars15.index.date).cumsum().replace(0.0, np.nan)
    bars15["trend_ok"] = (
        (bars15.close > bars15.ema50)
        & (bars15.ema20 > bars15.ema50)
        & (bars15.ema50 > bars15.ema50_prev)
        & (bars15.close > bars15.vwap)
    )

    completed15 = bars15[["close", "ema20", "ema50", "vwap", "trend_ok"]].copy()
    completed15.index = completed15.index + pd.Timedelta(minutes=15)
    lookup_times = bars5.index + pd.Timedelta(minutes=5)
    aligned = completed15.reindex(lookup_times, method="ffill")
    aligned.index = bars5.index
    bars5["trend15_ok"] = aligned.trend_ok.fillna(False).astype(bool)
    bars5["trend15_close"] = aligned.close
    bars5["trend15_ema50"] = aligned.ema50
    return raw, bars5


def close_location(row: pd.Series) -> float:
    span = float(row.high - row.low)
    return 0.5 if span <= 0 else float((row.close - row.low) / span)


def time_string(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def pivot_low_confirmed(bars: pd.DataFrame, current_i: int, strength: int) -> int | None:
    j = current_i - strength
    if j < strength:
        return None
    start = j - strength
    end = j + strength
    if end > current_i:
        return None
    window = bars.iloc[start:end + 1]
    if len(set(window.index.date)) != 1:
        return None
    candidate = float(bars.iloc[j].low)
    if candidate <= float(window.low.min()) + 1e-12:
        # Require at least one higher low on each side to avoid a long flat tie.
        if float(bars.iloc[start:j].low.min()) > candidate or float(bars.iloc[j + 1:end + 1].low.min()) > candidate:
            return j
    return None


def create_position(order: dict, fill_time: pd.Timestamp, fill_price: float, p: Params) -> dict | None:
    stop = float(order["L"])
    target = float(order["H"])
    risk = fill_price - stop
    reward = target - fill_price
    if risk <= 0 or reward <= 0:
        return None
    if risk / fill_price > p.max_stop_distance_pct:
        return None
    if reward / risk < p.min_reward_r:
        return None
    return {
        "source": order["source"], "variant": p.name,
        "L": stop, "H": target, "entry": float(fill_price),
        "stop": stop, "target": target, "risk": risk,
        "entry_time": fill_time, "setup_time": order["setup_time"],
        "pivot_time": order["pivot_time"], "high_time": order["high_time"],
        "leg_pct": order["leg_pct"], "leg_bars": order["leg_bars"],
        "entry_mode": p.entry_mode, "bars_held": 0,
    }


def evaluate_position_on_slice(position: dict, intrabars: pd.DataFrame):
    for ts, row in intrabars.iterrows():
        o, h, l = float(row.open), float(row.high), float(row.low)
        if o <= position["stop"]:
            return ts, o, "gap_stop"
        if o >= position["target"]:
            return ts, position["target"], "gap_target"
        hit_stop = l <= position["stop"]
        hit_target = h >= position["target"]
        if hit_stop and hit_target:
            return ts, position["stop"], "both_hit_stop_first"
        if hit_stop:
            return ts, position["stop"], "stop"
        if hit_target:
            return ts, position["target"], "target"
    return None


def fill_limit_and_evaluate(order: dict, raw_slice: pd.DataFrame, p: Params):
    level = float(order["entry_level"])
    for ts, row in raw_slice.iterrows():
        o, h, l = float(row.open), float(row.high), float(row.low)
        if o <= order["L"]:
            # A buy limit above a gap-down open would fill near the open, then the structural stop is already violated.
            fill = min(o, level)
            pos = create_position(order, ts, fill, p)
            if pos is None:
                return None, None
            return pos, (ts, o, "gap_through_L")
        if o <= level:
            fill = o
        elif l <= level <= h:
            fill = level
        else:
            continue
        pos = create_position(order, ts, fill, p)
        if pos is None:
            return None, None
        # Once filled inside this minute, ordering is unknown; stop-first is deliberately conservative.
        hit_stop = l <= pos["stop"]
        hit_target = h >= pos["target"]
        if hit_stop and hit_target:
            return pos, (ts, pos["stop"], "fill_bar_both_stop_first")
        if hit_stop:
            return pos, (ts, pos["stop"], "fill_bar_stop")
        if hit_target:
            return pos, (ts, pos["target"], "fill_bar_target")
        return pos, None
    return None, None


def backtest(source: str, data: pd.DataFrame, input_minutes: int, p: Params):
    raw, bars = prepare(data, input_minutes)
    trades: list[dict] = []
    leg = None
    pending_confirm = None
    limit_order = None
    position = None
    current_day = None
    daily_trades = 0
    last_entry_time = None

    def record_exit(exit_time, exit_price, reason):
        nonlocal position
        gross = float(exit_price - position["entry"])
        net = gross - p.round_trip_cost_per_share
        trades.append({
            "source": source, "variant": p.name,
            "pivot_time": position["pivot_time"].isoformat(),
            "high_time": position["high_time"].isoformat(),
            "setup_time": position["setup_time"].isoformat(),
            "entry_time": position["entry_time"].isoformat(),
            "exit_time": pd.Timestamp(exit_time).isoformat(),
            "L_on_entry": position["L"], "H_on_entry": position["H"],
            "entry": position["entry"], "stop": position["stop"],
            "target": position["target"], "exit": float(exit_price),
            "gross_pnl_per_share": gross, "net_pnl_per_share": net,
            "r_gross": gross / position["risk"], "r_net": net / position["risk"],
            "reward_r_at_entry": (position["target"] - position["entry"]) / position["risk"],
            "leg_pct": position["leg_pct"], "leg_bars": position["leg_bars"],
            "entry_mode": position["entry_mode"], "bars_held": position["bars_held"],
            "reason": reason,
        })
        position = None

    for i, (t, row) in enumerate(bars.iterrows()):
        day = t.date()
        next_t = bars.index[i + 1] if i + 1 < len(bars) else t + pd.Timedelta(minutes=5)
        if day != current_day:
            current_day = day
            daily_trades = 0
            leg = None
            pending_confirm = None
            limit_order = None
            if position is not None:
                record_exit(t, float(row.open), "unexpected_overnight_flat")

        if position is not None and time_string(t) >= p.force_flat_time:
            record_exit(t, float(row.open), "force_flat_1555")

        raw_slice = raw[(raw.index >= t) & (raw.index < next_t)]
        if raw_slice.empty:
            raw_slice = pd.DataFrame([row[["open", "high", "low", "close", "volume"]]], index=[t])

        # Confirmation order enters at the next 5-minute open.
        if (pending_confirm is not None and pending_confirm["entry_i"] == i
                and position is None and daily_trades < p.max_daily_trades):
            cooldown_ok = last_entry_time is None or (t - last_entry_time).total_seconds() >= p.cooldown_minutes * 60
            if cooldown_ok and p.first_entry_time <= time_string(t) <= p.last_entry_time:
                position = create_position(pending_confirm, t, float(row.open), p)
                if position is not None:
                    daily_trades += 1
                    last_entry_time = t
            pending_confirm = None
            leg = None

        # Limit order is valid for this bar only and is replaced at each completed bar.
        if (limit_order is not None and limit_order["bar_i"] == i
                and position is None and daily_trades < p.max_daily_trades):
            cooldown_ok = last_entry_time is None or (t - last_entry_time).total_seconds() >= p.cooldown_minutes * 60
            if cooldown_ok and p.first_entry_time <= time_string(t) <= p.last_entry_time:
                new_position, immediate_exit = fill_limit_and_evaluate(limit_order, raw_slice, p)
                if new_position is not None:
                    position = new_position
                    daily_trades += 1
                    last_entry_time = position["entry_time"]
                    leg = None
                    if immediate_exit is not None:
                        record_exit(*immediate_exit)
            limit_order = None

        if position is not None:
            ex = evaluate_position_on_slice(position, raw_slice)
            if ex is not None:
                record_exit(*ex)
            else:
                position["bars_held"] += 1

        # Confirm a causal pivot low after the required right-side bars have closed.
        pivot_i = pivot_low_confirmed(bars, i, p.pivot_strength)
        if pivot_i is not None and position is None and pending_confirm is None:
            pivot_time = bars.index[pivot_i]
            if pivot_time.date() == day:
                window = bars.iloc[pivot_i:i + 1]
                high_rel = int(np.argmax(window.high.to_numpy(float)))
                high_i = pivot_i + high_rel
                leg = {
                    "L": float(bars.iloc[pivot_i].low), "L_i": pivot_i,
                    "pivot_time": pivot_time,
                    "H": float(bars.iloc[high_i].high), "H_i": high_i,
                    "high_time": bars.index[high_i], "touch_started": False,
                    "touch_bars": 0,
                }

        if leg is not None and position is None and pending_confirm is None:
            if float(row.low) < leg["L"]:
                leg = None
            else:
                # A fresh high resets the retracement setup; we only act from a high known at a prior close.
                if i > leg["H_i"] and float(row.high) > leg["H"] and not leg["touch_started"]:
                    leg["H"] = float(row.high)
                    leg["H_i"] = i
                    leg["high_time"] = t

                leg_bars = leg["H_i"] - leg["L_i"]
                leg_pct = leg["H"] / leg["L"] - 1.0
                high_age = i - leg["H_i"]
                qualified = (p.min_leg_bars <= leg_bars <= p.max_leg_bars
                             and p.min_leg_pct <= leg_pct <= p.max_leg_pct)
                if high_age > p.max_wait_after_high_bars:
                    leg = None
                elif qualified and i > leg["H_i"]:
                    zone50 = leg["H"] - 0.50 * (leg["H"] - leg["L"])
                    zone618 = leg["H"] - 0.618 * (leg["H"] - leg["L"])
                    intersects = float(row.low) <= zone50 and float(row.high) >= zone618
                    trend_ok = (not p.trend_filter) or (bool(row.trend15_ok) and float(row.close) > float(row.vwap))

                    if p.entry_mode == "confirm":
                        if intersects:
                            leg["touch_started"] = True
                            leg["touch_bars"] += 1
                            bullish_confirmation = (float(row.close) > float(row.open)
                                                    and float(row.close) >= zone618
                                                    and close_location(row) >= p.confirm_close_location)
                            if (bullish_confirmation and trend_ok and i + 1 < len(bars)
                                    and bars.index[i + 1].date() == day
                                    and p.first_entry_time <= time_string(bars.index[i + 1]) <= p.last_entry_time):
                                pending_confirm = {
                                    "source": source, "entry_i": i + 1,
                                    "L": leg["L"], "H": leg["H"],
                                    "setup_time": t + pd.Timedelta(minutes=5),
                                    "pivot_time": leg["pivot_time"], "high_time": leg["high_time"],
                                    "leg_pct": leg_pct, "leg_bars": leg_bars,
                                }
                            elif leg["touch_bars"] >= 2 or float(row.close) < zone618:
                                leg = None
                    else:
                        # Place a one-bar limit using only information known at this close.
                        next_i = i + 1
                        if (trend_ok and next_i < len(bars) and bars.index[next_i].date() == day
                                and p.first_entry_time <= time_string(bars.index[next_i]) <= p.last_entry_time):
                            entry_level = zone50 if p.entry_mode == "limit50" else zone618
                            limit_order = {
                                "source": source, "bar_i": next_i,
                                "entry_level": entry_level,
                                "L": leg["L"], "H": leg["H"],
                                "setup_time": t + pd.Timedelta(minutes=5),
                                "pivot_time": leg["pivot_time"], "high_time": leg["high_time"],
                                "leg_pct": leg_pct, "leg_bars": leg_bars,
                            }

    if position is not None:
        record_exit(raw.index[-1], float(raw.iloc[-1].close), "dataset_end")

    tdf = pd.DataFrame(trades)
    days = int(pd.Index(bars.index.date).nunique())
    return tdf, summarize(tdf, days, source, p)


def profit_factor(net: pd.Series):
    wins = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    if losses > 0:
        return wins / losses
    return math.inf if wins > 0 else None


def max_drawdown_from_equity(equity: pd.Series, initial: float) -> float:
    eq = pd.concat([pd.Series([initial]), equity], ignore_index=True)
    return float((eq / eq.cummax() - 1.0).min())


def cash_sim(trades: pd.DataFrame, days: int, cost: float):
    equity = CAPITAL
    curve = []
    for _, tr in trades.sort_values("entry_time").iterrows():
        shares = int(equity // float(tr.entry))
        if shares <= 0:
            continue
        equity += shares * (float(tr.gross_pnl_per_share) - cost)
        curve.append(equity)
    years = days / 252.0
    cagr = (equity / CAPITAL) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    return {
        "final_equity_cash": equity,
        "cagr_cash_100pct": cagr,
        "max_drawdown_cash": max_drawdown_from_equity(pd.Series(curve), CAPITAL) if curve else 0.0,
    }


def summarize(tdf: pd.DataFrame, days: int, source: str, p: Params):
    result = {"source": source, "variant": p.name, "trading_days": days, "trades": int(len(tdf))}
    if tdf.empty:
        return result | {"net_pnl_1share": 0.0, "annualized_1share": 0.0,
                         "win_rate": None, "profit_factor": None, "avg_r_net": None,
                         "max_drawdown_1share": 0.0, "final_equity_cash": CAPITAL,
                         "cagr_cash_100pct": 0.0, "max_drawdown_cash": 0.0}
    net = tdf.net_pnl_per_share.astype(float)
    cash = cash_sim(tdf, days, p.round_trip_cost_per_share)
    return result | {
        "net_pnl_1share": float(net.sum()),
        "annualized_1share": float(net.sum() / CAPITAL * 252.0 / days),
        "win_rate": float((net > 0).mean()),
        "profit_factor": profit_factor(net),
        "avg_r_net": float(tdf.r_net.mean()),
        "max_drawdown_1share": max_drawdown_from_equity(CAPITAL + net.cumsum(), CAPITAL),
        **cash,
    }


def monte_carlo_cash(trades: pd.DataFrame, days: int, paths: int = 100000):
    if trades.empty:
        return None
    returns = ((trades.gross_pnl_per_share - 0.02) / trades.entry).to_numpy(float)
    annual_rate = len(returns) / days * 252.0
    rng = np.random.default_rng(SEED)
    counts = rng.poisson(annual_rate, paths)
    annual = np.zeros(paths)
    for n in np.unique(counts):
        ids = np.flatnonzero(counts == n)
        if n:
            draws = rng.choice(returns, size=(len(ids), int(n)), replace=True)
            annual[ids] = np.prod(1.0 + draws, axis=1) - 1.0
    return {"paths": paths, "annual_trade_rate": float(annual_rate),
            "mean_annual_return": float(annual.mean()),
            "median_annual_return": float(np.median(annual)),
            "p05_annual_return": float(np.quantile(annual, 0.05)),
            "p95_annual_return": float(np.quantile(annual, 0.95)),
            "probability_positive": float((annual > 0).mean())}


def main():
    out = Path(os.environ.get("OUTDIR", "system_b_results")); out.mkdir(parents=True, exist_ok=True)
    erik, erik_url = eng.load_erik()
    yw, yw_url = eng.load_ywexler()
    yahoo, yahoo_url = eng.load_yahoo()
    sources = [("ERIK_2015_1M", erik, 1), ("YWEXLER_2025_1M", yw, 1), ("YAHOO_RECENT_60D_5M", yahoo, 5)]

    metric_rows = []
    trade_frames = []
    for p in VARIANTS:
        for source, data, minutes in sources:
            trades, metrics = backtest(source, data, minutes, p)
            metric_rows.append({**metrics, **{f"param_{k}": v for k, v in asdict(p).items()}})
            if not trades.empty:
                trade_frames.append(trades)

    metrics = pd.DataFrame(metric_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    metrics.to_csv(out / "segment_metrics.csv", index=False)
    trades.to_csv(out / "all_trades.csv", index=False)

    days_by_source = {source: len(set(data.index.date)) for source, data, _ in sources}
    total_days = sum(days_by_source.values())
    recent_days = days_by_source["YAHOO_RECENT_60D_5M"]
    combined_rows = []
    mc_rows = []
    reports = {}

    for p in VARIANTS:
        vt = trades[trades.variant == p.name].copy() if not trades.empty else pd.DataFrame()
        recent = vt[vt.source == "YAHOO_RECENT_60D_5M"].copy() if not vt.empty else pd.DataFrame()
        if vt.empty:
            combined = {"variant": p.name, "trading_days": total_days, "trades": 0,
                        "cagr_cash_100pct": 0.0, "annualized_1share": 0.0}
            mc_all = mc_recent = None
        else:
            net = vt.net_pnl_per_share.astype(float)
            cash = cash_sim(vt, total_days, p.round_trip_cost_per_share)
            recent_cash = cash_sim(recent, recent_days, p.round_trip_cost_per_share) if not recent.empty else {"final_equity_cash": CAPITAL, "cagr_cash_100pct": 0.0, "max_drawdown_cash": 0.0}
            recent_net = recent.net_pnl_per_share.astype(float) if not recent.empty else pd.Series(dtype=float)
            combined = {
                "variant": p.name, "trading_days": total_days, "trades": int(len(vt)),
                "net_pnl_1share": float(net.sum()),
                "annualized_1share": float(net.sum() / CAPITAL * 252.0 / total_days),
                "win_rate": float((net > 0).mean()), "profit_factor": profit_factor(net),
                "avg_r_net": float(vt.r_net.mean()), **cash,
                "recent_trades": int(len(recent)),
                "recent_net_pnl_1share": float(recent_net.sum()),
                "recent_annualized_1share": float(recent_net.sum() / CAPITAL * 252.0 / recent_days),
                "recent_win_rate": float((recent_net > 0).mean()) if len(recent_net) else None,
                "recent_profit_factor": profit_factor(recent_net) if len(recent_net) else None,
                "recent_cagr_cash_100pct": recent_cash["cagr_cash_100pct"],
                "recent_max_drawdown_cash": recent_cash["max_drawdown_cash"],
            }
            mc_all = monte_carlo_cash(vt, total_days)
            mc_recent = monte_carlo_cash(recent, recent_days) if not recent.empty else None
        combined_rows.append(combined)
        if mc_all:
            mc_rows.append({"variant": p.name, "scope": "all_real_segments", **mc_all})
        if mc_recent:
            mc_rows.append({"variant": p.name, "scope": "recent_60d", **mc_recent})
        reports[p.name] = {"parameters": asdict(p), "combined": combined,
                           "monte_carlo_all": mc_all, "monte_carlo_recent": mc_recent}

    combined_df = pd.DataFrame(combined_rows).sort_values(["cagr_cash_100pct", "profit_factor"], ascending=False)
    mc_df = pd.DataFrame(mc_rows)
    combined_df.to_csv(out / "combined_metrics.csv", index=False)
    mc_df.to_csv(out / "monte_carlo.csv", index=False)

    # Cost sensitivity for the highest-ranked full-sample variant and the literal 50% baseline.
    best_name = str(combined_df.iloc[0].variant)
    cost_rows = []
    for variant in ["SYSTEM_B_EXACT_50", best_name]:
        vt = trades[trades.variant == variant].copy()
        for cost in [0.0, 0.02, 0.05, 0.10, 0.35]:
            for scope, subset, days in [
                ("all_real_segments", vt, total_days),
                ("recent_60d", vt[vt.source == "YAHOO_RECENT_60D_5M"], recent_days),
            ]:
                sim = cash_sim(subset, days, cost) if not subset.empty else {"final_equity_cash": CAPITAL, "cagr_cash_100pct": 0.0, "max_drawdown_cash": 0.0}
                cost_rows.append({"variant": variant, "scope": scope, "cost_per_share": cost,
                                  "trades": len(subset), **sim})
    pd.DataFrame(cost_rows).to_csv(out / "cost_sensitivity.csv", index=False)

    result = {"method": {"capital": CAPITAL, "cash_positioning": "100% cash, whole shares, no leverage",
                          "fixed_share_case": 1, "entry_and_exit": "causal; 1m execution where available",
                          "same_bar_policy": "stop first", "overnight": False},
              "system_b_definition": {"L": "confirmed intraday pivot low",
                                      "H": "highest high after L, known before order placement",
                                      "entry_zone": "50%-61.8% retracement of L-to-H",
                                      "take_profit": "H_on_entry", "stop_loss": "L_on_entry"},
              "sources": {"erik": erik_url, "ywexler": yw_url, "yahoo": yahoo_url,
                          "days_by_source": days_by_source},
              "best_full_sample_variant": best_name,
              "ranking": combined_df.to_dict("records"), "variants": reports}
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=True), encoding="utf-8")

    print("COMBINED_METRICS")
    print(combined_df.to_csv(index=False))
    print("SEGMENT_METRICS")
    print(metrics[["source", "variant", "trading_days", "trades", "annualized_1share", "cagr_cash_100pct", "win_rate", "profit_factor", "avg_r_net", "max_drawdown_cash"]].to_csv(index=False))
    print("MONTE_CARLO")
    print(mc_df.to_csv(index=False))
    print("RESULT_JSON")
    print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
