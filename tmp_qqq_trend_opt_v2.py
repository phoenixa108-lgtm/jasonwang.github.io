from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tmp_fvg_engine as loader

NY_TZ = "America/New_York"
CAPITAL = 3000.0
SEED = 20260619
BASE_COST_PER_SHARE = 0.05


@dataclass(frozen=True)
class Candidate:
    name: str
    start_time: str
    end_time: str
    rsi_min: float
    rsi_max: float
    atr_pct_min: float
    rvol_min: float
    adx_min: float
    slope_min: float
    pullback_lookback: int
    touch_mode: str
    max_extension_atr: float
    min_ema_spread_atr: float
    bullish_signal: bool
    exclude_friday: bool
    max_daily_trades: int
    stop_atr: float
    target_r: float
    time_stop_bars: int
    time_progress_r: float
    max_stop_pct: float = 0.008
    cooldown_minutes: int = 15
    max_consecutive_losses: int = 2
    max_daily_loss_r: float = 2.0


BASELINE = Candidate(
    name="BASELINE",
    start_time="09:45", end_time="15:30",
    rsi_min=45, rsi_max=72,
    atr_pct_min=0.0, rvol_min=0.0, adx_min=0.0, slope_min=0.0,
    pullback_lookback=6, touch_mode="either",
    max_extension_atr=99.0, min_ema_spread_atr=0.0,
    bullish_signal=False, exclude_friday=False, max_daily_trades=3,
    stop_atr=1.2, target_r=1.5, time_stop_bars=6, time_progress_r=0.5,
)

MANUAL_ROBUST = Candidate(
    name="MANUAL_ROBUST",
    start_time="10:15", end_time="11:30",
    rsi_min=45, rsi_max=62,
    atr_pct_min=0.0002, rvol_min=0.0, adx_min=0.0, slope_min=0.0,
    pullback_lookback=6, touch_mode="either",
    max_extension_atr=99.0, min_ema_spread_atr=0.0,
    bullish_signal=False, exclude_friday=True, max_daily_trades=2,
    stop_atr=1.2, target_r=1.5, time_stop_bars=6, time_progress_r=0.5,
)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = wilder(gain, n)
    al = wilder(loss, n)
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(al != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df.close.shift()
    tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return wilder(tr, n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df.high.diff()
    down = -df.low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * wilder(plus_dm, n) / a.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, n) / a.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return wilder(dx, n)


def session_resample(df: pd.DataFrame, rule: str, count_needed: int) -> pd.DataFrame:
    pieces = []
    for _, day in df.groupby(df.index.date):
        b = day.resample(rule, origin="start_day", offset="30min", label="left", closed="left").agg(
            open=("open","first"), high=("high","max"), low=("low","min"),
            close=("close","last"), volume=("volume","sum"), count=("close","count")
        )
        b = b[b["count"] >= count_needed].dropna(subset=["open","high","low","close"])
        pieces.append(b)
    return pd.concat(pieces).sort_index() if pieces else pd.DataFrame()


def rvol_by_slot(bars: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
    frame = pd.DataFrame({"volume": bars.volume, "date": bars.index.date,
                          "slot": bars.index.hour*60 + bars.index.minute}, index=bars.index)
    baseline = pd.Series(index=bars.index, dtype=float)
    for _, g in frame.groupby("slot"):
        vals = g.volume.shift(1).rolling(lookback_days, min_periods=5).median()
        baseline.loc[g.index] = vals
    return bars.volume / baseline.replace(0, np.nan)


def prepare(name: str, data: pd.DataFrame, input_minutes: int):
    raw = loader.rth_filter(loader.ensure_ohlcv(data))
    bars = loader.to_5m(raw, input_minutes)[["open","high","low","close","volume"]].copy()
    bars15 = session_resample(bars, "15min", 3)

    bars["ema20"] = ema(bars.close, 20)
    bars["ema50"] = ema(bars.close, 50)
    bars["ema200"] = ema(bars.close, 200)
    bars["rsi"] = rsi(bars.close, 14)
    bars["atr"] = atr(bars, 14)
    bars["atr_pct"] = bars.atr / bars.close
    bars["rvol"] = rvol_by_slot(bars)
    bars["bullish"] = bars.close > bars.open
    bars["trigger"] = bars.close > bars.high.shift(1)
    bars["extension_atr"] = (bars.close - bars.ema20) / bars.atr.replace(0, np.nan)
    bars["spread_atr"] = (bars.ema20 - bars.ema50) / bars.atr.replace(0, np.nan)
    touch20 = (bars.low <= bars.ema20) & (bars.high >= bars.ema20)
    touch50 = (bars.low <= bars.ema50) & (bars.high >= bars.ema50)
    for n in (3,4,6):
        bars[f"touch20_{n}"] = touch20.rolling(n, min_periods=1).max().fillna(0).astype(bool)
        bars[f"touch_either_{n}"] = (touch20 | touch50).rolling(n, min_periods=1).max().fillna(0).astype(bool)

    bars15["ema50"] = ema(bars15.close, 50)
    bars15["ema200"] = ema(bars15.close, 200)
    bars15["slope"] = bars15.ema200 / bars15.ema200.shift(12) - 1.0
    bars15["adx"] = adx(bars15, 14)
    typical = (bars15.high + bars15.low + bars15.close) / 3.0
    pv = typical * bars15.volume
    bars15["vwap"] = pv.groupby(bars15.index.date).cumsum() / bars15.volume.groupby(bars15.index.date).cumsum().replace(0,np.nan)
    bars15["base_trend"] = ((bars15.close > bars15.ema200) & (bars15.ema50 > bars15.ema200) &
                              (bars15.close > bars15.vwap))
    aligned = bars15[["base_trend","slope","adx"]].copy()
    aligned.index = aligned.index + pd.Timedelta(minutes=15)
    aligned = aligned.reindex(bars.index + pd.Timedelta(minutes=5), method="ffill")
    aligned.index = bars.index
    bars["base_trend"] = aligned.base_trend.fillna(False).astype(bool)
    bars["slope15"] = aligned.slope
    bars["adx15"] = aligned.adx
    bars["base_entry"] = ((bars.close > bars.ema200) & (bars.ema20 > bars.ema50) &
                            (bars.close > bars.ema20) & bars.trigger)

    days = np.array(sorted(set(bars.index.date)), dtype=object)
    if name == "ERIK_2015_1M":
        chunks = np.array_split(days, 3)
        fold_map = {d: f"E{i+1}" for i,ch in enumerate(chunks) for d in ch}
    elif name == "YAHOO_RECENT_60D_5M":
        chunks = np.array_split(days, 3)
        labels = ["H_TRAIN", "H_VALID", "H_HOLDOUT"]
        fold_map = {d: labels[i] for i,ch in enumerate(chunks) for d in ch}
    else:
        fold_map = {d: "W_STRESS" for d in days}
    bars["fold"] = [fold_map[d] for d in bars.index.date]
    return {"name":name, "raw":raw, "bars":bars, "input_minutes":input_minutes,
            "days":len(days), "fold_map":fold_map}


def hhmm(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def build_mask(b: pd.DataFrame, c: Candidate) -> np.ndarray:
    touch_col = f"touch20_{c.pullback_lookback}" if c.touch_mode == "ema20" else f"touch_either_{c.pullback_lookback}"
    mask = (b.base_trend & b.base_entry & b[touch_col] &
            b.rsi.between(c.rsi_min,c.rsi_max,inclusive="both") &
            (b.atr_pct >= c.atr_pct_min) &
            ((b.rvol >= c.rvol_min) | (c.rvol_min == 0)) &
            ((b.adx15 >= c.adx_min) | (c.adx_min == 0)) &
            ((b.slope15 >= c.slope_min) | (c.slope_min == 0)) &
            (b.extension_atr <= c.max_extension_atr) &
            (b.spread_atr >= c.min_ema_spread_atr))
    if c.bullish_signal:
        mask &= b.bullish
    return mask.fillna(False).to_numpy(bool)


def simulate(ds, c: Candidate):
    raw, b = ds["raw"], ds["bars"]
    mask = build_mask(b,c)
    signal_idx = np.flatnonzero(mask)
    trades = []
    daily = {}
    unavailable_until = pd.Timestamp.min.tz_localize(NY_TZ)

    def state(day):
        return daily.setdefault(day,{"n":0,"last_entry":None,"losses":0,"r":0.0})

    for i in signal_idx:
        if i+1 >= len(b):
            continue
        signal_time = b.index[i] + pd.Timedelta(minutes=5)
        entry_idx = i+1
        entry_time = b.index[entry_idx]
        if entry_time.date() != b.index[i].date():
            continue
        if entry_time <= unavailable_until:
            continue
        if not (c.start_time <= hhmm(entry_time) <= c.end_time):
            continue
        if c.exclude_friday and entry_time.dayofweek == 4:
            continue
        st = state(entry_time.date())
        if st["n"] >= c.max_daily_trades or st["losses"] >= c.max_consecutive_losses or st["r"] <= -c.max_daily_loss_r:
            continue
        if st["last_entry"] is not None and (entry_time-st["last_entry"]).total_seconds() < c.cooldown_minutes*60:
            continue
        entry = float(b.iloc[entry_idx].open)
        signal_atr = float(b.iloc[i].atr)
        if not np.isfinite(signal_atr) or signal_atr <= 0:
            continue
        risk = c.stop_atr * signal_atr
        if risk/entry > c.max_stop_pct:
            continue
        stop = entry-risk
        target = entry+c.target_r*risk
        last_j = entry_idx
        while last_j+1 < len(b) and b.index[last_j+1].date() == entry_time.date() and hhmm(b.index[last_j+1]) < "15:55":
            last_j += 1
        max_high = entry
        exit_time = b.index[last_j] + pd.Timedelta(minutes=5)
        exit_px = float(b.iloc[last_j].close)
        reason = "force_flat"
        bars_held = 0
        closed = False
        for j in range(entry_idx,last_j+1):
            t0=b.index[j]
            t1=b.index[j+1] if j+1<len(b) else t0+pd.Timedelta(minutes=5)
            intr=raw[(raw.index>=t0)&(raw.index<t1)]
            if intr.empty:
                intr=pd.DataFrame([b.iloc[j][["open","high","low","close","volume"]]],index=[t0])
            for xt,x in intr.iterrows():
                o,h,l=float(x.open),float(x.high),float(x.low)
                max_high=max(max_high,h)
                if o<=stop:
                    exit_time,exit_px,reason=xt,o,"gap_stop";closed=True;break
                if o>=target:
                    exit_time,exit_px,reason=xt,target,"gap_target";closed=True;break
                hs=l<=stop; ht=h>=target
                if hs and ht:
                    exit_time,exit_px,reason=xt,stop,"both_stop_first";closed=True;break
                if hs:
                    exit_time,exit_px,reason=xt,stop,"stop";closed=True;break
                if ht:
                    exit_time,exit_px,reason=xt,target,"target";closed=True;break
            bars_held += 1
            if closed:
                break
            if bars_held >= c.time_stop_bars and (max_high-entry)/risk < c.time_progress_r:
                exit_time,exit_px,reason=t1-pd.Timedelta(microseconds=1),float(b.iloc[j].close),"time_stop"
                closed=True;break
        gross=exit_px-entry
        net=gross-BASE_COST_PER_SHARE
        r_net=net/risk
        trades.append({"source":ds["name"],"fold":b.iloc[i].fold,"candidate":c.name,
                       "signal_time":signal_time.isoformat(),"entry_time":entry_time.isoformat(),"exit_time":pd.Timestamp(exit_time).isoformat(),
                       "entry":entry,"stop":stop,"target":target,"exit":exit_px,"risk":risk,
                       "gross_pnl":gross,"net_pnl":net,"r_net":r_net,"reason":reason,
                       "signal_rsi":float(b.iloc[i].rsi),"signal_atr_pct":float(b.iloc[i].atr_pct),
                       "signal_rvol":float(b.iloc[i].rvol) if np.isfinite(b.iloc[i].rvol) else None,
                       "signal_adx15":float(b.iloc[i].adx15) if np.isfinite(b.iloc[i].adx15) else None,
                       "signal_slope15":float(b.iloc[i].slope15) if np.isfinite(b.iloc[i].slope15) else None})
        st["n"] += 1
        st["last_entry"] = entry_time
        st["r"] += gross/risk
        st["losses"] = st["losses"]+1 if net<0 else 0
        unavailable_until = pd.Timestamp(exit_time)
    return pd.DataFrame(trades)


def pf(x: pd.Series):
    w=float(x[x>0].sum()); l=float(-x[x<0].sum())
    return w/l if l>0 else (math.inf if w>0 else None)


def metrics(t: pd.DataFrame, days: int):
    if t.empty:
        return {"trades":0,"net":0.0,"mean_r":0.0,"pf":None,"win":None,"ann":0.0,"maxdd":0.0}
    x=t.net_pnl.astype(float); r=t.r_net.astype(float)
    eq=pd.concat([pd.Series([CAPITAL]),CAPITAL+x.cumsum()],ignore_index=True)
    return {"trades":len(t),"net":float(x.sum()),"mean_r":float(r.mean()),"pf":pf(x),"win":float((x>0).mean()),
            "ann":float(x.sum()/CAPITAL*252/days),"maxdd":float((eq-eq.cummax()).min())}


def random_candidates(n=160):
    rng=random.Random(SEED)
    out=[BASELINE,MANUAL_ROBUST]
    seen={json.dumps(asdict(x),sort_keys=True) for x in out}
    while len(out)<n+2:
        start=rng.choice(["10:00","10:15","10:30"])
        end=rng.choice(["11:00","11:30","12:00","13:00","14:00"])
        if end<=start: continue
        rmin=rng.choice([45,48,50,52])
        rmax=rng.choice([60,62,65,68,70])
        if rmax<=rmin+5: continue
        c=Candidate(
            name=f"R{len(out)-1:03d}", start_time=start,end_time=end,rsi_min=rmin,rsi_max=rmax,
            atr_pct_min=rng.choice([0.0,0.00015,0.0002,0.00025,0.0003]),
            rvol_min=rng.choice([0.0,0.8,1.0,1.2]), adx_min=rng.choice([0.0,15.0,20.0,25.0]),
            slope_min=rng.choice([0.0,0.0001,0.0002,0.0003]), pullback_lookback=rng.choice([3,4,6]),
            touch_mode=rng.choice(["ema20","either"]), max_extension_atr=rng.choice([0.5,1.0,1.5,99.0]),
            min_ema_spread_atr=rng.choice([0.0,0.2,0.4]), bullish_signal=rng.choice([False,True]),
            exclude_friday=rng.choice([False,True]), max_daily_trades=rng.choice([1,2]),
            stop_atr=rng.choice([1.0,1.2,1.5]), target_r=rng.choice([1.5,2.0,2.5]),
            time_stop_bars=rng.choice([6,9,12]), time_progress_r=rng.choice([0.25,0.5]))
        key=json.dumps({k:v for k,v in asdict(c).items() if k!="name"},sort_keys=True)
        if key in seen: continue
        seen.add(key);out.append(c)
    return out


def fold_days(datasets):
    out={}
    for ds in datasets:
        b=ds["bars"]
        for fold,g in b.groupby("fold"):
            out[fold]=int(pd.Index(g.index.date).nunique())
    return out


def score_training(t, fd):
    train_folds=["E1","E2","E3","H_TRAIN"]
    vals=[]; active=0; total=0
    for f in train_folds:
        x=t[t.fold==f]
        total+=len(x)
        if len(x)>=3:
            active+=1; vals.append(float(x.r_net.mean()))
        else:
            vals.append(-0.15)
    a=np.array(vals,float)
    annual_trades=total/sum(fd[f] for f in train_folds)*252 if total else 0
    score=float(np.median(a)+0.5*a.min()+0.15*a.mean()-0.2*a.std()+0.003*min(total,40)-0.0015*max(annual_trades-60,0))
    if total<15: score-=0.5
    if active<3: score-=0.4
    return score,total,active,a,annual_trades


def bootstrap(t: pd.DataFrame, days: int, paths=200000):
    if t.empty:return None
    pnl=t.net_pnl.to_numpy(float); rate=len(pnl)/days*252
    rng=np.random.default_rng(SEED); counts=rng.poisson(rate,paths); ret=np.zeros(paths)
    for n in np.unique(counts):
        ids=np.flatnonzero(counts==n)
        if n:ret[ids]=rng.choice(pnl,(len(ids),int(n)),replace=True).sum(1)/CAPITAL
    return {"paths":paths,"annual_trade_rate":float(rate),"mean":float(ret.mean()),"median":float(np.median(ret)),
            "p05":float(np.quantile(ret,.05)),"p95":float(np.quantile(ret,.95)),"prob_positive":float((ret>0).mean())}


def sizing_sim(t: pd.DataFrame, risk_frac: float, fixed_roundtrip: float, slip_per_share: float, leverage: float=1.0):
    if t.empty:return None
    x=t.copy();x["dt"]=pd.to_datetime(x.entry_time,utc=True);x=x.sort_values("dt")
    equity=CAPITAL; peak=equity; maxdd=0; used=0
    for _,r in x.iterrows():
        max_qty=int((equity*0.95*leverage)//r.entry)
        risk_qty=int((equity*risk_frac)//r.risk)
        qty=min(max_qty,risk_qty)
        if qty<1:continue
        pnl=r.gross_pnl*qty-fixed_roundtrip-slip_per_share*qty
        equity+=pnl;used+=1;peak=max(peak,equity);maxdd=min(maxdd,equity-peak)
    return {"risk_fraction":risk_frac,"leverage":leverage,"trades_used":used,"ending_equity":equity,
            "total_return":equity/CAPITAL-1,"max_drawdown_dollars":maxdd}


def main():
    out=Path(os.environ.get("OUTDIR","trend_opt_results"));out.mkdir(parents=True,exist_ok=True)
    erik,eu=loader.load_erik();yw,yu=loader.load_ywexler();yh,yhu=loader.load_yahoo()
    datasets=[prepare("ERIK_2015_1M",erik,1),prepare("YWEXLER_2025_1M",yw,1),prepare("YAHOO_RECENT_60D_5M",yh,5)]
    fd=fold_days(datasets)
    candidates=random_candidates(160)
    rows=[];trade_store={}
    for k,c in enumerate(candidates):
        parts=[simulate(ds,c) for ds in datasets]
        t=pd.concat([x for x in parts if not x.empty],ignore_index=True) if any(not x.empty for x in parts) else pd.DataFrame()
        trade_store[c.name]=t
        sc,total,active,vals,annual=score_training(t,fd)
        valid=t[t.fold=="H_VALID"] if not t.empty else pd.DataFrame()
        stress=t[t.fold=="W_STRESS"] if not t.empty else pd.DataFrame()
        vm=metrics(valid,fd.get("H_VALID",1)); sm=metrics(stress,fd.get("W_STRESS",1))
        rows.append({"candidate":c.name,"train_score":sc,"train_trades":total,"active_train_folds":active,
                     "train_fold_r_min":float(vals.min()),"train_fold_r_median":float(np.median(vals)),"train_annual_trades":annual,
                     "valid_trades":vm["trades"],"valid_mean_r":vm["mean_r"],"valid_pf":vm["pf"],"valid_net":vm["net"],
                     "stress_trades":sm["trades"],"stress_net":sm["net"],"stress_pf":sm["pf"],**{f"p_{a}":b for a,b in asdict(c).items()}})
        if (k+1)%20==0:print(f"evaluated {k+1}/{len(candidates)}",flush=True)
    scores=pd.DataFrame(rows).sort_values("train_score",ascending=False).reset_index(drop=True)
    scores["train_rank"]=np.arange(1,len(scores)+1)
    top=scores.head(20).copy()
    # Validation chooses among the top-20 training candidates; final holdout remains unseen.
    top["selection_score"]=top.train_score+0.65*top.valid_mean_r.fillna(-0.2)+0.04*np.minimum(top.valid_trades,10)-0.05*(top.stress_net.fillna(0)<0)
    selected_name=top.sort_values("selection_score",ascending=False).iloc[0].candidate
    selected=next(c for c in candidates if c.name==selected_name)
    selected_trades=trade_store[selected_name]
    baseline_trades=trade_store["BASELINE"]
    manual_trades=trade_store["MANUAL_ROBUST"]

    def report_variant(name,t):
        scopes={}
        for scope,folds in {"TRAIN":["E1","E2","E3","H_TRAIN"],"VALID":["H_VALID"],"HOLDOUT":["H_HOLDOUT"],
                            "STRESS":["W_STRESS"],"ALL":["E1","E2","E3","H_TRAIN","H_VALID","H_HOLDOUT","W_STRESS"]}.items():
            x=t[t.fold.isin(folds)] if not t.empty else pd.DataFrame()
            days=sum(fd.get(f,0) for f in folds)
            scopes[scope]=metrics(x,max(days,1))
        return scopes

    reports={"BASELINE":report_variant("BASELINE",baseline_trades),"MANUAL_ROBUST":report_variant("MANUAL_ROBUST",manual_trades),
             selected_name:report_variant(selected_name,selected_trades)}
    all_days=sum(fd.values())
    hold_days=fd["H_HOLDOUT"]
    mc_all=bootstrap(selected_trades,all_days)
    mc_hold=bootstrap(selected_trades[selected_trades.fold=="H_HOLDOUT"],hold_days)

    cost_rows=[]
    for cost in [0,0.02,0.05,0.35,0.70,1.0]:
        for label,x,days in [("ALL",selected_trades,all_days),("HOLDOUT",selected_trades[selected_trades.fold=="H_HOLDOUT"],hold_days)]:
            net=float((x.gross_pnl-cost).sum()) if not x.empty else 0.0
            cost_rows.append({"scope":label,"roundtrip_cost_per_share":cost,"trades":len(x),"net":net,"annualized":net/CAPITAL*252/max(days,1)})
    cost_df=pd.DataFrame(cost_rows)

    sizing=[]
    for risk_frac in [0.0025,0.005]:
        for leverage in [1.0,2.0]:
            z=sizing_sim(selected_trades,risk_frac,fixed_roundtrip=0.70,slip_per_share=0.02,leverage=leverage)
            if z:sizing.append(z)

    live_gate={
        "holdout_trades_at_least_5":reports[selected_name]["HOLDOUT"]["trades"]>=5,
        "holdout_net_positive":reports[selected_name]["HOLDOUT"]["net"]>0,
        "holdout_pf_at_least_1_3":(reports[selected_name]["HOLDOUT"]["pf"] or 0)>=1.3,
        "validation_net_positive":reports[selected_name]["VALID"]["net"]>0,
        "all_pf_at_least_1_5":(reports[selected_name]["ALL"]["pf"] or 0)>=1.5,
        "positive_at_0_35_cost":float(cost_df[(cost_df.scope=="ALL")&(cost_df.roundtrip_cost_per_share==0.35)].net.iloc[0])>0,
    }
    live_gate["pass"] = all(live_gate.values())

    scores.to_csv(out/"candidate_scores.csv",index=False)
    top.sort_values("selection_score",ascending=False).to_csv(out/"top20_validation.csv",index=False)
    selected_trades.to_csv(out/"selected_trades.csv",index=False)
    baseline_trades.to_csv(out/"baseline_trades.csv",index=False)
    cost_df.to_csv(out/"cost_sensitivity.csv",index=False)
    pd.DataFrame(sizing).to_csv(out/"risk_sizing.csv",index=False)
    result={"method":{"candidate_count":len(candidates),"base_cost_per_share":BASE_COST_PER_SHARE,
                      "training_folds":["E1","E2","E3","H_TRAIN"],"validation_fold":"H_VALID","final_holdout":"H_HOLDOUT",
                      "same_bar_policy":"stop_first","entry":"next_5m_open","seed":SEED},
            "fold_days":fd,"selected":{"name":selected_name,"params":asdict(selected)},"reports":reports,
            "monte_carlo":{"all":mc_all,"holdout":mc_hold},"cost_sensitivity":cost_rows,"risk_sizing":sizing,"live_gate":live_gate,
            "data_sources":{"erik":eu,"ywexler":yu,"yahoo":yhu}}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str,allow_nan=True),encoding="utf-8")
    print("SELECTED="+json.dumps(result["selected"],default=str))
    print("REPORTS="+json.dumps(reports,default=str,allow_nan=True))
    print("LIVE_GATE="+json.dumps(live_gate))
    print("MONTE_CARLO="+json.dumps(result["monte_carlo"],allow_nan=True))
    print("TOP5")
    print(top.sort_values("selection_score",ascending=False).head(5)[["candidate","train_score","valid_trades","valid_mean_r","valid_pf","selection_score"]].to_csv(index=False))

if __name__=="__main__":main()
