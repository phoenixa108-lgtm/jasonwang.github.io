from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tmp_fvg_engine as loader

NY_TZ = "America/New_York"
CAPITAL = 3000.0
COST = 0.05
SEED = 20260619


@dataclass(frozen=True)
class Variant:
    name: str
    trend_mode: str
    start_time: str = "09:45"
    last_entry_time: str = "15:30"
    exclude_friday: bool = False
    max_daily_trades: int = 3


VARIANTS = [
    Variant("FVG_NO_TREND", "none"),
    Variant("FVG_PRICE_VWAP", "price_vwap"),
    Variant("FVG_TREND_FAST", "fast"),
    Variant("FVG_TREND_LOOSE", "loose"),
    Variant("FVG_TREND_SLOPE", "slope"),
    Variant("FVG_TREND_LOOSE_AM", "loose", start_time="10:00", last_entry_time="12:00"),
    Variant("FVG_TREND_FAST_AM", "fast", start_time="10:00", last_entry_time="12:00"),
]


@dataclass
class Gap:
    formed_i: int
    lower: float
    upper: float
    touches: int
    swing_high: float
    impulse_pct: float
    gap_pct: float
    invalid: bool = False


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def session_resample(df, rule, count_needed):
    out=[]
    for _,day in df.groupby(df.index.date):
        x=day.resample(rule,origin="start_day",offset="30min",label="left",closed="left").agg(
            open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum"),count=("close","count"))
        x=x[x["count"]>=count_needed].dropna(subset=["open","high","low","close"])
        out.append(x)
    return pd.concat(out).sort_index() if out else pd.DataFrame()


def prepare(name,data,input_minutes):
    raw=loader.rth_filter(loader.ensure_ohlcv(data))
    b=loader.to_5m(raw,input_minutes)[["open","high","low","close","volume"]].copy()
    q=session_resample(b,"15min",3)
    q["ema20"]=ema(q.close,20);q["ema50"]=ema(q.close,50);q["ema200"]=ema(q.close,200)
    q["ema200_prev"]=q.ema200.shift(12)
    typical=(q.high+q.low+q.close)/3
    q["vwap"]=(typical*q.volume).groupby(q.index.date).cumsum()/q.volume.groupby(q.index.date).cumsum().replace(0,np.nan)
    q["price_vwap"]=(q.close>q.ema200)&(q.close>q.vwap)
    q["fast"]=(q.close>q.ema50)&(q.ema20>q.ema50)&(q.close>q.vwap)
    q["loose"]=(q.close>q.ema200)&(q.ema50>q.ema200)&(q.close>q.vwap)
    q["slope"]=q.loose&(q.ema200>q.ema200_prev)
    aligned=q[["price_vwap","fast","loose","slope"]].copy();aligned.index=aligned.index+pd.Timedelta(minutes=15)
    aligned=aligned.reindex(b.index+pd.Timedelta(minutes=5),method="ffill");aligned.index=b.index
    for col in ["price_vwap","fast","loose","slope"]: b[col]=aligned[col].fillna(False).astype(bool)
    days=np.array(sorted(set(b.index.date)),dtype=object)
    if name=="ERIK_2015_1M":
        chunks=np.array_split(days,3);fm={d:f"E{i+1}" for i,ch in enumerate(chunks) for d in ch}
    elif name=="YAHOO_RECENT_60D_5M":
        chunks=np.array_split(days,3);labels=["H_TRAIN","H_VALID","H_HOLDOUT"];fm={d:labels[i] for i,ch in enumerate(chunks) for d in ch}
    else: fm={d:"W_STRESS" for d in days}
    b["fold"]=[fm[d] for d in b.index.date]
    return {"name":name,"raw":raw,"bars":b,"days":len(days)}


def hhmm(ts):return ts.strftime("%H:%M")


def trend_ok(row,mode):
    return True if mode=="none" else bool(row[mode])


def close_location(row):
    r=float(row.high-row.low)
    return 0.5 if r<=0 else float((row.close-row.low)/r)


def backtest(ds,v):
    raw,b=ds["raw"],ds["bars"]
    active=[];pending=None;position=None;trades=[];daily={};last_entry=None

    def state(day):return daily.setdefault(day,{"n":0})
    def close_pos(t,px,reason):
        nonlocal position
        gross=float(px-position["entry"]);net=gross-COST
        trades.append({"source":ds["name"],"fold":position["fold"],"variant":v.name,
                       "formation_time":position["formation_time"].isoformat(),"signal_time":position["signal_time"].isoformat(),
                       "entry_time":position["entry_time"].isoformat(),"exit_time":pd.Timestamp(t).isoformat(),
                       "entry":position["entry"],"stop":position["stop"],"target":position["target"],"exit":float(px),
                       "risk":position["risk"],"gross_pnl":gross,"net_pnl":net,"r_net":net/position["risk"],"reason":reason,
                       "impulse_pct":position["impulse_pct"],"gap_pct":position["gap_pct"],"touch":position["touch"]})
        position=None

    for i,(t,row) in enumerate(b.iterrows()):
        day=t.date();nxt=b.index[i+1] if i+1<len(b) else t+pd.Timedelta(minutes=5)
        if position is not None and hhmm(t)>="15:55": close_pos(t,float(row.open),"force_flat")
        if pending is not None and pending["entry_i"]==i and position is None:
            st=state(day)
            cooldown=last_entry is None or (t-last_entry).total_seconds()>=15*60
            if st["n"]<v.max_daily_trades and cooldown and t.date()==pending["day"]:
                entry=float(row.open);stop=pending["lower"]*(1-0.002);risk=entry-stop
                if risk>0 and risk/entry<=0.012:
                    target=max(entry+risk,pending["swing_high"])
                    position={**pending,"entry_time":t,"entry":entry,"stop":stop,"risk":risk,"target":target,"fold":pending["fold"]}
                    st["n"]+=1;last_entry=t
            pending=None
        if position is not None:
            intr=raw[(raw.index>=t)&(raw.index<nxt)]
            if intr.empty:intr=pd.DataFrame([row[["open","high","low","close","volume"]]],index=[t])
            for xt,x in intr.iterrows():
                o,h,l=float(x.open),float(x.high),float(x.low)
                if o<=position["stop"]:close_pos(xt,o,"gap_stop");break
                if o>=position["target"]:close_pos(xt,position["target"],"gap_target");break
                hs=l<=position["stop"];ht=h>=position["target"]
                if hs and ht:close_pos(xt,position["stop"],"both_stop_first");break
                if hs:close_pos(xt,position["stop"],"stop");break
                if ht:close_pos(xt,position["target"],"target");break

        # Evaluate existing gaps before forming a new one.
        if position is None and pending is None:
            for g in reversed(active):
                if g.invalid:continue
                age=i-g.formed_i
                if age<1:continue
                if age>2:g.invalid=True;continue
                intersects=float(row.low)<=g.upper and float(row.high)>=g.lower
                if not intersects:
                    g.swing_high=max(g.swing_high,float(row.high));continue
                g.touches+=1
                if g.touches>2 or float(row.low)<g.lower or float(row.close)<g.lower:
                    g.invalid=True;continue
                if close_location(row)<0.5:continue
                if not trend_ok(row,v.trend_mode):continue
                if v.exclude_friday and t.dayofweek==4:continue
                if i+1>=len(b) or b.index[i+1].date()!=day:continue
                et=b.index[i+1]
                if not(v.start_time<=hhmm(et)<=v.last_entry_time):continue
                stop=g.lower*(1-0.002);entry=float(b.iloc[i+1].open)
                if entry<=stop or (entry-stop)/entry>0.012:continue
                pending={"entry_i":i+1,"day":day,"formation_time":b.index[g.formed_i]+pd.Timedelta(minutes=5),
                         "signal_time":t+pd.Timedelta(minutes=5),"lower":g.lower,"swing_high":g.swing_high,
                         "impulse_pct":g.impulse_pct,"gap_pct":g.gap_pct,"touch":g.touches,"fold":row.fold}
                g.invalid=True;break

        if i>=2 and b.index[i-2].date()==day and b.index[i-1].date()==day:
            a=b.iloc[i-2];m=b.iloc[i-1];c=row
            lower=float(a.high);upper=float(c.low)
            if lower<upper:
                gap_pct=(upper-lower)/lower
                impulse=(float(c.high)-float(a.low))/float(a.low)
                bulls=sum(float(x.close)>float(x.open) for x in (a,m,c))
                body=abs(float(m.close-m.open))/max(float(m.high-m.low),1e-12)
                if gap_pct>=0.0005 and impulse>=0.004 and bulls>=2 and body>=0.4:
                    active.append(Gap(i,lower,upper,0,float(c.high),impulse,gap_pct))
        # invalidate prior-session gaps
        for g in active:
            if b.index[g.formed_i].date()!=day:g.invalid=True
    if position is not None:close_pos(raw.index[-1],float(raw.iloc[-1].close),"dataset_end")
    return pd.DataFrame(trades)


def pf(s):
    w=float(s[s>0].sum());l=float(-s[s<0].sum())
    return w/l if l>0 else (math.inf if w>0 else None)


def metric(t,days):
    if t.empty:return {"trades":0,"net":0.0,"mean_r":0.0,"pf":None,"win":None,"ann":0.0,"maxdd":0.0}
    x=t.net_pnl.astype(float);eq=pd.concat([pd.Series([CAPITAL]),CAPITAL+x.cumsum()],ignore_index=True)
    return {"trades":len(t),"net":float(x.sum()),"mean_r":float(t.r_net.mean()),"pf":pf(x),"win":float((x>0).mean()),
            "ann":float(x.sum()/CAPITAL*252/max(days,1)),"maxdd":float((eq-eq.cummax()).min())}


def bootstrap(t,days,paths=200000):
    if t.empty:return None
    pnl=t.net_pnl.to_numpy(float);rate=len(pnl)/days*252;rng=np.random.default_rng(SEED);counts=rng.poisson(rate,paths);ret=np.zeros(paths)
    for n in np.unique(counts):
        ids=np.flatnonzero(counts==n)
        if n:ret[ids]=rng.choice(pnl,(len(ids),int(n)),replace=True).sum(1)/CAPITAL
    return {"paths":paths,"annual_trade_rate":float(rate),"mean":float(ret.mean()),"median":float(np.median(ret)),
            "p05":float(np.quantile(ret,.05)),"p95":float(np.quantile(ret,.95)),"prob_positive":float((ret>0).mean())}


def risk_return(t,days,risk_fraction):
    if t.empty:return None
    expected_r=float(t.r_net.sum())/days*252
    return {"risk_fraction":risk_fraction,"annualized_simple_estimate":expected_r*risk_fraction,"annual_r":expected_r}


def main():
    out=Path(os.environ.get("OUTDIR","hybrid_results"));out.mkdir(parents=True,exist_ok=True)
    erik,eu=loader.load_erik();yw,yu=loader.load_ywexler();yh,yhu=loader.load_yahoo()
    ds=[prepare("ERIK_2015_1M",erik,1),prepare("YWEXLER_2025_1M",yw,1),prepare("YAHOO_RECENT_60D_5M",yh,5)]
    fold_days={}
    for d in ds:
        for f,g in d["bars"].groupby("fold"):fold_days[f]=int(pd.Index(g.index.date).nunique())
    store={};rows=[]
    scopes={"TRAIN":["E1","E2","E3","H_TRAIN"],"VALID":["H_VALID"],"HOLDOUT":["H_HOLDOUT"],"STRESS":["W_STRESS"],
            "ALL":["E1","E2","E3","H_TRAIN","H_VALID","H_HOLDOUT","W_STRESS"]}
    reports={}
    for v in VARIANTS:
        parts=[backtest(d,v) for d in ds];t=pd.concat([x for x in parts if not x.empty],ignore_index=True) if any(not x.empty for x in parts) else pd.DataFrame()
        store[v.name]=t;reports[v.name]={}
        for s,folds in scopes.items():
            x=t[t.fold.isin(folds)] if not t.empty else pd.DataFrame();days=sum(fold_days.get(f,0) for f in folds)
            reports[v.name][s]=metric(x,days)
        train=reports[v.name]["TRAIN"];valid=reports[v.name]["VALID"]
        score=train["mean_r"]+0.5*min(train["mean_r"],0)+0.6*valid["mean_r"]+0.03*min(valid["trades"],10)
        rows.append({"variant":v.name,"selection_score":score,**{f"train_{k}":z for k,z in train.items()},**{f"valid_{k}":z for k,z in valid.items()},**asdict(v)})
    ranking=pd.DataFrame(rows).sort_values("selection_score",ascending=False)
    selected=ranking.iloc[0].variant;t=store[selected];all_days=sum(fold_days.values());hold_days=fold_days["H_HOLDOUT"]
    mc={"all":bootstrap(t,all_days),"holdout":bootstrap(t[t.fold=="H_HOLDOUT"],hold_days)}
    risk=[risk_return(t,all_days,x) for x in [0.0025,0.005,0.0075]]
    costs=[]
    for cost in [0,0.02,0.05,0.35,0.70]:
        for scope,folds in {"ALL":scopes["ALL"],"HOLDOUT":scopes["HOLDOUT"]}.items():
            x=t[t.fold.isin(folds)] if not t.empty else pd.DataFrame();days=sum(fold_days[f] for f in folds)
            net=float((x.gross_pnl-cost).sum()) if not x.empty else 0
            costs.append({"scope":scope,"cost_per_share":cost,"trades":len(x),"net":net,"ann":net/CAPITAL*252/days})
    gate={"holdout_trades_ge_5":reports[selected]["HOLDOUT"]["trades"]>=5,
          "holdout_net_positive":reports[selected]["HOLDOUT"]["net"]>0,
          "holdout_pf_ge_1_3":(reports[selected]["HOLDOUT"]["pf"] or 0)>=1.3,
          "train_net_positive":reports[selected]["TRAIN"]["net"]>0,
          "valid_net_positive":reports[selected]["VALID"]["net"]>0,
          "all_pf_ge_1_5":(reports[selected]["ALL"]["pf"] or 0)>=1.5}
    gate["pass"]=all(gate.values())
    ranking.to_csv(out/"variant_ranking.csv",index=False);t.to_csv(out/"selected_trades.csv",index=False);pd.DataFrame(costs).to_csv(out/"cost_sensitivity.csv",index=False)
    result={"method":{"cost":COST,"entry":"next_5m_open","same_bar":"stop_first","selection":"training_plus_validation_only","holdout":"last_20_recent_trading_days"},
            "fold_days":fold_days,"selected":selected,"ranking":ranking.to_dict("records"),"reports":reports,"monte_carlo":mc,
            "risk_based_estimates":risk,"cost_sensitivity":costs,"live_gate":gate,"sources":{"erik":eu,"ywexler":yu,"yahoo":yhu}}
    (out/"result.json").write_text(json.dumps(result,indent=2,default=str,allow_nan=True),encoding="utf-8")
    print("SELECTED="+selected);print("RANKING\n"+ranking.to_csv(index=False));print("REPORTS="+json.dumps(reports,default=str));print("GATE="+json.dumps(gate));print("MC="+json.dumps(mc))

if __name__=="__main__":main()
