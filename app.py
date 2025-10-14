
import io
import uuid
import json
import time
import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil import parser
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Trade Surveillance: IFA Detector", layout="wide")

APP_VERSION = "v0.1"
DEFAULT_RANDOM_SEED = 42
np.random.seed(DEFAULT_RANDOM_SEED)
random.seed(DEFAULT_RANDOM_SEED)

# -----------------------------
# HELPERS
# -----------------------------
def _to_dt(x):
    if isinstance(x, (datetime, np.datetime64)):
        return pd.to_datetime(x)
    try:
        return pd.to_datetime(parser.parse(str(x)))
    except Exception:
        return pd.NaT

def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def winsorize(s, p=0.01):
    lower = s.quantile(p)
    upper = s.quantile(1-p)
    return s.clip(lower, upper)

def safe_div(a, b):
    b = (b if b not in [0, None, np.nan] else 1e-9)
    return a / b

def generate_synth_trades(n=1500, start_date=None):
    """Generate synthetic blotter-like data with occasional suspicious trades near 'client orders' and news."""
    if start_date is None:
        start_date = datetime.now(tz=timezone.utc) - timedelta(days=5)
    symbols = ["AAPL","MSFT","AMZN","TSLA","NVDA","META","JPM","GS","C","BAC","NFLX","ORCL","SHOP"]
    desks = ["EQ-Agency","EQ-Prop","EQ-Delta1","Derivs-Flow"]
    traders = [f"TRDR{str(i).zfill(3)}" for i in range(1,31)]
    accounts = [f"ACCT{str(i).zfill(4)}" for i in range(1,61)]
    side = ["BUY","SELL"]

    rows = []
    # Simulate a "client order tape" and "news events"
    client_orders = []
    news_events = []
    for s in symbols:
        for d in range(5):
            t = start_date + timedelta(days=d, hours=np.random.randint(14, 21))
            client_orders.append({
                "symbol": s,
                "client_order_time": t,
                "client_order_qty": np.random.randint(5_000, 50_000)
            })
            # News either earnings or rating change
            if np.random.rand() < 0.5:
                tnews = t - timedelta(hours=np.random.randint(1, 6)) if np.random.rand()<0.5 else t + timedelta(hours=np.random.randint(1, 6))
                news_events.append({
                    "symbol": s,
                    "news_time": tnews,
                    "news_type": np.random.choice(["EARNINGS","RATING_CHANGE","GUIDANCE","M&A_RUMOR"]),
                    "impact": np.random.choice(["LOW","MEDIUM","HIGH"], p=[0.3,0.5,0.2])
                })

    base_price = {s: np.random.uniform(20, 500) for s in symbols}
    for i in range(n):
        s = np.random.choice(symbols)
        t = start_date + timedelta(minutes=np.random.randint(0, 5*24*60))
        px = max(1, np.random.normal(base_price[s], base_price[s]*0.02))
        qty = int(np.abs(np.random.normal(2000, 1500))) + 10
        row = {
            "trade_id": str(uuid.uuid4()),
            "timestamp": t,
            "symbol": s,
            "side": np.random.choice(side, p=[0.55, 0.45]),
            "price": round(px, 2),
            "quantity": qty,
            "trader_id": np.random.choice(traders),
            "account_id": np.random.choice(accounts),
            "desk": np.random.choice(desks),
            "client_order_overlap": 0,  # will fill below
            "pnl_est": np.random.normal(0, 300),
        }
        rows.append(row)

    trades = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    client_df = pd.DataFrame(client_orders)
    news_df = pd.DataFrame(news_events)

    # Feature engineering: overlap with client order time window (±15 min)
    trades["client_order_overlap"] = 0
    if not client_df.empty:
        for idx, co in client_df.iterrows():
            mask = (trades["symbol"]==co["symbol"]) & (np.abs((trades["timestamp"] - co["client_order_time"]).dt.total_seconds()) <= 15*60)
            trades.loc[mask, "client_order_overlap"] = 1
            # Inflate some suspicious quantities around overlap to simulate front-running
            trades.loc[mask & (np.random.rand(mask.sum())>0.5), "quantity"] *= np.random.randint(2,5)

    # Approximate market features per 5-min bucket
    trades["bucket"] = trades["timestamp"].dt.floor("5min")
    grp = trades.groupby(["symbol","bucket"])
    trades["bucket_vol"] = grp["quantity"].transform("sum")
    trades["bucket_trades"] = grp["quantity"].transform("count")
    trades["avg_qty_symbol"] = trades.groupby("symbol")["quantity"].transform("mean")
    trades["qty_vs_avg"] = trades["quantity"] / (trades["avg_qty_symbol"] + 1e-9)

    # News proximity (minutes) by nearest event per symbol
    trades["mins_to_news"] = np.nan
    if not news_df.empty:
        news_map = {s: news_df.loc[news_df["symbol"]==s, "news_time"].sort_values().to_list() for s in symbols}
        def min_minutes_to_news(sym, ts):
            if sym not in news_map or len(news_map[sym])==0:
                return np.nan
            return min([abs((ts - nt).total_seconds())/60.0 for nt in news_map[sym]])
        trades["mins_to_news"] = trades.apply(lambda r: min_minutes_to_news(r["symbol"], r["timestamp"]), axis=1)

    # Abnormal bucket volume z-score per symbol
    trades["bucket_vol_z"] = trades.groupby("symbol")["bucket_vol"].transform(lambda s: zscore(s))

    # Price move proxy: simulate % change next 15min (for demo)
    # Higher move if near "news"
    move = np.random.normal(0, 0.3, size=len(trades))
    near_news = (trades["mins_to_news"].fillna(999) <= 30).astype(float)
    move += near_news * np.random.normal(0.6, 0.4, size=len(trades))
    trades["ret_fwd_15m_pct"] = move

    # Directional alignment heuristic: BUY before positive move, SELL before negative move
    trades["directional_align"] = np.where(
        ((trades["side"]=="BUY") & (trades["ret_fwd_15m_pct"]>0)) |
        ((trades["side"]=="SELL") & (trades["ret_fwd_15m_pct"]<0)), 1, 0
    )

    # PnL uplift proxy
    trades["pnl_uplift"] = trades["quantity"] * trades["ret_fwd_15m_pct"] * trades["price"] * np.where(trades["side"]=="BUY", 1, -1)
    trades["pnl_uplift"] = trades["pnl_uplift"] + np.random.normal(0, 200, size=len(trades))

    # Clean types
    return trades, client_df, news_df


def engineer_features(df):
    """Compute additional features used by rules/ML."""
    out = df.copy()
    out["qty_z"] = out.groupby("symbol")["quantity"].transform(lambda s: zscore(winsorize(s)))
    out["pnl_uplift_z"] = out.groupby("symbol")["pnl_uplift"].transform(lambda s: zscore(winsorize(s)))
    out["mins_to_news"] = pd.to_numeric(out["mins_to_news"], errors="coerce")
    out["mins_to_news_filled"] = out["mins_to_news"].fillna(9999)
    out["align_and_overlap"] = out["directional_align"] * out["client_order_overlap"]
    # Feature set for ML
    feat_cols = [
        "quantity","qty_vs_avg","qty_z","bucket_vol","bucket_vol_z",
        "pnl_uplift","pnl_uplift_z","ret_fwd_15m_pct","directional_align",
        "client_order_overlap","mins_to_news_filled"
    ]
    X = out[feat_cols].fillna(0).astype(float)
    return out, X, feat_cols


def rules_engine(row, cfg):
    """Simple rules for IFA-like front-running signals."""
    flags = []

    if row["client_order_overlap"]==1 and row["qty_vs_avg"] >= cfg["qty_vs_avg_threshold"]:
        flags.append("Size spike near client order")

    if row["bucket_vol_z"] >= cfg["bucket_vol_z_threshold"]:
        flags.append("Abnormal bucket volume")

    if row["directional_align"]==1 and row["mins_to_news_filled"] <= cfg["mins_to_news_threshold"]:
        flags.append("Trade aligned with move near news")

    if row["pnl_uplift_z"] >= cfg["pnl_uplift_z_threshold"]:
        flags.append("Unusual PnL uplift")

    score = len(flags)
    return score, flags


def train_iforest(X, seed=DEFAULT_RANDOM_SEED):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=300,
        contamination=0.04,
        random_state=seed,
        n_jobs=-1,
        verbose=0
    )
    model.fit(Xs)
    return model, scaler


def score_model(model, scaler, X):
    Xs = scaler.transform(X)
    # IsolationForest: lower score -> more abnormal; convert to [0,1] risk
    raw = model.score_samples(Xs)
    # invert and normalize
    risk = (raw.min() - raw) / (raw.min() - raw.max() + 1e-9)
    return risk


def ai_agent_explain_and_actions(trade_row, rule_flags, ml_risk):
    """
    Lightweight "agent" to generate an explanation + remediation steps.
    (No external LLM needed; deterministic and demo-friendly.)
    """
    rationale_bits = []
    if "Size spike near client order" in rule_flags:
        rationale_bits.append("Trade size spiked within ±15 minutes of a client order in the same symbol.")
    if "Abnormal bucket volume" in rule_flags:
        rationale_bits.append("Order executed during a 5-minute window with unusually high volume.")
    if "Trade aligned with move near news" in rule_flags:
        rationale_bits.append("Directionally aligned with a price move and close to a news event.")
    if "Unusual PnL uplift" in rule_flags:
        rationale_bits.append("Outlier positive P&L uplift relative to symbol history.")

    if not rationale_bits:
        rationale_bits.append("Elevated anomaly score from the model across multiple features.")

    severity = "High" if ml_risk >= 0.85 or len(rule_flags) >= 3 else ("Medium" if ml_risk >= 0.65 or len(rule_flags) >= 2 else "Low")

    # Proposed actions (workflow)
    actions = [
        "Auto-create case in Surveillance System with all trade context and computed features.",
        "Check Restricted List & Watch List for the symbol; attach snapshot.",
        "Pull communications (email/chat/voice) ±1 day for trader & desk; keyword scan for MNPI or client order mentions.",
        "Cross-reference internal client orders: confirm overlap, principal vs agency capacity.",
        "Notify Desk Supervisor & Compliance Duty Officer; require same-day trader attestation on MNPI exposure.",
        "Run peer comparison across the desk for similar patterns in ±3 trading days.",
        "If confirmed risk: place account under heightened monitoring, restrict symbol, and document escalation outcome."
    ]

    memo = f"""
Compliance Case Memo (Auto-Draft)
Severity: {severity}
Trader: {trade_row.get('trader_id')}  |  Account: {trade_row.get('account_id')}  |  Desk: {trade_row.get('desk')}
Symbol: {trade_row.get('symbol')}  |  Side/Qty/Price: {trade_row.get('side')}/{trade_row.get('quantity')}/{trade_row.get('price')}
Timestamp (UTC): {trade_row.get('timestamp')}

Indicators:
- {'; '.join(rationale_bits)}

Model risk score (0-1): {ml_risk:.2f}

Initial Assessment:
This trade exhibits characteristics consistent with potential insider/front-running behavior (IFA). The combination of timing, size/volume abnormalities, directional alignment near news, and realized P&L uplift warrants review.

Next Actions:
- {actions[0]}
- {actions[1]}
- {actions[2]}
- {actions[3]}
- {actions[4]}
- {actions[5]}
- {actions[6]}

Prepared: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z
"""
    return severity, actions, memo


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("IFA Detector")
st.sidebar.caption(f"Prototype {APP_VERSION}")

st.sidebar.subheader("Rules Thresholds")
qty_vs_avg_threshold = st.sidebar.slider("Qty vs Avg (×)", 1.0, 10.0, 3.0, 0.5)
bucket_vol_z_threshold = st.sidebar.slider("Bucket Volume Z", 0.0, 10.0, 3.0, 0.5)
mins_to_news_threshold = st.sidebar.slider("Minutes to News (≤)", 5, 240, 30, 5)
pnl_uplift_z_threshold = st.sidebar.slider("PnL Uplift Z", 0.0, 10.0, 2.5, 0.5)

st.sidebar.subheader("ML Settings")
contamination = st.sidebar.slider("Contamination (IForest)", 0.01, 0.20, 0.04, 0.01)
est = st.sidebar.slider("Estimators", 100, 600, 300, 50)

# -----------------------------
# DATA LOAD
# -----------------------------
st.title("Trade Surveillance • IFA (Insider / Front-running Activity) Prototype")
st.write("Upload a trade blotter CSV, or use the synthetic demo data. The app runs rules + ML to flag potential IFA, and an AI agent drafts resolution steps & a compliance memo.")

uploaded = st.file_uploader("Upload blotter CSV (optional). Columns expected: timestamp, symbol, side, price, quantity, trader_id, account_id, desk", type=["csv"])
use_synth = st.toggle("Use synthetic data if no upload", value=True)

with st.expander("Optional: Sample schema & tips", expanded=False):
    st.markdown("""
**Minimum columns**:  
- `timestamp` (ISO or any parseable datetime)  
- `symbol` (e.g., AAPL)  
- `side` (BUY/SELL)  
- `price` (float)  
- `quantity` (int)  
- `trader_id`, `account_id`, `desk` (strings)

**Optional (improves signals)**:  
- `client_order_overlap` (0/1)  
- `pnl_est` (float, realized/estimated)  
- Any bucketed volumes, news proximity fields, etc.  
If missing, the app will engineer proxies.
""")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    # Basic coercions
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(_to_dt)
    else:
        st.error("Missing 'timestamp' column.")
        st.stop()
    req_cols = ["symbol","side","price","quantity","trader_id","account_id","desk"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    # Fill optional
    for c in ["client_order_overlap","pnl_est"]:
        if c not in df.columns:
            df[c] = 0
    client_df = pd.DataFrame(columns=["symbol","client_order_time","client_order_qty"])
    news_df = pd.DataFrame(columns=["symbol","news_time","news_type","impact"])
else:
    if use_synth:
        df, client_df, news_df = generate_synth_trades(n=1500)
    else:
        st.info("Please upload a CSV or toggle synthetic data.")
        st.stop()

df = df.sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
fe_df, X, feat_cols = engineer_features(df)

# -----------------------------
# RULES + ML
# -----------------------------
cfg = {
    "qty_vs_avg_threshold": qty_vs_avg_threshold,
    "bucket_vol_z_threshold": bucket_vol_z_threshold,
    "mins_to_news_threshold": mins_to_news_threshold,
    "pnl_uplift_z_threshold": pnl_uplift_z_threshold
}

# Train model (unsupervised, on full set for demo simplicity)
model, scaler = train_iforest(X.copy(), seed=DEFAULT_RANDOM_SEED)
# Allow updated contamination / estimators via re-train:
model.set_params(contamination=contamination, n_estimators=est)
model.fit(scaler.transform(X))

# Score
ml_risk = score_model(model, scaler, X)
fe_df["ml_risk_0_1"] = ml_risk

# Rules
rule_scores = []
rule_details = []
for _, r in fe_df.iterrows():
    s, flags = rules_engine(r, cfg)
    rule_scores.append(s)
    rule_details.append(flags)
fe_df["rule_score"] = rule_scores
fe_df["rule_flags"] = rule_details

# Combined risk
fe_df["combined_risk"] = 0.6*fe_df["ml_risk_0_1"] + 0.4*(fe_df["rule_score"]/4.0)
fe_df["alert"] = fe_df["combined_risk"] >= 0.65

# -----------------------------
# UI: OVERVIEW
# -----------------------------
st.subheader("Overview")
left, right = st.columns([2,1])
with left:
    fig = px.scatter(
        fe_df,
        x="timestamp", y="ml_risk_0_1",
        color=fe_df["alert"].map({True:"ALERT", False:"OK"}),
        hover_data=["symbol","side","quantity","price","trader_id","account_id","desk","rule_score"],
        title="Model Risk over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    total = len(fe_df)
    alerts = int(fe_df["alert"].sum())
    st.metric("Total Trades", f"{total:,}")
    st.metric("Alerts", f"{alerts:,}")
    st.metric("Alert Rate", f"{alerts/total*100:.1f}%")

# Symbol summary
sym_agg = fe_df.groupby("symbol").agg(
    trades=("trade_id","count"),
    alerts=("alert","sum"),
    avg_risk=("combined_risk","mean")
).reset_index().sort_values(["alerts","avg_risk"], ascending=[False, False])

st.markdown("### Symbols with most alerts")
st.dataframe(sym_agg, use_container_width=True, height=280)

# -----------------------------
# FILTERS
# -----------------------------
st.subheader("Investigate Alerts")
f1, f2, f3, f4 = st.columns(4)
sym_sel = f1.multiselect("Symbol", sorted(fe_df["symbol"].dropna().unique().tolist()))
trader_sel = f2.multiselect("Trader", sorted(fe_df["trader_id"].dropna().unique().tolist()))
desk_sel = f3.multiselect("Desk", sorted(fe_df["desk"].dropna().unique().tolist()))
risk_min = f4.slider("Min Combined Risk", 0.0, 1.0, 0.65, 0.05)

flt = fe_df.copy()
if sym_sel:
    flt = flt[flt["symbol"].isin(sym_sel)]
if trader_sel:
    flt = flt[flt["trader_id"].isin(trader_sel)]
if desk_sel:
    flt = flt[flt["desk"].isin(desk_sel)]
flt = flt[flt["combined_risk"] >= risk_min].sort_values("combined_risk", ascending=False)

st.dataframe(
    flt[[
        "timestamp","symbol","side","price","quantity","trader_id","account_id","desk",
        "rule_score","rule_flags","ml_risk_0_1","combined_risk","client_order_overlap","bucket_vol_z","qty_vs_avg","mins_to_news","pnl_uplift"
    ]].reset_index(drop=True),
    use_container_width=True,
    height=360
)

# -----------------------------
# CASE VIEW + "AI AGENT"
# -----------------------------
st.markdown("### Case View & AI Agent")
case_idx = st.number_input("Pick row index from the filtered table above (0-based)", min_value=0, max_value=max(0, len(flt)-1), value=0, step=1)

if len(flt) > 0:
    case_row = flt.iloc[int(case_idx)].to_dict()
    # Agent
    severity, actions, memo = ai_agent_explain_and_actions(case_row, case_row.get("rule_flags", []), case_row.get("combined_risk", 0.0))

    c1, c2 = st.columns([1.1,1])
    with c1:
        st.write("**Selected Trade**")
        st.json({
            "trade_id": case_row.get("trade_id"),
            "timestamp": str(case_row.get("timestamp")),
            "symbol": case_row.get("symbol"),
            "side": case_row.get("side"),
            "price": float(case_row.get("price")),
            "quantity": int(case_row.get("quantity")),
            "trader_id": case_row.get("trader_id"),
            "account_id": case_row.get("account_id"),
            "desk": case_row.get("desk"),
            "rule_flags": case_row.get("rule_flags"),
            "ml_risk_0_1": round(float(case_row.get("ml_risk_0_1")), 3),
            "combined_risk": round(float(case_row.get("combined_risk")), 3),
            "client_order_overlap": int(case_row.get("client_order_overlap")),
            "mins_to_news": None if pd.isna(case_row.get("mins_to_news")) else round(float(case_row.get("mins_to_news")), 1),
        })

        # Local feature importance proxy (z-scores & high values)
        local_feats = {k: float(case_row.get(k)) for k in ["qty_vs_avg","bucket_vol_z","pnl_uplift_z","directional_align","client_order_overlap","mins_to_news_filled"] if k in case_row}
        st.write("**Feature Snapshot**")
        st.json(local_feats)

    with c2:
        st.write(f"**AI Agent Verdict: {severity}**")
        st.write("**Proposed Actions**")
        for i, a in enumerate(actions, 1):
            st.write(f"{i}. {a}")
        st.download_button("Download Compliance Memo (.txt)", data=memo.encode("utf-8"), file_name="ifa_compliance_memo.txt")

    # Visualization around time bucket
    st.markdown("#### Volume Context")
    sym = case_row.get("symbol")
    t0 = pd.to_datetime(case_row.get("timestamp"))
    win = (fe_df[(fe_df["symbol"]==sym) & (fe_df["timestamp"].between(t0 - timedelta(hours=2), t0 + timedelta(hours=2)))]
           .copy()
           .sort_values("timestamp"))

    if not win.empty:
        fig2 = px.bar(win, x="timestamp", y="bucket_vol", color=win["timestamp"]==t0, title=f"{sym} 5-min Bucket Volume (±2h)")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(win, x="timestamp", y="ret_fwd_15m_pct", title=f"{sym} Forward 15m Return Proxy (±2h)")
        st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# EXPORTS
# -----------------------------
st.markdown("### Exports")
colA, colB = st.columns(2)
with colA:
    out_csv = fe_df.copy()
    out_csv["timestamp"] = out_csv["timestamp"].astype(str)
    st.download_button(
        "Download Scored Trades CSV",
        data=out_csv.to_csv(index=False).encode("utf-8"),
        file_name="ifa_scored_trades.csv",
        mime="text/csv"
    )
with colB:
    config = {
        "rules": cfg,
        "ml": {"contamination": contamination, "estimators": est},
        "version": APP_VERSION,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")+"Z"
    }
    st.download_button(
        "Download Run Config (JSON)",
        data=json.dumps(config, indent=2).encode("utf-8"),
        file_name="ifa_run_config.json",
        mime="application/json"
    )

st.caption("Demo only. Synthetic signals approximate trade-surveillance patterns for IFA (insider/front-running) scenarios.")
