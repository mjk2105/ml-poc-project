"""
Fixed Streamlit entry point for the project template.
Transfer Scout — Value-for-Money
Full Streamlit Application

Group 7: Megann Kouandjeu, Salma Kamoun, Axel Chartier
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths — resolved relative to this script ──────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / ".." / "datasets"
MODELS = BASE / ".." / "models"

# ── Feature set — identical to D03 ────────────────────────────────────────────
FEATURES_TREE = [
    "age", "age_squared", "prime_age_flag", "height",
    "goals_per_90", "assists_per_90", "goal_contributions_per_90",
    "minutes_per_game_proxy",
    "injury_risk_score", "is_injury_prone",
    "is_left_footed", "is_ambidextrous",
    "pos_ATT", "pos_DEF", "pos_MID", "pos_WNG",
]

POS_COLORS = {
    "GK": "#9b59b6", "DEF": "#3498db",
    "MID": "#2ecc71", "WNG": "#f39c12", "ATT": "#e74c3c",
}

# ── Page config (Called exactly ONCE at the absolute top level) ───────────────
st.set_page_config(
    page_title="Transfer Scout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Analytics Theme CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color:#0b0d13; color:#f1f3f9; }
  [data-testid="stSidebar"] { background-color:#131722; border-right: 1px solid #1f2434; }
  
  /* Modern translucent metric and data containers */
  .kpi-card {
    background: linear-gradient(135deg, #181c2b, #1e2336);
    border: 1px solid #282f48; 
    border-radius: 12px;
    padding: 18px; 
    text-align: center; 
    margin: 8px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
  .kpi-label { font-size:11px; color:#8e9aa8; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; }
  .kpi-value { font-size:26px; font-weight:700; color:#ffffff; margin:6px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
  .kpi-sub   { font-size:12px; color:#2ecc71; font-weight:500; }
  
  .player-card {
    background: linear-gradient(135deg, #181c2b, #22293f);
    border: 1px solid #2d3553; 
    border-radius: 14px;
    padding: 20px; 
    margin: 10px 0;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
  }
  .pname { font-size:18px; font-weight:700; color:#ffffff; letter-spacing:-0.3px; }
  .psub  { font-size:12px; color:#8e9aa8; margin-bottom:12px; }
  
  /* Clean dynamic labels */
  .tag { display:inline-block; padding:4px 12px; border-radius:20px;
         font-size:11px; font-weight:600; margin:3px; text-transform:uppercase; letter-spacing:0.5px; }
  .tag-buy   { background:rgba(46,204,113,0.15); color:#2ecc71; border:1px solid #2ecc71; }
  .tag-risk  { background:rgba(231,76,60,0.15); color:#e74c3c; border:1px solid #e74c3c; }
  .tag-prime { background:rgba(52,152,219,0.15); color:#3498db; border:1px solid #3498db; }
  .tag-left  { background:rgba(155,89,182,0.15); color:#9b59b6; border:1px solid #9b59b6; }
  
  .section-hd {
    font-size:22px; font-weight:700; color:#ffffff;
    border-left:5px solid #2ecc71; padding-left:14px;
    margin:26px 0 16px 0;
  }
  .subtitle  { font-size:15px; color:#8e9aa8; margin-bottom:22px; }
  .gap-pill  {
    display:inline-block; background:rgba(46,204,113,0.2);
    color:#2ecc71; border:1px solid #2ecc71; border-radius:20px;
    padding:4px 14px; font-weight:700; font-size:13px;
  }
  .divider   { border:none; border-top:1px solid #1f2434; margin:24px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    df = pd.read_csv(DATA / "df_engineered_outfield.csv").fillna(0)
    try:
        profiles = pd.read_csv(
            DATA / "player_profiles.csv",
            usecols=["player_id", "player_name", "citizenship"],
        )
        df = df.merge(profiles, on="player_id", how="left")
    except Exception:
        df["player_name"] = "Player #" + df["player_id"].astype(str)
        df["citizenship"] = ""

    df["player_name"] = df["player_name"].fillna("Player #" + df["player_id"].astype(str))
    df["citizenship"] = df["citizenship"].fillna("")
    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = MODELS / "xgb_model.json"
    if not path.exists():
        return None
    m = XGBRegressor()
    m.load_model(str(path))
    return m


@st.cache_data(show_spinner="Computing predictions…")
def compute_predictions(_df, _model):
    df = _df.copy()
    feats = [f for f in FEATURES_TREE if f in df.columns]
    df["predicted_value"] = np.expm1(_model.predict(df[feats]))
    df["valuation_gap"]   = df["predicted_value"] - df["market_value_eur"]
    df["roi_pct"]         = (
        df["valuation_gap"] / df["market_value_eur"].replace(0, np.nan) * 100
    ).fillna(0)
    return df


# ── Global Bootstrap Initialization ───────────────────────────────────────────
df_raw = load_data()
xgb    = load_model()

if xgb is None:
    st.error(
        "XGBoost model not found at `../models/xgb_model.json`.\n\n"
        "Run D03 first — it saves the model with `xgb_model.save_model('../models/xgb_model.json')`.",
        icon="🚨",
    )
    st.stop()

df = compute_predictions(df_raw, xgb)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def risk_label(score: float) -> tuple:
    if score < 0.25:  return "Low 🟢",    "#2ecc71"
    if score < 0.55:  return "Medium 🟡", "#f39c12"
    return                   "High 🔴",   "#e74c3c"


def find_clones(target_idx: int, n: int = 3) -> pd.DataFrame:
    feats  = [f for f in FEATURES_TREE if f in df.columns]
    X_norm = MinMaxScaler().fit_transform(df[feats])
    sims   = cosine_similarity(X_norm[target_idx].reshape(1, -1), X_norm)[0]

    t      = df.iloc[target_idx]
    mask   = (
        (df["market_value_eur"] < t["market_value_eur"]) &
        (df.index != target_idx) &
        (df["valuation_gap"] > 0) &
        (df["position_group"] == t["position_group"])
    )
    cands               = df[mask].copy()
    cands["similarity"] = (sims[cands.index] * 100).round(1)
    return cands.nlargest(n, "similarity")


def radar_fig(t, c):
    cats = ["G+A /90", "Min/Game", "Age Score", "Fitness", "Height"]
    def v(r):
        return [
            min(r["goal_contributions_per_90"] / 1.5, 1),
            r["minutes_per_game_proxy"] / 90,
            max(0, 1 - abs(r["age"] - 24) / 12),
            1 - r["injury_risk_score"],
            max(0, (r["height"] - 165) / 30) if r["height"] > 0 else 0.5,
        ]
    vt, vc = v(t), v(c)
    fig = go.Figure()
    for vals, name, col in [(vt, t["player_name"], "#e74c3c"),
                             (vc, c["player_name"], "#2ecc71")]:
        hex_ = col.lstrip("#")
        r2, g2, b2 = int(hex_[:2],16), int(hex_[2:4],16), int(hex_[4:],16)
        fill = f"rgba({r2},{g2},{b2},0.12)"
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", name=name, line_color=col, fillcolor=fill,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#8e9aa8", size=9), gridcolor="#1f2434"),
            angularaxis=dict(gridcolor="#1f2434", tickfont=dict(size=11))
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f3f9", size=11), showlegend=True,
        legend=dict(orientation="h", y=-0.25, x=0.0),
        margin=dict(t=30, b=50, l=30, r=30), height=290,
    )
    return fig


def gauge_fig(score: float):
    col = "#2ecc71" if score < 0.25 else "#f39c12" if score < 0.55 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100),
        number={"suffix":"%","font":{"color":"#ffffff","size":26}},
        gauge={
            "axis":  {"range":[0,100],"tickcolor":"#8e9aa8"},
            "bar":   {"color":col,"thickness":0.24},
            "bgcolor":"#181c2b",
            "steps": [{"range":[0,25],"color":"rgba(46,204,113,0.1)"},
                      {"range":[25,55],"color":"rgba(243,156,18,0.1)"},
                      {"range":[55,100],"color":"rgba(231,76,60,0.1)"}],
        },
        title={"text":"Injury Risk","font":{"color":"#8e9aa8","size":12}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=175,
                      margin=dict(t=30,b=0,l=10,r=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# THE STREAMLIT ENTRYPOINT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_app() -> None:
    """Wrapped application engine executing navigation rendering layout routines."""
    n_total = len(df)
    n_under = int((df["valuation_gap"] > 0).sum())

    # ── Sidebar Internal Routing Options ──────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚽ Transfer Scout")
        st.markdown("<div style='color:#8e9aa8;font-size:13px;margin-bottom:16px;'>"
                    "Value-for-Money · Group 7</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1f2434;'>", unsafe_allow_html=True)
        page = st.radio("Navigation", [
            "🏠  Overview",
            "🔍  Budget Clone Finder",
            "💼  Scout Search",
            "📊  Model Story",
        ], label_visibility="collapsed")
        st.markdown("<hr style='border-color:#1f2434;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:11px;color:#8e9aa8;line-height:2;'>
        <b style='color:#ffffff;'>Dataset</b><br>
        {n_total:,} outfield players<br>
        <span style='color:#2ecc71;'>{n_under:,} undervalued</span><br><br>
        <b style='color:#ffffff;'>Model</b><br>XGBoost (D03)<br>
        RMSLE · MAE · R²
        </div>""", unsafe_allow_html=True)

    # ── PAGE 1 — OVERVIEW ─────────────────────────────────────────────────────
    if page == "🏠  Overview":
        st.markdown("# ⚽ The Value-for-Money Transfer Scout")
        st.markdown("<div class='subtitle'>An ML-powered scouting tool that identifies undervalued "
                    "football players for mid-tier clubs with limited transfer budgets.</div>",
                    unsafe_allow_html=True)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("""
            <div class='section-hd'>The Problem</div>
            <p style='color:#c8cdd8;line-height:1.7;'>
            Mid-tier clubs cannot afford superstars yet need top-tier performance to stay competitive.
            The transfer market is driven by hype and agent fees — not objective performance data.
            </p>
            <div class='section-hd'>The Solution</div>
            <p style='color:#c8cdd8;line-height:1.7;'>
            We train an <b>XGBoost regression model</b> on Transfermarkt data to predict every
            player's <i>Fair Market Value</i> from pure statistics. When the model predicts
            <b>€20M</b> but the asking price is <b>€5M</b>, that
            <b style='color:#2ecc71;'>€15M gap is your scouting opportunity</b>.
            </p>""", unsafe_allow_html=True)

            st.markdown("<div class='section-hd'>Three Tools in One App</div>", unsafe_allow_html=True)
            for icon, title, desc in [
                ("🔍", "Budget Clone Finder", "Select any world-class player — get 3 statistically similar alternatives at a fraction of the cost."),
                ("💼", "Scout Search", "Set your budget and target position — get the top undervalued gems on the market."),
                ("📊", "Model Story", "Understand which player attributes drive value and how the model was built."),
            ]:
                st.markdown(f"""
                <div class='player-card' style='padding:14px 18px;'>
                  <span style='font-size:20px;'>{icon}</span>
                  <b style='font-size:15px;margin-left:8px;color:#ffffff;'>{title}</b>
                  <div style='color:#8e9aa8;font-size:12.5px;margin-top:6px;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown("<div class='section-hd'>Live Dataset Stats</div>", unsafe_allow_html=True)
            median_val = df["market_value_eur"].median() / 1e6
            max_gap    = df["valuation_gap"].max() / 1e6
            avg_gap    = df.loc[df["valuation_gap"] > 0, "valuation_gap"].mean() / 1e6
            fi_dict    = dict(zip([f for f in FEATURES_TREE if f in df.columns], xgb.feature_importances_))
            top_feat   = max(fi_dict, key=fi_dict.get).replace("_", " ") if fi_dict else "N/A"

            for label, val, sub in [
                ("Outfield Players",       f"{n_total:,}",        None),
                ("Undervalued Players",    f"{n_under:,}",        f"{n_under/n_total*100:.0f}% of dataset"),
                ("Median Market Value",    f"€{median_val:.2f}M", None),
                ("Biggest Valuation Gap",   f"€{max_gap:.1f}M",     "Best buy signal"),
                ("Avg Gap (undervalued)",   f"€{avg_gap:.1f}M",     None),
                ("Top Predictive Feature", top_feat,              None),
            ]:
                sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
                st.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{label}</div>
                  <div class='kpi-value'>{val}</div>
                  {sub_html}
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-hd'>Top 10 Biggest Valuation Gaps — Live</div>", unsafe_allow_html=True)

        top10 = df.nlargest(10, "valuation_gap").copy()
        top10["label"] = top10["player_name"] + " (" + top10["position_group"] + ")"
        top10["price_M"] = (top10["market_value_eur"] / 1e6).round(2)
        top10["pred_M"]  = (top10["predicted_value"]  / 1e6).round(2)

        fig_t10 = go.Figure()
        fig_t10.add_trace(go.Bar(
            name="Asking Price (€M)", y=top10["label"], x=top10["price_M"],
            orientation="h", marker_color="#e74c3c",
        ))
        fig_t10.add_trace(go.Bar(
            name="Predicted Value (€M)", y=top10["label"], x=top10["pred_M"],
            orientation="h", marker_color="#2ecc71", opacity=0.85,
        ))
        fig_t10.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f3f9"), legend=dict(orientation="h", y=1.15, x=0.0),
            xaxis=dict(title="€ Millions", gridcolor="#1f2434"),
            yaxis=dict(gridcolor="#1f2434", autorange="reversed", automargin=True),
            height=400, margin=dict(t=60, b=40, l=150, r=20),
        )
        st.plotly_chart(fig_t10, use_container_width=True)

    # ── PAGE 2 — BUDGET CLONE FINDER ──────────────────────────────────────────
    elif page == "🔍  Budget Clone Finder":
        st.markdown("# 🔍 Budget Clone Finder")
        st.markdown("<div class='subtitle'>Select a world-class player. The model finds 3 statistically "
                    "similar players your club can actually afford.</div>", unsafe_allow_html=True)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            pos_sel = st.selectbox("Position group", ["ATT", "MID", "WNG", "DEF"])
        with c2:
            pool = df[df["position_group"] == pos_sel].nlargest(60, "market_value_eur")
            chosen = st.selectbox(f"Select a {pos_sel} player (top 60 by market value)", pool["player_name"].tolist())

        if chosen not in df["player_name"].values:
            st.warning("Player not found.")
            st.stop()

        t_idx = df.index[df["player_name"] == chosen][0]
        t     = df.iloc[t_idx]

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-hd'>Target Player</div>", unsafe_allow_html=True)

        rl, _ = risk_label(t["injury_risk_score"])
        for col, (lbl, val) in zip(st.columns(5), [
            ("Market Price",   f"€{t['market_value_eur']/1e6:.1f}M"),
            ("Predicted Value",f"€{t['predicted_value']/1e6:.1f}M"),
            ("G+A per 90",     f"{t['goal_contributions_per_90']:.2f}"),
            ("Min / Game",     f"{t['minutes_per_game_proxy']:.0f}'"),
            ("Injury Risk",    rl),
        ]):
            with col:
                st.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value'>{val}</div>
                </div>""", unsafe_allow_html=True)

        with st.spinner("Finding budget alternatives…"):
            clones = find_clones(t_idx, n=3)

        if clones.empty:
            st.warning("No undervalued alternatives found in the same position at a lower price.")
            st.stop()

        st.markdown("<div class='section-hd'>3 Budget Alternatives</div>", unsafe_allow_html=True)

        for rank, (_, c) in enumerate(clones.iterrows()):
            gap_m   = c["valuation_gap"] / 1e6
            savings = (t["market_value_eur"] - c["market_value_eur"]) / 1e6
            sim     = c.get("similarity", 0.0)
            prime   = 21 <= c["age"] <= 27

            with st.expander(f"#{rank+1}  {c['player_name']}  ·  {c['position_clean']}  ·  Local Cost: €{c['market_value_eur']/1e6:.1f}M  ·  {sim:.0f}% Match", expanded=(rank == 0)):
                left, mid, right = st.columns([1.8, 2.2, 1.4])

                with left:
                    tags  = '<span class="tag tag-buy">💰 BUY</span>'
                    tags += '<span class="tag tag-prime">🌟 Prime Age</span>' if prime else ""
                    tags += '<span class="tag tag-risk">⚠ High Risk</span>' if c["injury_risk_score"] > 0.35 else ""
                    tags += '<span class="tag tag-left">🦶 Left Foot</span>' if c["is_left_footed"] else ""
                    nat = f" · {c['citizenship']}" if c.get("citizenship") else ""
                    st.markdown(f"""
                    <div class='player-card'>
                      <div class='pname'>{c['player_name']}</div>
                      <div class='psub'>{c['position_clean']} · Age {c['age']:.0f}{nat}</div>
                      <div style='margin:8px 0;'>{tags}</div>
                      <table style='width:100%;font-size:13px;border-collapse:collapse;color:#f1f3f9;'>
                        <tr><td style='color:#8e9aa8;padding:4px 0;'>Market Price</td><td style='text-align:right;'><b>€{c['market_value_eur']/1e6:.1f}M</b></td></tr>
                        <tr><td style='color:#8e9aa8;padding:4px 0;'>Predicted Value</td><td style='text-align:right;'><b style='color:#2ecc71;'>€{c['predicted_value']/1e6:.1f}M</b></td></tr>
                        <tr><td style='color:#8e9aa8;padding:4px 0;'>Valuation Gap</td><td style='text-align:right;'><span class='gap-pill'>+€{gap_m:.1f}M</span></td></tr>
                        <tr><td style='color:#8e9aa8;padding:4px 0;'>Savings vs Target</td><td style='text-align:right;'><b>€{savings:.1f}M</b></td></tr>
                        <tr><td style='color:#8e9aa8;padding:4px 0;'>G+A per 90</td><td style='text-align:right;'><b>{c['goal_contributions_per_90']:.2f}</b></td></tr>
                        <tr><td style='color:#8e9aa8;padding:4px 0;'>Min / Game</td><td style='text-align:right;'><b>{c['minutes_per_game_proxy']:.0f}'</b></td></tr>
                        <tr><td style='color:#8e9aa8;padding:4px 0;'>Days Injured</td><td style='text-align:right;'><b>{c['days_missed']:.0f} days</b></td></tr>
                      </table>
                    </div>""", unsafe_allow_html=True)

                with mid:
                    st.plotly_chart(radar_fig(t, c), use_container_width=True, key=f"radar_{rank}")

                with right:
                    st.plotly_chart(gauge_fig(c["injury_risk_score"]), use_container_width=True, key=f"gauge_{rank}")
                    st.markdown(f"""
                    <div style='text-align:center;margin-top:6px;'>
                      <div style='color:#8e9aa8;font-size:11px;'>Days Injured</div>
                      <div style='font-size:20px;font-weight:700;color:#ffffff;'>{c['days_missed']:.0f}</div>
                      <div style='color:#8e9aa8;font-size:11px;margin-top:4px;'>Risk Score</div>
                      <div style='font-size:20px;font-weight:700;color:#ffffff;'>{c['injury_risk_score']:.2f}</div>
                    </div>""", unsafe_allow_html=True)

# ── PAGE 3 — SCOUT SEARCH (Truncation Fixed) ──────────────────────────────
    elif page == "💼  Scout Search":
        st.markdown("# 💼 Scout Search")
        st.markdown("<div class='subtitle'>Set your budget and target position. The model returns the most undervalued players.</div>", unsafe_allow_html=True)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            pos_sel   = st.selectbox("Position", ["ATT", "MID", "WNG", "DEF"])
        with f2:
            budget_m  = st.slider("Max Budget (€M)", 1, 50, 10)
        with f3:
            min_mpg   = st.slider("Min Minutes / Game", 20, 85, 50, step=5)
        with f4:
            max_risk  = st.slider("Max Injury Risk", 0.10, 1.0, 0.50, step=0.05)

        top_n = st.slider("Number of results", 3, 10, 5)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        results = df[
            (df["position_group"]          == pos_sel) &
            (df["market_value_eur"]        <= budget_m * 1_000_000) &
            (df["minutes_per_game_proxy"]  >= min_mpg) &
            (df["injury_risk_score"]       <= max_risk) &
            (df["valuation_gap"]           > 0)
        ].nlargest(top_n, "valuation_gap").copy()

        if results.empty:
            st.warning("No undervalued players match your filters.")
            st.stop()

        st.markdown(f"<div class='section-hd'>{len(results)} Result(s) — {pos_sel} · Budget ≤ €{budget_m}M</div>", unsafe_allow_html=True)

        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(name="Asking Price (€M)", y=results["player_name"].tolist(), x=(results["market_value_eur"] / 1e6).tolist(), orientation="h", marker_color="#e74c3c"))
        fig_s.add_trace(go.Bar(name="Predicted Value (€M)", y=results["player_name"].tolist(), x=(results["predicted_value"] / 1e6).tolist(), orientation="h", marker_color="#2ecc71", opacity=0.8))
        fig_s.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f3f9"), legend=dict(orientation="h", y=1.15, x=0.0),
            xaxis=dict(title="€ Millions", gridcolor="#1f2434"),
            yaxis=dict(gridcolor="#1f2434", autorange="reversed", automargin=True),
            height=max(260, len(results) * 60), margin=dict(t=60, b=40, l=150, r=20),
        )
        st.plotly_chart(fig_s, use_container_width=True)

        medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
        for rank, (_, r) in enumerate(results.iterrows()):
            gap_m = r["valuation_gap"] / 1e6
            prime = 21 <= r["age"] <= 27

            # Helper helper function to cleanly format sub-million values to prevent layout text truncation
            def format_currency(val_eur):
                if val_eur >= 1_000_000:
                    return f"€{val_eur / 1_000_000:.1f}M"
                elif val_eur >= 1_000:
                    return f"€{int(val_eur / 1_000)}K"
                return f"€{int(val_eur)}"

            # Optimized layout distribution grid with wider columns for metrics to stop text clipping
            # Expanded c6 (ROI) and metrics while shaving excess room off the player profile text block (c1)
            c0, c1, c2, c3, c4, c5, c6 = st.columns([0.4, 1.9, 1.3, 1.7, 1.2, 1.2, 1.7])
            with c0:
                st.markdown(f"<div style='font-size:24px;text-align:center;padding-top:14px;'>{medals[rank]}</div>", unsafe_allow_html=True)
            with c1:
                tags  = '<span class="tag tag-buy">💰 BUY</span>'
                tags += '<span class="tag tag-prime">🌟 Prime</span>' if prime else ""
                tags += '<span class="tag tag-risk">⚠ Risk</span>' if r["injury_risk_score"] > 0.35 else ""
                nat = f" · {r['citizenship']}" if r.get("citizenship") else ""
                st.markdown(f"""
                <div class='player-card' style='padding:10px 14px;'>
                  <div class='pname' style='font-size:14px;'>{r['player_name']}</div>
                  <div class='psub'>{r['position_clean']} · Age {r['age']:.0f}{nat}</div>
                  <div>{tags}</div>
                </div>""", unsafe_allow_html=True)
            
            with c2: 
                st.metric("Price", format_currency(r['market_value_eur']))
            with c3: 
                delta_str = f"+{format_currency(r['valuation_gap'])}"
                st.metric("Predicted", format_currency(r['predicted_value']), delta=delta_str)
            with c4: 
                st.metric("G+A /90", f"{r['goal_contributions_per_90']:.2f}")
            with c5: 
                st.metric("Min/Game", f"{r['minutes_per_game_proxy']:.0f}'")
            with c6: 
                st.metric("ROI %", f"+{r['roi_pct']:.0f}%")

            st.markdown("<hr class='divider' style='margin:4px 0;'>", unsafe_allow_html=True)

    # ── PAGE 4 — MODEL STORY ──────────────────────────────────────────────────
    elif page == "📊  Model Story":
        st.markdown("# 📊 Model Story")
        st.markdown("<div class='subtitle'>How the XGBoost model was built, what it learned, and how confident we are in its predictions.</div>", unsafe_allow_html=True)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        st.markdown("<div class='section-hd'>Model Comparison — D03 Results</div>", unsafe_allow_html=True)
        for col, (name, color, role, rmsle, mae, r2) in zip(st.columns(3), [
            ("Ridge Regression", "#3498db", "Baseline",   1.35, 1.76, 0.28),
            ("Random Forest",    "#e67e22", "Benchmark",  1.32, 1.73, 0.31),
            ("XGBoost",          "#2ecc71", "⭐ Primary",  1.31, 1.71, 0.32),
        ]):
            with col:
                border = f"border:2px solid {color};" if name == "XGBoost" else ""
                st.markdown(f"""
                <div class='player-card' style='{border}'>
                  <div class='pname' style='color:{color};'>{name}</div>
                  <div class='psub'>{role}</div>
                  <table style='width:100%;font-size:13px;border-collapse:collapse;color:#f1f3f9;'>
                    <tr><td style='color:#8e9aa8;padding:5px 0;'>RMSLE ↓</td><td style='text-align:right;'><b>{rmsle}</b></td></tr>
                    <tr><td style='color:#8e9aa8;padding:5px 0;'>MAE (€M) ↓</td><td style='text-align:right;'><b>€{mae}M</b></td></tr>
                    <tr><td style='color:#8e9aa8;padding:5px 0;'>R² ↑</td><td style='text-align:right;'><b>{r2}</b></td></tr>
                  </table>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("<div class='section-hd'>Feature Importance (Live from model)</div>", unsafe_allow_html=True)
            feats       = [f for f in FEATURES_TREE if f in df.columns]
            importances = pd.Series(xgb.feature_importances_[:len(feats)], index=[f.replace("_", " ") for f in feats]).sort_values(ascending=True)
            fi_colors   = ["#2ecc71" if v >= importances.quantile(0.66) else "#e67e22" if v >= importances.quantile(0.33) else "#e74c3c" for v in importances.values]

            fig_fi = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h", marker_color=fi_colors))
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f3f9", size=11),
                xaxis=dict(title="Feature importance (gain)", gridcolor="#1f2434"),
                yaxis=dict(gridcolor="#1f2434", automargin=True),
                height=420, margin=dict(t=20, b=40, l=150, r=20),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        with right_col:
            st.markdown("<div class='section-hd'>Market Value Distribution by Position</div>", unsafe_allow_html=True)
            cap = df["market_value_eur"].quantile(0.95)
            fig_viol = go.Figure()
            for pos in ["DEF", "MID", "WNG", "ATT"]:
                vals = df.loc[df["position_group"] == pos, "market_value_eur"].clip(upper=cap) / 1e6
                fig_viol.add_trace(go.Violin(y=vals, name=pos, box_visible=True, line_color=POS_COLORS[pos], fillcolor=POS_COLORS[pos], opacity=0.4, x0=pos))
            fig_viol.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f3f9"), showlegend=False,
                yaxis=dict(title="Market Value (€M, capped 95th pct)", gridcolor="#1f2434"),
                xaxis=dict(gridcolor="#1f2434"), height=420, margin=dict(t=10, b=20, l=10, r=20),
            )
            st.plotly_chart(fig_viol, use_container_width=True)


# ── Launcher Execution Check ─────────────────────────────────────────────────
if __name__ == "__main__":
    build_app()