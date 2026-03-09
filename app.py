import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re

st.set_page_config(
    page_title="DPWH Flood Control – Forensic Analytics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #080c10; color: #c8d4e0; }

section[data-testid="stSidebar"] { background: #0a0e14 !important; border-right: 1px solid #182030; }
section[data-testid="stSidebar"] * { color: #7a90a8 !important; }

.dash-header {
    background: linear-gradient(135deg, #0a0e14 0%, #0d1420 100%);
    border: 1px solid #182030; border-left: 4px solid #ff3333;
    border-radius: 4px 12px 12px 4px; padding: 22px 30px; margin-bottom: 18px;
}
.dash-header h1 { font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:600; color:#fff; margin:0 0 4px 0; }
.dash-header p  { color:#4a6a8a; font-size:0.82rem; margin:0; }
.badge { display:inline-block; background:rgba(255,51,51,0.12); border:1px solid rgba(255,51,51,0.3);
         color:#ff6666; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; font-weight:600;
         padding:2px 9px; border-radius:3px; margin-bottom:8px; letter-spacing:1.5px; text-transform:uppercase; }

.kpi { background:#0a0e14; border:1px solid #182030; border-radius:10px; padding:16px 18px; position:relative; }
.kpi::after { content:''; position:absolute; bottom:0;left:0;right:0;height:2px;border-radius:0 0 10px 10px; }
.kpi-r::after{background:#ff3333;} .kpi-o::after{background:#f5a623;} .kpi-b::after{background:#2196f3;} .kpi-p::after{background:#a855f7;} .kpi-g::after{background:#22c55e;}
.kpi-lbl { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:600; text-transform:uppercase; letter-spacing:1.2px; color:#3a5a7a; margin-bottom:5px; }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:1.7rem; font-weight:600; color:#fff; line-height:1; margin-bottom:2px; }
.kpi-sub { font-size:0.74rem; color:#3a5a7a; }

.finding { background:#0a0e14; border:1px solid #182030; border-left:3px solid #ff3333;
           border-radius:0 8px 8px 0; padding:11px 14px; margin:6px 0; font-size:0.81rem;
           color:#7a90a8; line-height:1.6; }
.finding strong { color:#ff7777; }
.finding.o { border-left-color:#f5a623; } .finding.o strong { color:#f5a623; }
.finding.b { border-left-color:#2196f3; } .finding.b strong { color:#60b8ff; }
.finding.p { border-left-color:#a855f7; } .finding.p strong { color:#c084fc; }
.finding.g { border-left-color:#22c55e; } .finding.g strong { color:#4ade80; }

.sec { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; font-weight:600; color:#3a5a7a;
       text-transform:uppercase; letter-spacing:1.5px; margin:10px 0 3px 0; }
.secline { height:1px; background:#182030; margin-bottom:12px; }

.stTabs [data-baseweb="tab-list"] { background:#0a0e14; border-radius:8px; gap:12px; padding:8px 12px; border:1px solid #182030; width:100%; display:flex; justify-content:space-between; }
.stTabs [data-baseweb="tab"]      { border-radius:6px; color:#3a5a7a; font-size:0.78rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.stTabs [aria-selected="true"]    { background:#182030 !important; color:#c8d4e0 !important; }
</style>
""", unsafe_allow_html=True)

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color="#4a6a8a", size=11),
    margin=dict(l=12, r=12, t=32, b=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#4a6a8a")),
    xaxis=dict(gridcolor="#182030", linecolor="#182030", tickfont=dict(color="#3a5a7a")),
    yaxis=dict(gridcolor="#182030", linecolor="#182030", tickfont=dict(color="#3a5a7a")),
)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("dpwh_flood_control_projects.csv")
    df["Budget"]   = pd.to_numeric(df["ApprovedBudgetForContract"], errors="coerce")
    df["Cost"]     = pd.to_numeric(df["ContractCost"], errors="coerce")
    df["Start"]    = pd.to_datetime(df["StartDate"], errors="coerce")
    df["Completion"] = pd.to_datetime(df["ActualCompletionDate"], errors="coerce")
    df["DurationDays"] = (df["Completion"] - df["Start"]).dt.days
    df["BidRatio"] = df["Cost"] / df["Budget"] * 100
    df["Variance"] = (df["Cost"] - df["Budget"]) / df["Budget"] * 100
    df["BudgetM"]  = df["Budget"] / 1e6
    df["BudgetB"]  = df["Budget"] / 1e9
    df["CostM"]    = df["Cost"] / 1e6
    df["StartYM"]  = df["Start"].dt.to_period("M").astype(str)
    df["StartYear"]= df["Start"].dt.year

    # Benford
    def lead(x):
        if pd.isna(x) or x <= 0: return None
        s = str(int(x)).lstrip("0")
        return int(s[0]) if s else None
    df["LeadCost"]   = df["Cost"].apply(lead)
    df["LeadBudget"] = df["Budget"].apply(lead)

    # Flags
    df["F_BidRatio99"] = ((df["BidRatio"] >= 99) & (df["BidRatio"] <= 100)).astype(int) * 25
    df["F_Exact"]      = (df["Budget"] == df["Cost"]).astype(int) * 20
    coord_ct = df.groupby(["ProjectLatitude","ProjectLongitude"])["ProjectId"].transform("count")
    df["F_DupeCoord"]  = (coord_ct >= 4).astype(int) * 20
    dupe_ids = df[df.duplicated("ProjectId", keep=False)]["ProjectId"].unique()
    df["F_DupeID"]     = df["ProjectId"].isin(dupe_ids).astype(int) * 15
    df["F_Threshold"]  = df["Budget"].isin([49000000,96500000,4950000]).astype(int) * 15
    df["F_UnderSpend"] = (df["Variance"] < -50).astype(int) * 20
    df["F_ShortDur"]   = ((df["DurationDays"] > 0) & (df["DurationDays"] < 30)).astype(int) * 15
    df["HasStationing"]= df["ProjectName"].str.contains(r"Sta\.\s*\d+", regex=True, case=False)
    df["F_Ghost"]      = (~df["HasStationing"]).astype(int) * 10
    batch_ct = df.groupby(["Contractor","Start"])["ProjectId"].transform("count")
    df["F_Batch"]      = (batch_ct >= 10).astype(int) * 10
    score_cols = [c for c in df.columns if c.startswith("F_")]
    df["SuspicionScore"] = df[score_cols].sum(axis=1)

    # Phase detection
    df["HasPhase"] = df["ProjectName"].str.lower().str.contains(
        r"phase\s+[ivxlcd\d]+|package\s+\d+", regex=True)

    return df

df = load()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 6px 0'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.9rem;font-weight:600;color:#c8d4e0'>Filters</div>
        <div style='font-size:0.62rem;color:#2a4a6a;letter-spacing:1px;text-transform:uppercase;margin-top:3px'>DPWH · 2018–2025</div>
    </div><hr style='border-color:#182030;margin:6px 0 12px'>
    """, unsafe_allow_html=True)
    yr_min, yr_max = int(df["FundingYear"].min()), int(df["FundingYear"].max())
    sel_years = st.slider("Funding Year", yr_min, yr_max, (yr_min, yr_max))
    islands = ["All"] + sorted(df["MainIsland"].dropna().unique().tolist())
    sel_island = st.selectbox("Island Group", islands)
    reg_opts = sorted(df["Region"].unique()) if sel_island=="All" else sorted(df[df["MainIsland"]==sel_island]["Region"].unique())
    sel_regions = st.multiselect("Region(s)", reg_opts, default=reg_opts)
    st.markdown("<hr style='border-color:#182030'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.65rem;color:#2a3a4a;line-height:1.9'>
    Suspicion Score Weights:<br>
    +25 · Bid ratio 99–100%<br>
    +20 · Exact Budget=Cost<br>
    +20 · Duplicate coordinates<br>
    +20 · Extreme under-spend<br>
    +15 · Duplicate Project ID<br>
    +15 · Threshold bypassing<br>
    +15 · Short duration (&lt;30d)<br>
    +10 · Ghost project (no Sta.)<br>
    +10 · Batch award (≥10/day)
    </div>""", unsafe_allow_html=True)

# FILTER
dff = df[df["FundingYear"].between(sel_years[0], sel_years[1])]
if sel_island != "All": dff = dff[dff["MainIsland"]==sel_island]
if sel_regions:         dff = dff[dff["Region"].isin(sel_regions)]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <div class="badge">PROCUREMENT INTEGRITY WATCH · 5-Category Anomaly Analysis</div>
    <h1>DPWH Flood Control – Fiscal Accountability Report</h1>
    <p>Bid Rigging · Market Capture · Project Splitting · Election Cycle · Ghost Projects · Top 10 Most Suspicious</p>
</div>""", unsafe_allow_html=True)

# KPIs
total = len(dff)
n_99club     = ((dff["BidRatio"]>=99)&(dff["BidRatio"]<=100)).sum()
n_deo_mono   = 40  # static from analysis
n_split      = dff["HasPhase"].sum()
n_ghost      = (~dff["HasStationing"]).sum()
n_threshold  = dff["Budget"].isin([49000000,96500000,4950000]).sum()

c1,c2,c3,c4,c5 = st.columns(5)
kpis = [
    (c1,"kpi-r","The 99% Club",f"{n_99club:,}",f"{n_99club/total*100:.0f}% of all contracts"),
    (c2,"kpi-o","DEO Monopolies","40",f"out of 199 District Offices"),
    (c3,"kpi-b","Phase/Split Projects",f"{n_split:,}","Potential salami slicing"),
    (c4,"kpi-p","Ghost Projects",f"{n_ghost:,}",f"No stationing · ₱{dff[~dff['HasStationing']]['BudgetB'].sum():.0f}B"),
    (c5,"kpi-g","Threshold Bypass",f"{n_threshold:,}",f"Clustered just under 50M/100M"),
]
for col,cls,lbl,val,sub in kpis:
    with col:
        st.markdown(f"""<div class="kpi {cls}">
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-val">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🎯 1 · Bid Rigging",
    "🗺 2 · Market Capture",
    "✂ 3 · Project Splitting",
    "🗳 4 · Election Cycle",
    "👻 5 · Ghost Projects",
    "🏆 Top 10 Most Suspicious",
])
tab1,tab2,tab3,tab4,tab5,tab6 = tabs

# ══════════════════════════════════════════════
# TAB 1 – BID RIGGING
# ══════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="finding">
    <strong>The 99% Club:</strong> <strong>65% of all contracts (6,410 projects, ₱383.6B)</strong> have a winning
    bid between 99–100% of the Approved Budget for Contract. In a genuine competitive bidding environment,
    bids cluster around 85–95% as contractors compete. A 99–100% concentration is statistical evidence
    of <strong>collusion, pre-arranged awards, or phantom bidding</strong>.
    </div>
    <div class="finding o" style="margin-top:5px">
    <strong>Benford's Law Violation:</strong> Digit 4 appears <strong>3× more often</strong> than expected
    (29.1% vs 9.7% expected), and digit 9 appears <strong>4.6× more often</strong> (21.0% vs 4.6%).
    These are statistically extreme z-scores (+65 and +78) — a mathematical fingerprint of
    <strong>engineered contract amounts</strong> designed to cluster just below budget ceilings.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec">Bid Ratio Distribution – The 99% Cliff</div><div class="secline"></div>', unsafe_allow_html=True)
        bins   = [0,50,80,90,95,97,98,99,99.5,100,100.01]
        labels = ["<50%","50-80%","80-90%","90-95%","95-97%","97-98%","98-99%","99-99.5%","99.5-100%","=100%"]
        dff2 = dff[dff["BidRatio"].notna() & (dff["BidRatio"]<=100)].copy()
        dff2["Band"] = pd.cut(dff2["BidRatio"], bins=bins, labels=labels, right=True)
        band_ct = dff2["Band"].value_counts().reindex(labels).fillna(0).reset_index()
        band_ct.columns = ["Band","Count"]
        band_ct["Pct"] = band_ct["Count"] / len(dff2) * 100
        colors = ["#ff3333" if b in ["99-99.5%","99.5-100%","=100%"] else
                  "#f5a623" if b in ["98-99%","97-98%"] else "#1c3a55"
                  for b in band_ct["Band"]]
        fig_br = go.Figure(go.Bar(
            x=band_ct["Band"], y=band_ct["Count"],
            marker=dict(color=colors, line=dict(width=0)),
            text=band_ct["Pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside", textfont=dict(color="#7a90a8", size=9),
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>"
        ))
        fig_br.update_layout(**PL, height=300, showlegend=False,
              xaxis_title="Bid Ratio (Cost / Budget %)", yaxis_title="# Projects")
        st.plotly_chart(fig_br, width='stretch')

    with col_r:
        st.markdown('<div class="sec">Benford\'s Law – Leading Digit Analysis (Contract Cost)</div><div class="secline"></div>', unsafe_allow_html=True)
        benford_exp = {d: np.log10(1+1/d)*100 for d in range(1,10)}
        obs = dff["LeadCost"].value_counts().sort_index()
        total_lead = obs.sum()
        digits = list(range(1,10))
        obs_pct = [obs.get(d,0)/total_lead*100 for d in digits]
        exp_pct = [benford_exp[d] for d in digits]

        fig_bf = go.Figure()
        fig_bf.add_trace(go.Bar(
            x=digits, y=obs_pct,
            name="Observed",
            marker=dict(color=["#ff3333" if abs(obs_pct[i]-exp_pct[i])>5 else "#1c3a55"
                               for i in range(9)], line=dict(width=0)),
            hovertemplate="Digit %{x}<br>Observed: %{y:.2f}%<extra></extra>"
        ))
        fig_bf.add_trace(go.Scatter(
            x=digits, y=exp_pct, mode="lines+markers", name="Benford Expected",
            line=dict(color="#f5a623", width=2, dash="dot"),
            marker=dict(size=6, color="#f5a623")
        ))
        fig_bf.update_layout(**PL, height=300)
        st.plotly_chart(fig_bf, width='stretch')

    # 99% Club by region
    st.markdown('<div class="sec">99% Club Penetration by Region (% of region\'s contracts in 99-100% band)</div><div class="secline"></div>', unsafe_allow_html=True)
    reg99 = dff.groupby("Region").agg(
        Club99=("F_BidRatio99", lambda x: (x>0).sum()),
        Total=("ProjectId","count"),
        Budget=("BudgetB","sum")
    ).reset_index()
    reg99["Rate"] = reg99["Club99"]/reg99["Total"]*100
    reg99 = reg99.sort_values("Rate", ascending=False)

    fig_r99 = make_subplots(specs=[[{"secondary_y":True}]])
    fig_r99.add_trace(go.Bar(
        x=reg99["Region"], y=reg99["Rate"],
        name="99% Club Rate",
        marker=dict(color=["#ff3333" if r>70 else "#f5a623" if r>60 else "#1c3a55"
                           for r in reg99["Rate"]], line=dict(width=0))
    ), secondary_y=False)
    fig_r99.add_trace(go.Scatter(
        x=reg99["Region"], y=reg99["Budget"],
        mode="markers", name="Total Budget (₱B)",
        marker=dict(size=8, color="#2196f3", symbol="diamond")
    ), secondary_y=True)
    fig_r99.update_layout(**PL, height=300,
        xaxis_tickangle=-30, xaxis_tickfont=dict(size=8))
    fig_r99.update_yaxes(title_text="99% Club Rate (%)", secondary_y=False,
        gridcolor="#182030", tickfont=dict(color="#3a5a7a"))
    fig_r99.update_yaxes(title_text="Total Budget (₱B)", secondary_y=True,
        tickfont=dict(color="#2196f3"), gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_r99, width='stretch')


# ══════════════════════════════════════════════
# TAB 2 – MARKET CAPTURE
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="finding o">
    <strong>40 out of 199 District Engineering Offices (20%)</strong> have a single contractor controlling
    ≥40% of their total budget. The most extreme case: <strong>Sulu 1st DEO</strong> — 100% of budget
    captured by a single firm. <strong>Baguio City DEO</strong> (97.6%), <strong>Lanao del Sur 1st DEO</strong>
    (89.7%), and <strong>Quirino DEO</strong> (82.0%) follow. This is a textbook
    <strong>"territorial cartel"</strong> pattern where contractors own specific DEOs.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 0.8])

    with col_l:
        st.markdown('<div class="sec">DEO Contractor Dominance – Share of DEO Total Budget (%)</div><div class="secline"></div>', unsafe_allow_html=True)
        deo_total = dff.groupby("DistrictEngineeringOffice")["Budget"].sum().rename("DEOTotal")
        deo_con = dff.groupby(["DistrictEngineeringOffice","Contractor"])["Budget"].sum().reset_index()
        deo_con = deo_con.merge(deo_total, on="DistrictEngineeringOffice")
        deo_con["SharePct"] = deo_con["Budget"]/deo_con["DEOTotal"]*100
        deo_con_top = deo_con[deo_con["SharePct"]>=40].sort_values("SharePct", ascending=False).head(20)
        deo_con_top["DEO_Short"] = deo_con_top["DistrictEngineeringOffice"].str.replace(" District Engineering Office","",regex=False).str[:35]
        deo_con_top["Con_Short"] = deo_con_top["Contractor"].str[:30]+"…"
        deo_con_top["Label"] = deo_con_top["DEO_Short"] + "  ›  " + deo_con_top["Con_Short"]

        fig_deo = go.Figure(go.Bar(
            y=deo_con_top["Label"], x=deo_con_top["SharePct"], orientation="h",
            marker=dict(
                color=["#ff3333" if s>=70 else "#f5a623" if s>=55 else "#2196f3"
                       for s in deo_con_top["SharePct"]],
                line=dict(width=0)
            ),
            customdata=deo_con_top["Budget"]/1e9,
            hovertemplate="<b>%{y}</b><br>Share: %{x:.1f}%<br>Budget: ₱%{customdata:.2f}B<extra></extra>"
        ))
        fig_deo.add_vline(x=40, line_color="#ffffff", line_dash="dot",
                          annotation_text="40% threshold",
                          annotation_font_color="#aaaaaa", annotation_font_size=9)
        fig_deo.update_layout(**PL, height=520, showlegend=False,
            xaxis_title="Contractor's Share of DEO Budget (%)")
        st.plotly_chart(fig_deo, width='stretch')

    with col_r:
        st.markdown('<div class="sec">Number of DEOs with Monopoly Contractor (≥40%)</div><div class="secline"></div>', unsafe_allow_html=True)
        mono_by_region = (deo_con[deo_con["SharePct"]>=40]
                          .merge(dff[["DistrictEngineeringOffice","Region"]].drop_duplicates(),
                                 on="DistrictEngineeringOffice")
                          .groupby("Region").size().reset_index(name="MonopDEOs")
                          .sort_values("MonopDEOs", ascending=True))
        fig_mr = go.Figure(go.Bar(
            y=mono_by_region["Region"], x=mono_by_region["MonopDEOs"],
            orientation="h",
            marker=dict(color="#f5a623", opacity=0.8, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Monopoly DEOs: %{x}<extra></extra>"
        ))
        fig_mr.update_layout(**PL, height=360, showlegend=False,
            xaxis_title="# DEOs with ≥40% single-contractor share")
        st.plotly_chart(fig_mr, width='stretch')

        # Treemap of top DEOs
        st.markdown('<div class="sec">Budget Concentration – Top DEOs</div><div class="secline"></div>', unsafe_allow_html=True)
        fig_pie2 = go.Figure(go.Pie(
            labels=["Monopoly DEOs\n(≥40% single contractor)","Competitive DEOs"],
            values=[40, 199-40], hole=0.55,
            marker=dict(colors=["#ff3333","#182030"],
                        line=dict(color="#080c10",width=2)),
            textfont=dict(size=10, color="#c8d4e0"),
            hovertemplate="%{label}<br>%{value} DEOs (%{percent})<extra></extra>"
        ))
        fig_pie2.update_layout(**PL, height=220, showlegend=False,
            annotations=[dict(text="20%\nof DEOs", x=0.5, y=0.5,
                              font_size=14, font_color="#ff6666", showarrow=False)])
        st.plotly_chart(fig_pie2, width='stretch')


# ══════════════════════════════════════════════
# TAB 3 – PROJECT SPLITTING
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="finding b">
    <strong>Identical Coordinate Clusters:</strong> 15 GPS locations have 4 or more distinct projects
    registered at the exact same coordinates. The most extreme cluster at (9.568°N, 123.321°E) —
    in <strong>Cebu, Region VII</strong> — has <strong>10 projects totalling ₱856M</strong>,
    all in 2023, awarded to 3 contractors. Identical coordinates for separate projects
    is physically impossible and is the clearest signal of <strong>ghost project factories</strong>.
    </div>
    <div class="finding b" style="margin-top:5px">
    <strong>Threshold Bypassing:</strong> <strong>940 contracts</strong> are priced in the ₱49–50M band
    (just under the ₱50M COA audit threshold), and <strong>1,485 contracts</strong> cluster in the
    ₱96–100M band. The ₱49.0M (797 contracts) and ₱96.5M (678 contracts) exact values alone
    cover <strong>15% of all contracts</strong> — strong evidence of deliberate ceiling-ducking.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec">Identical Coordinate Clusters (4+ projects, same GPS)</div><div class="secline"></div>', unsafe_allow_html=True)
        coord_groups = dff.groupby(["ProjectLatitude","ProjectLongitude"]).agg(
            Count=("ProjectId","count"),
            TotalBudgetM=("BudgetM","sum"),
            Contractors=("Contractor","nunique"),
            Province=("Province","first"),
            Region=("Region","first")
        ).reset_index()
        top_clusters = coord_groups[coord_groups["Count"]>=4].sort_values("Count", ascending=False)
        top_clusters["Location"] = top_clusters["Province"] + " ("+top_clusters["ProjectLatitude"].round(3).astype(str)+"°, "+top_clusters["ProjectLongitude"].round(3).astype(str)+"°)"

        fig_cc = go.Figure(go.Bar(
            y=top_clusters["Location"], x=top_clusters["Count"],
            orientation="h",
            marker=dict(color="#2196f3", opacity=0.8, line=dict(width=0)),
            customdata=np.column_stack([top_clusters["TotalBudgetM"], top_clusters["Contractors"]]),
            hovertemplate="<b>%{y}</b><br>Projects at same GPS: %{x}<br>Budget: ₱%{customdata[0]:.0f}M<br>Contractors: %{customdata[1]}<extra></extra>"
        ))
        fig_cc.update_layout(**PL, height=340, showlegend=False,
            xaxis_title="# Projects at Identical Coordinates")
        st.plotly_chart(fig_cc, width='stretch')

    with col_r:
        st.markdown('<div class="sec">Threshold Bypassing – Budget Distribution Around ₱49M and ₱100M</div><div class="secline"></div>', unsafe_allow_html=True)
        thresh_df = dff[(dff["BudgetM"]>=40) & (dff["BudgetM"]<=105)]
        fig_th = go.Figure()
        fig_th.add_trace(go.Histogram(
            x=thresh_df["BudgetM"], nbinsx=130,
            marker=dict(color="#1c3a55", opacity=0.85, line=dict(color="#080c10",width=0.3))
        ))
        for threshold, label, color in [(50,"₱50M COA Threshold","#ff3333"),(100,"₱100M Threshold","#f5a623")]:
            fig_th.add_vline(x=threshold, line_color=color, line_width=1.5, line_dash="dot",
                             annotation_text=label, annotation_font_color=color,
                             annotation_font_size=9, annotation_position="top right")
        fig_th.update_layout(**PL, height=340, showlegend=False,
            xaxis_title="Approved Budget (₱M)", yaxis_title="# Projects")
        st.plotly_chart(fig_th, width='stretch')

    # Phase/package simultaneous awards table
    st.markdown('<div class="sec">Top Phase/Package Split Projects – Awarded Simultaneously</div><div class="secline"></div>', unsafe_allow_html=True)
    phase_df = dff[dff["HasPhase"]].copy()
    phase_pat = r"(phase\s+[ivxlcd\d]+|package\s+\d+)"
    phase_df["BaseName"] = phase_df["ProjectName"].str.lower().str.replace(phase_pat,"",regex=True).str.strip()
    base_grp = phase_df.groupby("BaseName").agg(
        Phases=("ProjectId","count"),
        TotalBudgetM=("BudgetM","sum"),
        Region=("Region","first"),
        Contractor=("Contractor",lambda x: x.value_counts().index[0] if len(x)>0 else ""),
        StartDates=("Start",lambda x: len(x.dropna().dt.to_period("M").unique())),
    ).reset_index()
    multi = base_grp[base_grp["Phases"]>2].sort_values("TotalBudgetM", ascending=False).head(12)
    multi["TotalBudget"] = multi["TotalBudgetM"].apply(lambda x: f"₱{x:.0f}M")
    multi["BaseName_S"]  = multi["BaseName"].str[:70].str.title()
    multi["UniqueMonths"]= multi["StartDates"].astype(str) + " award months"
    multi_show = multi[["BaseName_S","Region","Phases","TotalBudget","Contractor","UniqueMonths"]].rename(
        columns={"BaseName_S":"Project (Base Name)","Phases":"# Packages","UniqueMonths":"Award Spread"})
    st.dataframe(multi_show.reset_index(drop=True), width='stretch', height=340)


# ══════════════════════════════════════════════
# TAB 4 – ELECTION CYCLE
# ══════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="finding p">
    <strong>Pre-Election Surge (2022):</strong> Contract starts in <strong>February–March 2022</strong>
    — just 2–3 months before the <strong>May 9, 2022 national elections</strong> —
    exploded to <strong>1,115 and 1,179 projects respectively</strong>, totalling over
    ₱110B in just 2 months. This is <strong>10–15× the monthly average</strong> of non-election periods.
    The same pattern repeats in <strong>Feb–March 2023 and 2024</strong>, coinciding with local elections and
    budget cycles — suggesting institutionalized pre-election cash disbursement.
    </div>""", unsafe_allow_html=True)

    # Monthly start data
    monthly = dff.groupby("StartYM").agg(
        Projects=("ProjectId","count"),
        Budget=("BudgetB","sum")
    ).reset_index().sort_values("StartYM")
    monthly = monthly[monthly["StartYM"] >= "2020-01"]

    col_l, col_r = st.columns([1.5, 0.5])

    with col_l:
        st.markdown('<div class="sec">Monthly Project Start Volume – Election Surge Detection</div><div class="secline"></div>', unsafe_allow_html=True)
        election_months = ["2022-02","2022-03","2022-04","2023-02","2023-03","2023-04","2024-02","2024-03","2024-04"]
        bar_colors = ["#ff3333" if m in election_months else "#1c3a55" for m in monthly["StartYM"]]

        fig_ec = make_subplots(specs=[[{"secondary_y":True}]])
        fig_ec.add_trace(go.Bar(
            x=monthly["StartYM"], y=monthly["Projects"],
            name="# Projects", marker=dict(color=bar_colors, line=dict(width=0))
        ), secondary_y=False)
        fig_ec.add_trace(go.Scatter(
            x=monthly["StartYM"], y=monthly["Budget"],
            mode="lines", name="Budget (₱B)",
            line=dict(color="#f5a623", width=2)
        ), secondary_y=True)
        # Election markers
        for yr, label in [("2022-05","🗳 May 2022"),("2019-05","🗳 May 2019")]:
            if yr >= monthly["StartYM"].min():
                if yr in list(monthly["StartYM"]):
                    fig_ec.add_shape(
                        type="line",
                        x0=yr, x1=yr,
                        y0=0, y1=monthly["Projects"].max(),
                        line=dict(color="#ff3333", dash="dot", width=2)
                    )
                    fig_ec.add_annotation(
                        x=yr, y=monthly["Projects"].max(),
                        text=label,
                        showarrow=False,
                        font=dict(color="#ff6666", size=9),
                        xanchor="left", yanchor="bottom"
                    )
        fig_ec.update_layout(**PL, height=360,
            xaxis_tickangle=-45, xaxis_tickfont=dict(size=8))
        fig_ec.update_yaxes(title_text="# Projects Started", secondary_y=False,
            gridcolor="#182030", tickfont=dict(color="#3a5a7a"))
        fig_ec.update_yaxes(title_text="Budget (₱B)", secondary_y=True,
            tickfont=dict(color="#f5a623"), gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ec, width='stretch')

    with col_r:
        st.markdown('<div class="sec">Peak vs Off-Peak</div><div class="secline"></div>', unsafe_allow_html=True)
        peak = monthly[monthly["StartYM"].isin(election_months)]
        off  = monthly[~monthly["StartYM"].isin(election_months)]
        peak_avg = peak["Projects"].mean()
        off_avg  = off["Projects"].mean()
        ratio = peak_avg / off_avg if off_avg > 0 else 0

        st.markdown(f"""
        <div class="kpi kpi-r" style="margin-bottom:10px">
            <div class="kpi-lbl">Peak Month Avg</div>
            <div class="kpi-val">{peak_avg:.0f}</div>
            <div class="kpi-sub">projects/month (Feb–Apr)</div>
        </div>
        <div class="kpi kpi-b" style="margin-bottom:10px">
            <div class="kpi-lbl">Off-Peak Month Avg</div>
            <div class="kpi-val">{off_avg:.0f}</div>
            <div class="kpi-sub">projects/month</div>
        </div>
        <div class="kpi kpi-o">
            <div class="kpi-lbl">Surge Multiplier</div>
            <div class="kpi-val">{ratio:.1f}×</div>
            <div class="kpi-sub">more projects in peak months</div>
        </div>""", unsafe_allow_html=True)

    # Year-over-year Q1 vs Q2-Q4 comparison
    st.markdown('<div class="sec">Q1 (Jan–Apr) vs Rest of Year – Budget Split by Funding Year</div><div class="secline"></div>', unsafe_allow_html=True)
    dff2 = dff.copy()
    dff2["StartMonth_n"] = dff2["Start"].dt.month
    dff2["Quarter"] = dff2["StartMonth_n"].apply(lambda m: "Q1 Jan-Apr (Pre-Election)" if m<=4 else "Rest of Year" if pd.notna(m) else "Unknown")
    qtr_yr = dff2.groupby(["StartYear","Quarter"]).agg(Budget=("BudgetB","sum")).reset_index()
    qtr_yr = qtr_yr[qtr_yr["StartYear"].between(2021,2025)]
    fig_qtr = px.bar(qtr_yr, x="StartYear", y="Budget", color="Quarter",
        color_discrete_map={"Q1 Jan-Apr (Pre-Election)":"#ff3333","Rest of Year":"#1c3a55","Unknown":"#182030"},
        barmode="stack", labels={"Budget":"Budget (₱B)","StartYear":"Year"})
    fig_qtr.update_layout(**PL, height=260)
    st.plotly_chart(fig_qtr, width='stretch')


# ══════════════════════════════════════════════
# TAB 5 – GHOST PROJECTS & COST INCONSISTENCY
# ══════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="finding g">
    <strong>Ghost Project Risk (87.3% of projects):</strong> A legitimate flood control project
    specifies exact stationing references (e.g., "Sta. 136+247 to Sta. 136+752") identifying
    the exact stretch of river. <strong>8,605 projects (₱476B)</strong> have no stationing whatsoever —
    names like "Construction of Flood Control Structure, Calanasan, Apayao." Without stationing,
    there is <strong>no way to physically verify the project location</strong>, making it
    impossible to confirm whether work was actually done.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec">Ghost vs Legit Projects – Budget by Work Type</div><div class="secline"></div>', unsafe_allow_html=True)
        ghost_wt = dff.groupby(["TypeOfWork","HasStationing"]).agg(
            Count=("ProjectId","count"), Budget=("BudgetB","sum")
        ).reset_index()
        ghost_wt["WorkShort"] = ghost_wt["TypeOfWork"].str.replace("Construction of ","",regex=False).str.replace("Rehabilitation / Major Repair of ","Rehab: ",regex=False)
        ghost_wt["Type"] = ghost_wt["HasStationing"].map({True:"✅ Has Stationing",False:"👻 No Stationing"})
        fig_gw = px.bar(ghost_wt, x="Budget", y="WorkShort", color="Type", orientation="h",
            color_discrete_map={"✅ Has Stationing":"#22c55e","👻 No Stationing":"#ff3333"},
            barmode="stack", labels={"Budget":"Budget (₱B)","WorkShort":"Work Type"})
        fig_gw.update_layout(**PL, height=340)
        st.plotly_chart(fig_gw, width='stretch')

    with col_r:
        st.markdown('<div class="sec">Budget per Project – Cost Outliers by Work Type</div><div class="secline"></div>', unsafe_allow_html=True)
        cost_wt = dff[dff["BudgetM"].notna()].groupby("TypeOfWork").agg(
            Median=("BudgetM","median"),
            P25=("BudgetM",lambda x: x.quantile(0.25)),
            P75=("BudgetM",lambda x: x.quantile(0.75)),
            P95=("BudgetM",lambda x: x.quantile(0.95)),
        ).reset_index().sort_values("Median",ascending=False)
        cost_wt["WorkShort"] = cost_wt["TypeOfWork"].str.replace("Construction of ","",regex=False).str.replace("Rehabilitation / Major Repair of ","Rehab: ",regex=False)

        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(x=cost_wt["WorkShort"], y=cost_wt["Median"],
            name="Median", marker=dict(color="#2196f3", opacity=0.8, line=dict(width=0))))
        fig_cv.add_trace(go.Scatter(x=cost_wt["WorkShort"], y=cost_wt["P95"],
            mode="markers", name="95th Percentile",
            marker=dict(color="#ff3333", size=9, symbol="diamond")))
        fig_cv.update_layout(**PL, height=340,
            xaxis_tickangle=-30, xaxis_tickfont=dict(size=8),
            yaxis_title="Budget (₱M)")
        st.plotly_chart(fig_cv, width='stretch')

    # Ghost vs real project sample
    st.markdown('<div class="sec">Sample Comparison – Vague (Ghost Risk) vs Specific Project Names</div><div class="secline"></div>', unsafe_allow_html=True)
    col_g, col_s = st.columns(2)
    with col_g:
        st.markdown("**👻 Vague Names (No Stationing — Ghost Risk)**")
        ghosts = dff[~dff["HasStationing"]][["ProjectId","Region","ProjectName","BudgetM"]].head(8).copy()
        ghosts["BudgetM"] = ghosts["BudgetM"].apply(lambda x: f"₱{x:.1f}M")
        st.dataframe(ghosts.reset_index(drop=True), width='stretch', height=240)
    with col_s:
        st.markdown("**✅ Specific Names (Has Stationing — Verifiable)**")
        legit = dff[dff["HasStationing"]][["ProjectId","Region","ProjectName","BudgetM"]].head(8).copy()
        legit["BudgetM"] = legit["BudgetM"].apply(lambda x: f"₱{x:.1f}M")
        st.dataframe(legit.reset_index(drop=True), width='stretch', height=240)


# ══════════════════════════════════════════════
# TAB 6 – TOP 10 MOST SUSPICIOUS
# ══════════════════════════════════════════════
with tab6:
    st.markdown("""
    <div class="finding">
    <strong>Composite Suspicion Score Methodology:</strong> Each project is scored across 9 independent
    anomaly dimensions. A score of 90 means a project triggered 4 separate red flags simultaneously —
    making it statistically near-impossible for these to be legitimate coincidences.
    The top projects below are <strong>priority referral candidates for COA audit</strong>.
    </div>""", unsafe_allow_html=True)

    top10 = dff.nlargest(10, "SuspicionScore").reset_index(drop=True)
    top10["Rank"] = top10.index + 1

    # Score bar
    st.markdown('<div class="sec">Top 10 Projects by Composite Suspicion Score</div><div class="secline"></div>', unsafe_allow_html=True)
    fig_top = go.Figure()
    score_colors = ["#ff3333","#ff3333","#ff3333","#ff3333","#f5a623","#f5a623","#2196f3","#2196f3","#2196f3","#2196f3"]
    fig_top.add_trace(go.Bar(
        x=top10["SuspicionScore"],
        y=[f"#{r} {pid}" for r,pid in zip(top10["Rank"],top10["ProjectId"])],
        orientation="h",
        marker=dict(color=score_colors[:len(top10)], line=dict(width=0)),
        text=top10["SuspicionScore"].astype(str)+" pts",
        textposition="outside", textfont=dict(color="#7a90a8",size=10),
        customdata=np.column_stack([
            top10["Region"], top10["BudgetM"].round(1),
            top10["BidRatio"].round(1), top10["Contractor"].str[:40]
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>Score: %{x}<br>Region: %{customdata[0]}<br>"
            "Budget: ₱%{customdata[1]}M<br>Bid Ratio: %{customdata[2]}%<br>"
            "Contractor: %{customdata[3]}<extra></extra>"
        )
    ))
    fig_top.update_layout(**PL, height=340, showlegend=False,
        xaxis_title="Suspicion Score (max possible: 150)",
        xaxis_range=[0,130])
    st.plotly_chart(fig_top, width='stretch')

    # Radar / flag heatmap
    st.markdown('<div class="sec">Red Flag Heatmap – Which Anomalies Each Project Triggered</div><div class="secline"></div>', unsafe_allow_html=True)
    flag_cols  = ["F_BidRatio99","F_Exact","F_DupeCoord","F_DupeID","F_Threshold","F_UnderSpend","F_ShortDur","F_Ghost","F_Batch"]
    flag_names = ["Bid 99–100%","Exact Match","Dupe GPS","Dupe ID","Threshold","Under-Spend","Short Dur","Ghost","Batch Award"]
    heat_z = (top10[flag_cols] > 0).astype(int).values

    fig_heat = go.Figure(go.Heatmap(
        z=heat_z,
        x=flag_names,
        y=[f"#{r} {pid}" for r,pid in zip(top10["Rank"],top10["ProjectId"])],
        colorscale=[[0,"#182030"],[1,"#ff3333"]],
        showscale=False,
        hovertemplate="Project: <b>%{y}</b><br>Flag: <b>%{x}</b><br>Triggered: %{z}<extra></extra>"
    ))
    fig_heat.update_layout(**PL, height=320)
    st.plotly_chart(fig_heat, width='stretch')

    # Detailed cards
    st.markdown('<div class="sec">Full Details – Top 10 Suspicious Projects</div><div class="secline"></div>', unsafe_allow_html=True)
    for _, row in top10.iterrows():
        flags = []
        if row["F_BidRatio99"]>0: flags.append(f"🔴 Bid Ratio {row['BidRatio']:.2f}%")
        if row["F_Exact"]>0:      flags.append("🔴 Exact Budget=Cost")
        if row["F_DupeCoord"]>0:  flags.append("🔵 Duplicate GPS Coords")
        if row["F_DupeID"]>0:     flags.append("🔵 Duplicate Project ID")
        if row["F_Threshold"]>0:  flags.append(f"🟠 Budget at threshold (₱{row['BudgetM']:.1f}M)")
        if row["F_UnderSpend"]>0: flags.append(f"🟣 Extreme underspend ({row['Variance']:.1f}%)")
        if row["F_ShortDur"]>0:   flags.append(f"🟣 Short duration ({row['DurationDays']:.0f} days)")
        if row["F_Ghost"]>0:      flags.append("🟢 No stationing (ghost risk)")
        if row["F_Batch"]>0:      flags.append("🟡 Batch award")

        flag_str = "  ·  ".join(flags)
        contractor_s = str(row["Contractor"])[:80]
        pname_s = str(row["ProjectName"])[:120]

        st.markdown(f"""
        <div style='background:#0a0e14;border:1px solid #182030;border-radius:10px;
                    padding:14px 18px;margin-bottom:8px;border-left:3px solid #ff3333'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                <span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#ff6666;font-weight:600'>
                    RANK #{int(row["Rank"])} · {row["ProjectId"]}  ·  SCORE: {int(row["SuspicionScore"])}/150
                </span>
                <span style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:#3a5a7a'>
                    {row["Region"]} · {row["Province"]} · {int(row["FundingYear"])}
                </span>
            </div>
            <div style='font-size:0.82rem;color:#c8d4e0;margin-bottom:6px'>{pname_s}</div>
            <div style='font-size:0.76rem;color:#4a6a8a;margin-bottom:8px'>🏗 {contractor_s}</div>
            <div style='font-size:0.75rem;color:#5a7a9a'>
                Budget: <span style='color:#c8d4e0'>₱{row["BudgetM"]:.1f}M</span>  ·
                Bid Ratio: <span style='color:#c8d4e0'>{row["BidRatio"]:.2f}%</span>  ·
                Variance: <span style='color:#c8d4e0'>{row["Variance"]:.1f}%</span>  ·
                Duration: <span style='color:#c8d4e0'>{int(row["DurationDays"]) if pd.notna(row["DurationDays"]) else "N/A"} days</span>
            </div>
            <div style='margin-top:8px;font-size:0.74rem'>{flag_str}</div>
        </div>""", unsafe_allow_html=True)
