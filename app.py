"""
Enterprise Agentic Supply Chain Optimizer
SAP BTP AI Core · SupplyGraph · M5
Author: Namrath Basavaraju
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE   = "#0070f3"
GREEN  = "#107e3e"
ORANGE = "#e76500"
RED    = "#bb0000"

# -- CSS -----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: "Inter", Arial, sans-serif;
        color: #1a1a2e;
    }}

    /* Remove default Streamlit top padding */
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

    /* Header */
    .app-header {{
        border-bottom: 2px solid {BLUE};
        padding-bottom: 14px;
        margin-bottom: 24px;
    }}
    .app-header h2 {{
        margin: 0 0 4px 0;
        font-size: 1.45rem;
        font-weight: 700;
        color: {BLUE};
        letter-spacing: -0.3px;
    }}
    .app-header p {{
        margin: 0;
        font-size: 0.8rem;
        color: #666;
    }}

    /* KPI metric card */
    .metric-card {{
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }}
    .metric-card .mc-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888;
        margin-bottom: 6px;
    }}
    .metric-card .mc-value {{
        font-size: 1.7rem;
        font-weight: 700;
        color: {BLUE};
        line-height: 1;
    }}
    .metric-card .mc-sub {{
        font-size: 0.72rem;
        color: #aaa;
        margin-top: 5px;
    }}
    .metric-card.green  .mc-value {{ color: {GREEN};  }}
    .metric-card.orange .mc-value {{ color: {ORANGE}; }}
    .metric-card.purple .mc-value {{ color: #7c3aed;  }}

    /* Agent status card */
    .agent-card {{
        background: #fff;
        border: 1px solid #e8eaed;
        border-top: 3px solid #e8eaed;
        border-radius: 6px;
        padding: 14px 12px;
        text-align: center;
    }}
    .agent-card.done {{ border-top-color: {GREEN}; }}
    .agent-card .ac-num  {{ font-size: 1.2rem; font-weight: 700; color: {BLUE}; }}
    .agent-card .ac-name {{ font-size: 0.75rem; font-weight: 600; color: #333; margin-top: 4px; }}
    .agent-card .ac-desc {{ font-size: 0.65rem; color: #999; margin-top: 4px; line-height: 1.35; }}
    .agent-card .ac-badge {{
        display: inline-block;
        margin-top: 8px;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
    }}
    .badge-done    {{ background: #d4edda; color: {GREEN};  }}
    .badge-pending {{ background: #f3f4f6; color: #aaa;    }}

    /* Terminal */
    .terminal {{
        background: #0f172a;
        color: #94fa7a;
        font-family: "Courier New", monospace;
        font-size: 0.72rem;
        padding: 14px 16px;
        border-radius: 6px;
        max-height: 280px;
        overflow-y: auto;
        line-height: 1.55;
        border: 1px solid #1e293b;
    }}

    /* Section heading */
    .sec-title {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #555;
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #e8eaed;
    }}

    /* Bench table status */
    .status-ok   {{ color: {GREEN};  font-weight: 600; }}
    .status-warn {{ color: {ORANGE}; font-weight: 600; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: #f8fafc;
        border-right: 1px solid #e8eaed;
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

    /* Tab strip */
    button[data-baseweb="tab"] {{
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
        color: #666 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {BLUE} !important;
        border-bottom: 2px solid {BLUE} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -- Helpers -------------------------------------------------------------------
def metric_card(label, value, sub="", variant=""):
    cls = f"metric-card {variant}".strip()
    sub_html = f'<div class="mc-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="{cls}">
        <div class="mc-label">{label}</div>
        <div class="mc-value">{value}</div>
        {sub_html}
    </div>
    """


def chart_layout(fig, height=340, title=""):
    fig.update_layout(
        height=height,
        title=title,
        title_font_size=13,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_family="Inter, Arial, sans-serif",
        font_color="#333",
        margin=dict(l=12, r=12, t=40 if title else 16, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font_size=11,
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e8eaed", linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f2f5", linecolor="#e8eaed")
    return fig


# -- Session state -------------------------------------------------------------
for k, v in [("results", None), ("agent_logs", []), ("run_complete", False)]:
    if k not in st.session_state:
        st.session_state[k] = v


# -- Header --------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h2>Enterprise Agentic Supply Chain Optimizer</h2>
        <p>Transforming FMCG Pack-Size Strategy on SAP BTP &nbsp;&middot;&nbsp;
           SupplyGraph (Wasi et al., AAAI 2024) &nbsp;&middot;&nbsp;
           M5 (Makridakis et al.) &nbsp;&middot;&nbsp;
           Namrath Basavaraju &mdash; MSc Data Science, University of Mannheim</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Configuration")
    st.markdown("---")

    @st.cache_data(show_spinner=False)
    def _load_filter_data():
        try:
            from utils.data_loader import UnifiedDataLoader
            loader = UnifiedDataLoader()
            df, _ = loader.load()
            return df
        except Exception:
            return pd.DataFrame()

    filter_df = _load_filter_data()

    if not filter_df.empty:
        all_skus   = sorted(filter_df["SkuId"].unique().tolist())
        all_groups = sorted(filter_df["ProductGroup"].unique().tolist())
        all_plants = sorted(filter_df["PlantId"].unique().tolist())
        date_min   = filter_df["Date"].min()
        date_max   = filter_df["Date"].max()
    else:
        all_skus   = [f"SKU_{i:03d}" for i in range(1, 11)]
        all_groups = ["S", "P", "E"]
        all_plants = ["2120", "2130", "2140"]
        date_min   = pd.Timestamp("2023-01-01")
        date_max   = pd.Timestamp("2023-08-31")

    selected_group = st.selectbox("Product Group", ["All"] + all_groups)
    selected_plant = st.selectbox("Plant", ["All"] + all_plants)

    if selected_group != "All" and not filter_df.empty:
        sku_pool = filter_df.loc[
            filter_df["ProductGroup"] == selected_group, "SkuId"
        ].unique().tolist()
    else:
        sku_pool = all_skus

    selected_skus = st.multiselect("SKUs (leave empty for all)", sku_pool, default=[])
    horizon_days  = st.slider("Forecast Horizon (days)", 7, 90, 30)

    st.markdown("---")
    run_btn = st.button("Run All Agents", type="primary", use_container_width=True)

    # Download section (shown after pipeline runs)
    if st.session_state.run_complete:
        st.markdown("---")
        st.markdown("#### Download Results")

        res = st.session_state.results

        def to_csv(df):
            return df.to_csv(index=False, encoding="utf-8").encode("utf-8")

        downloads = [
            ("Demand Forecast",        "01_demand_forecast.csv",        res["demand"]["forecast_df"]),
            ("Pack Recommendations",   "02_pack_recommendations.csv",   res["pack_size"]["recommendations_df"]),
            ("SKU Velocity Profiles",  "03_sku_velocity_profiles.csv",  res["pack_size"]["velocity_profile_df"]),
            ("Financial Impact",       "04_financial_impact.csv",       res["financial"]["financial_table_df"]),
            ("Production Schedule",    "05_production_schedule.csv",    res["production"]["production_schedule_df"]),
            ("Plant Utilisation",      "07_plant_utilisation.csv",      res["production"]["utilisation_summary_df"]),
            ("Dispatch Plan",          "08_dispatch_plan.csv",          res["dispatch"]["dispatch_plan_df"]),
            ("Routing Paths",          "09_routing_paths.csv",          res["dispatch"]["path_df"]),
        ]

        # Summary + benchmarks always at top
        ps = res.get("pipeline_summary", {})
        fin_s = res["financial"]["summary"]
        bench_df = pd.DataFrame([
            ("PBT Uplift %",          fin_s.get("PBT_Uplift_Pct", 0),              10, 18),
            ("Revenue Uplift %",      fin_s.get("Revenue_Uplift_Pct", 0),           8, 15),
            ("Inventory Reduction %", fin_s.get("Inventory_Reduction_Pct", 0),      20, 30),
            ("Lead Time Reduction %", res["dispatch"]["summary"]["avg_lead_time_reduction_pct"], 20, 30),
            ("Debtor Improvement %",  fin_s.get("Debtor_Cycle_Improvement_Pct", 0), 25, 35),
        ], columns=["Metric", "Achieved", "Target_Min", "Target_Max"])
        bench_df["Status"] = bench_df.apply(
            lambda r: "On Target" if r.Target_Min <= r.Achieved <= r.Target_Max else "Out of Range", axis=1
        )

        st.download_button(
            "Download Benchmark Summary",
            data=to_csv(bench_df),
            file_name="00_benchmark_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        for label, fname, df in downloads:
            st.download_button(
                f"Download {label}",
                data=to_csv(df),
                file_name=fname,
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{fname}",
            )

    st.markdown("---")
    sap_mode = st.toggle("SAP BTP AI Core Simulation", value=True)
    if sap_mode:
        st.success("SAP BTP AI Core: Active")
    else:
        st.info("Standalone mode")
    st.caption(
        "SAP BTP AI Core Simulation uses enterprise terminology (Joule Agents, "
        "HANA Cloud Vector Engine, SAP Analytics Cloud) for illustrative purposes. "
        "This application does not connect to any SAP servers or live SAP services."
    )

    st.markdown("---")
    st.caption("SupplyGraph · M5 · SAP BTP AI Core")


# -- Run orchestrator ----------------------------------------------------------
def run_pipeline(sku_filter, horizon):
    from orchestrator.orchestrator import AgentOrchestrator

    log_placeholder = st.empty()
    logs_so_far = []

    def progress_cb(step, total, msg):
        ts = pd.Timestamp.now().strftime("%H:%M:%S")
        logs_so_far.append(f"[{ts}]  {msg}")
        log_html = "<br>".join(logs_so_far[-40:])
        log_placeholder.markdown(
            f'<div class="terminal">{log_html}</div>', unsafe_allow_html=True
        )

    orch = AgentOrchestrator(progress_callback=progress_cb)
    results = orch.run(filters={"sku_filter": sku_filter or None, "horizon_days": horizon})
    st.session_state.agent_logs = results.get("orchestration_log", [])
    return results


if run_btn:
    with st.spinner("Running 5-agent pipeline ..."):
        results = run_pipeline(selected_skus, horizon_days)
    st.session_state.results      = results
    st.session_state.run_complete = results.get("status") == "success"
    if st.session_state.run_complete:
        st.toast("Pipeline completed successfully.")
    else:
        st.error(f"Pipeline error: {results.get('error', 'Unknown')}")


# -- Tabs ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Orchestration",
    "Demand Intelligence",
    "Pack Size & Inventory",
    "Financial Impact",
    "Production & Dispatch",
])


# ===============================================================================
# TAB 1 -- Orchestration
# ===============================================================================
with tab1:
    # Agent pipeline status cards
    agents_meta = [
        ("1", "Demand Intelligence",  "RandomForest · M5 calendar signals"),
        ("2", "Pack Size Optimization","KMeans · 3 velocity classes"),
        ("3", "Financial Impact",      "PBT · Debtor cycle · Revenue"),
        ("4", "Production Planning",   "EOQ batch sizing · Utilisation"),
        ("5", "Dispatch Optimization", "NetworkX shortest-path routing"),
    ]
    done = st.session_state.run_complete
    cols = st.columns(5)
    for i, (num, name, desc) in enumerate(agents_meta):
        badge = '<span class="ac-badge badge-done">Completed</span>' if done else '<span class="ac-badge badge-pending">Pending</span>'
        with cols[i]:
            st.markdown(
                f'<div class="agent-card {"done" if done else ""}">'
                f'<div class="ac-num">Agent {num}</div>'
                f'<div class="ac-name">{name}</div>'
                f'<div class="ac-desc">{desc}</div>'
                f'{badge}</div>',
                unsafe_allow_html=True,
            )

    # Pipeline summary KPIs
    if st.session_state.run_complete:
        ps = st.session_state.results.get("pipeline_summary", {})
        st.markdown('<div class="sec-title">Pipeline Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(metric_card("SKUs Forecasted",    str(ps.get("skus_forecasted", 0))), unsafe_allow_html=True)
        c2.markdown(metric_card("PBT Uplift",         f"{ps.get('pbt_uplift_pct', 0):.1f}%",  "Target 10–18%", "green"),  unsafe_allow_html=True)
        c3.markdown(metric_card("Inventory Reduction",f"{ps.get('inventory_reduction_pct', 0):.1f}%","Target 20–30%","orange"),unsafe_allow_html=True)
        c4.markdown(metric_card("Lead Time Reduction",f"{ps.get('avg_lead_time_reduction_pct', 0):.1f}%","Target 20–30%",""),unsafe_allow_html=True)
        c5.markdown(metric_card("Debtor Improvement", f"{ps.get('debtor_cycle_improvement_pct', 0):.1f}%","Target 25–35%","purple"),unsafe_allow_html=True)

    # Terminal log
    st.markdown('<div class="sec-title">Orchestration Log</div>', unsafe_allow_html=True)
    logs = st.session_state.agent_logs
    if logs:
        st.markdown(
            f'<div class="terminal">{"<br>".join(logs)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="terminal" style="color:#4a5568">Waiting for pipeline execution. '
            'Press <strong>Run All Agents</strong> in the sidebar.</div>',
            unsafe_allow_html=True,
        )


# ===============================================================================
# TAB 2 -- Demand Intelligence
# ===============================================================================
with tab2:
    if not st.session_state.run_complete:
        st.info("Run the pipeline to see demand forecasts.")
    else:
        res        = st.session_state.results
        demand     = res.get("demand", {})
        metrics    = demand.get("metrics", {})
        forecast_df = demand.get("forecast_df", pd.DataFrame())
        actuals_df  = demand.get("actuals_df",  pd.DataFrame())

        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("MAPE",           f"{metrics.get('MAPE', 0):.2f}%",  "Mean Absolute % Error"), unsafe_allow_html=True)
        c2.markdown(metric_card("RMSE",           f"{metrics.get('RMSE', 0):,.1f}",  "Root Mean Sq Error"),    unsafe_allow_html=True)
        c3.markdown(metric_card("SKUs Forecasted",str(metrics.get("skus_forecasted", 0)), f"Horizon: {metrics.get('horizon_days', 0)}d"), unsafe_allow_html=True)

        # Actual vs Forecast
        if not forecast_df.empty and not actuals_df.empty:
            st.markdown('<div class="sec-title">Actual vs Forecasted Demand</div>', unsafe_allow_html=True)
            sku_opts = forecast_df["SkuId"].unique().tolist()
            sel_sku  = st.selectbox("Select SKU", sku_opts[:20], key="demand_sku")

            fcast_sku = forecast_df[forecast_df["SkuId"] == sel_sku]
            act_sku   = actuals_df[actuals_df["SkuId"] == sel_sku].tail(90)

            fig_ts = go.Figure()
            if not act_sku.empty:
                fig_ts.add_trace(go.Scatter(
                    x=act_sku["Date"], y=act_sku["ActualQty"],
                    name="Actual", line=dict(color=BLUE, width=2),
                ))
            if not fcast_sku.empty:
                fig_ts.add_trace(go.Scatter(
                    x=fcast_sku["Date"], y=fcast_sku["ForecastQty"],
                    name="Forecast", line=dict(color=ORANGE, width=2, dash="dash"),
                ))
                fig_ts.add_trace(go.Scatter(
                    x=pd.concat([fcast_sku["Date"], fcast_sku["Date"][::-1]]),
                    y=pd.concat([fcast_sku["CI_Upper"], fcast_sku["CI_Lower"][::-1]]),
                    fill="toself", fillcolor="rgba(231,101,0,0.1)",
                    line=dict(color="rgba(0,0,0,0)"), name="80% CI",
                    showlegend=True,
                ))
            st.plotly_chart(chart_layout(fig_ts, 340, f"Demand Forecast — {sel_sku}"), use_container_width=True)

        # Volatility heatmap
        master_df = res.get("master_df", pd.DataFrame())
        if not master_df.empty:
            st.markdown('<div class="sec-title">Demand Volatility by Product Group &amp; Month</div>', unsafe_allow_html=True)
            _tmp = master_df.copy()
            _tmp["Month"] = pd.to_datetime(_tmp["Date"]).dt.strftime("%b")
            hm    = _tmp.groupby(["ProductGroup", "Month"])["SalesOrderQty"].std().reset_index()
            mo    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            hm["Month"] = pd.Categorical(hm["Month"], categories=mo, ordered=True)
            pivot = hm.sort_values("Month").pivot_table(
                index="ProductGroup", columns="Month",
                values="SalesOrderQty", fill_value=0,
            )
            fig_hm = px.imshow(
                pivot,
                color_continuous_scale=[[0, "#f0f4ff"], [0.5, BLUE], [1, "#003d82"]],
                aspect="auto", height=260,
            )
            st.plotly_chart(chart_layout(fig_hm, 260), use_container_width=True)

        # Feature importance
        fi = metrics.get("feature_importance", {})
        if fi:
            st.markdown('<div class="sec-title">RandomForest Feature Importance</div>', unsafe_allow_html=True)
            fi_df = pd.DataFrame(
                sorted(fi.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Importance"],
            )
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color_discrete_sequence=[BLUE], height=280,
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(chart_layout(fig_fi, 280), use_container_width=True)


# ===============================================================================
# TAB 3 -- Pack Size & Inventory
# ===============================================================================
with tab3:
    if not st.session_state.run_complete:
        st.info("Run the pipeline to see pack recommendations.")
    else:
        res    = st.session_state.results
        pack   = res.get("pack_size", {})
        reco_df = pack.get("recommendations_df",  pd.DataFrame())
        vel_df  = pack.get("velocity_profile_df", pd.DataFrame())
        pack_s  = pack.get("summary", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Fast Movers",          str(pack_s.get("fast_movers", 0)),   "Bulk Logistics Case",    "green"),  unsafe_allow_html=True)
        c2.markdown(metric_card("Medium Movers",        str(pack_s.get("medium_movers", 0)), "Standard Consumer Pack", ""),       unsafe_allow_html=True)
        c3.markdown(metric_card("Slow Movers",          str(pack_s.get("slow_movers", 0)),   "Promotional Multipack",  "orange"), unsafe_allow_html=True)
        c4.markdown(metric_card("Avg Inv. Reduction",   f"{pack_s.get('avg_inventory_reduction_pct', 0):.1f}%", "Target 20–30%", "green"), unsafe_allow_html=True)

        # Scatter
        if not vel_df.empty:
            st.markdown('<div class="sec-title">SKU Velocity Clustering</div>', unsafe_allow_html=True)
            fig_sc = px.scatter(
                vel_df.fillna(0),
                x="hist_mean", y="fcast_mean",
                color="velocity_class",
                color_discrete_map={
                    "fast_mover": GREEN, "medium_mover": BLUE, "slow_mover": ORANGE,
                },
                hover_data=["SkuId", "ProductGroup", "hist_cv"],
                labels={
                    "hist_mean": "Historical Mean Daily Demand",
                    "fcast_mean": "Forecast Mean Daily Demand",
                    "velocity_class": "Velocity",
                },
                height=360,
            )
            fig_sc.update_traces(marker=dict(size=10, opacity=0.75))
            st.plotly_chart(chart_layout(fig_sc, 360), use_container_width=True)

        if not reco_df.empty:
            col_l, col_r = st.columns([2, 1])

            with col_l:
                st.markdown('<div class="sec-title">Pack Recommendations</div>', unsafe_allow_html=True)
                display_cols = [
                    "SkuId", "ProductGroup", "VelocityClass", "PackConfig",
                    "PackMultiplier", "RecommendedPackQty", "JustificationScore",
                    "EstInventoryReductionPct", "CapacityFeasible",
                ]
                disp = reco_df[[c for c in display_cols if c in reco_df.columns]].copy()
                disp["JustificationScore"] = disp["JustificationScore"].round(1)
                try:
                    styled = disp.style.format({
                        "EstInventoryReductionPct": "{:.1f}%",
                        "RecommendedPackQty": "{:,.0f}",
                    }).background_gradient(subset=["JustificationScore"], cmap="Blues")
                    st.dataframe(styled, use_container_width=True, height=380)
                except Exception:
                    st.dataframe(disp, use_container_width=True, height=380)

            with col_r:
                st.markdown('<div class="sec-title">Pack Config Distribution</div>', unsafe_allow_html=True)
                pie = reco_df["PackConfig"].value_counts().reset_index()
                pie.columns = ["PackConfig", "Count"]
                fig_pie = px.pie(
                    pie, values="Count", names="PackConfig",
                    color_discrete_sequence=[GREEN, BLUE, ORANGE],
                    hole=0.4,
                    height=340,
                )
                fig_pie.update_traces(textposition="outside", textinfo="percent+label")
                fig_pie.update_layout(showlegend=False, paper_bgcolor="#fff",
                                      margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)


# ===============================================================================
# TAB 4 -- Financial Impact
# ===============================================================================
with tab4:
    if not st.session_state.run_complete:
        st.info("Run the pipeline to see financial projections.")
    else:
        res   = st.session_state.results
        fin   = res.get("financial", {})
        fin_s = fin.get("summary", {})
        fin_df = fin.get("financial_table_df", pd.DataFrame())

        # KPI row 1 -- uplift %
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("PBT Uplift",            f"{fin_s.get('PBT_Uplift_Pct', 0):.1f}%",               "Target 10–18%", "green"),  unsafe_allow_html=True)
        c2.markdown(metric_card("Revenue Uplift",        f"{fin_s.get('Revenue_Uplift_Pct', 0):.1f}%",            "Target 8–15%",  ""),       unsafe_allow_html=True)
        c3.markdown(metric_card("Inventory Reduction",   f"{fin_s.get('Inventory_Reduction_Pct', 0):.1f}%",       "Target 20–30%", "orange"), unsafe_allow_html=True)
        c4.markdown(metric_card("Debtor Improvement",    f"{fin_s.get('Debtor_Cycle_Improvement_Pct', 0):.1f}%",  "Target 25–35%", "purple"), unsafe_allow_html=True)
        st.caption(
            "Benchmark ranges (PBT 10–18%, Revenue 8–15%, Debtor 25–35%, Inventory 20–30%) are "
            "sourced from McKinsey Global Institute, Gartner Supply Chain Research, and SymphonyAI "
            "FMCG industry reports. These are published industry benchmarks and do not represent "
            "empirical results from this simulation."
        )

        # KPI row 2 -- absolute £
        def fmt_m(v):
            return f"£{v/1e6:.1f}M" if abs(v) >= 1e6 else f"£{v:,.0f}"

        before_rev = fin_s.get("Before_TotalRevenue", 0)
        after_rev  = fin_s.get("After_TotalRevenue",  0)
        before_pbt = fin_s.get("Before_TotalPBT", 0)
        after_pbt  = fin_s.get("After_TotalPBT",  0)

        c5, c6, c7, c8 = st.columns(4)
        c5.markdown(metric_card("Revenue (Before)", fmt_m(before_rev)), unsafe_allow_html=True)
        c6.markdown(metric_card("Revenue (After)",  fmt_m(after_rev),  f"+{fmt_m(after_rev - before_rev)}", "green"), unsafe_allow_html=True)
        c7.markdown(metric_card("PBT (Before)",     fmt_m(before_pbt)), unsafe_allow_html=True)
        c8.markdown(metric_card("PBT (After)",      fmt_m(after_pbt),  f"+{fmt_m(after_pbt - before_pbt)}", "green"), unsafe_allow_html=True)

        if not fin_df.empty:
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown('<div class="sec-title">Before vs After PBT — Top 20 SKUs</div>', unsafe_allow_html=True)
                fin_top = fin_df.nlargest(20, "PBT_Uplift_Pct")[
                    ["SkuId", "Before_PBT", "After_PBT"]
                ].melt(id_vars="SkuId", var_name="Scenario", value_name="PBT")
                fig_bar = px.bar(
                    fin_top, x="SkuId", y="PBT", color="Scenario",
                    barmode="group",
                    color_discrete_map={"Before_PBT": "#d1d5db", "After_PBT": GREEN},
                    height=320,
                )
                fig_bar.update_layout(xaxis_tickangle=-40)
                st.plotly_chart(chart_layout(fig_bar, 320), use_container_width=True)

            with col_r:
                st.markdown('<div class="sec-title">PBT Bridge (£M)</div>', unsafe_allow_html=True)
                rev_delta  = after_rev - before_rev
                cogs_save  = fin_df["Before_COGS"].sum() - fin_df["After_COGS"].sum() if "After_COGS" in fin_df.columns else rev_delta * 0.05
                inv_save   = fin_df["Before_CarryingCost"].sum() - fin_df["After_CarryingCost"].sum() if "After_CarryingCost" in fin_df.columns else rev_delta * 0.03
                debt_save  = fin_df["Before_DebtorCost"].sum() - fin_df["After_DebtorCost"].sum() if "After_DebtorCost" in fin_df.columns else rev_delta * 0.02
                net        = after_pbt - before_pbt

                fig_wf = go.Figure(go.Waterfall(
                    orientation="v",
                    x=["Revenue Uplift", "COGS Reduction", "Inventory Savings", "Debtor Savings", "Net PBT"],
                    y=[rev_delta/1e6, cogs_save/1e6, inv_save/1e6, debt_save/1e6, net/1e6],
                    connector=dict(line=dict(color="#d1d5db")),
                    increasing=dict(marker=dict(color=GREEN)),
                    decreasing=dict(marker=dict(color=RED)),
                    totals=dict(marker=dict(color=BLUE)),
                ))
                st.plotly_chart(chart_layout(fig_wf, 320), use_container_width=True)

        # Benchmark table
        st.markdown('<div class="sec-title">Benchmark Comparison</div>', unsafe_allow_html=True)
        achieved = [
            fin_s.get("PBT_Uplift_Pct", 0),
            fin_s.get("Revenue_Uplift_Pct", 0),
            fin_s.get("Debtor_Cycle_Improvement_Pct", 0),
            fin_s.get("Inventory_Reduction_Pct", 0),
        ]
        targets = [(10, 18), (8, 15), (25, 35), (20, 30)]
        bench = pd.DataFrame({
            "Metric":     ["PBT Uplift", "Revenue Turnover", "Debtor Cycle Savings", "Inventory Reduction"],
            "Target Min": ["10%", "8%", "25%", "20%"],
            "Target Max": ["18%", "15%", "35%", "30%"],
            "Achieved":   [f"{a:.1f}%" for a in achieved],
            "Status":     [
                "On Target" if t[0] <= a <= t[1] else ("Above Target" if a > t[1] else "Below Target")
                for a, t in zip(achieved, targets)
            ],
        })
        st.dataframe(bench, use_container_width=True, hide_index=True)


# ===============================================================================
# TAB 5 -- Production & Dispatch
# ===============================================================================
with tab5:
    if not st.session_state.run_complete:
        st.info("Run the pipeline to see production and dispatch results.")
    else:
        res    = st.session_state.results
        prod   = res.get("production", {})
        disp   = res.get("dispatch", {})
        prod_s = prod.get("summary", {})
        disp_s = disp.get("summary", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Plants Analysed",      str(prod_s.get("plants_analysed", 0))),                        unsafe_allow_html=True)
        c2.markdown(metric_card("Avg Plant Utilisation",f"{prod_s.get('avg_target_utilisation', 0):.1f}%", "Target ~82%", ""), unsafe_allow_html=True)
        c3.markdown(metric_card("Lead Time Reduction",  f"{disp_s.get('avg_lead_time_reduction_pct', 0):.1f}%", "Target 20–30%", "green"), unsafe_allow_html=True)
        c4.markdown(metric_card("Factory Issue Rate",   f"{prod_s.get('avg_factory_issue_rate', 0):.2f}%",  "Target < 8%", "orange"), unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="sec-title">Plant Utilisation</div>', unsafe_allow_html=True)
            util_df = prod.get("utilisation_summary_df", pd.DataFrame())
            if not util_df.empty:
                u = util_df.copy()
                u["Utilisation_%"] = (u["TargetUtilisation"] * 100).round(1)
                fig_util = px.bar(
                    u, x="PlantId", y="Utilisation_%",
                    text="Utilisation_%",
                    color_discrete_sequence=[BLUE],
                    height=300,
                )
                fig_util.add_hline(y=82, line_dash="dot", line_color=ORANGE,
                                   annotation_text="Target 82%", annotation_font_size=11)
                fig_util.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(chart_layout(fig_util, 300), use_container_width=True)

        with col_r:
            st.markdown('<div class="sec-title">Supply Network</div>', unsafe_allow_html=True)
            graph = disp.get("graph", None)
            if graph is not None and graph.number_of_nodes() > 0:
                try:
                    import networkx as nx
                    pos = nx.spring_layout(graph, seed=42, k=0.8)

                    edge_x, edge_y = [], []
                    for u, v in graph.edges():
                        if u in pos and v in pos:
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

                    node_x = [pos[n][0] for n in graph.nodes() if n in pos]
                    node_y = [pos[n][1] for n in graph.nodes() if n in pos]
                    node_l = [str(n) for n in graph.nodes() if n in pos]

                    fig_net = go.Figure()
                    fig_net.add_trace(go.Scatter(
                        x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=1, color="#d1d5db"), hoverinfo="none",
                    ))
                    fig_net.add_trace(go.Scatter(
                        x=node_x, y=node_y, mode="markers+text",
                        marker=dict(size=16, color=BLUE, line=dict(width=2, color="#fff")),
                        text=node_l, textposition="top center",
                        textfont=dict(size=8, color="#333"),
                        hoverinfo="text",
                    ))
                    fig_net.update_layout(
                        height=300, showlegend=False,
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        paper_bgcolor="#fff", plot_bgcolor="#fff",
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_net, use_container_width=True)
                except Exception as e:
                    st.warning(f"Network graph error: {e}")
            else:
                st.info("Network graph not available.")

        # Production schedule
        st.markdown('<div class="sec-title">Production Schedule</div>', unsafe_allow_html=True)
        schedule_df = prod.get("production_schedule_df", pd.DataFrame())
        if not schedule_df.empty:
            sched_cols = [
                "SkuId", "PlantId", "ProductGroup", "VelocityClass",
                "OptimalBatchSize", "TargetDailyProduction", "EstLeadTimeDays",
                "ProductionEfficiencyScore", "SchedulePriority",
            ]
            sched_show = schedule_df[
                [c for c in sched_cols if c in schedule_df.columns]
            ].sort_values("TargetDailyProduction", ascending=False).head(30)
            st.dataframe(sched_show.round(1), use_container_width=True, height=300)

        # Dispatch plan
        st.markdown('<div class="sec-title">Dispatch Plan — Lead Time Optimisation</div>', unsafe_allow_html=True)
        dispatch_plan = disp.get("dispatch_plan_df", pd.DataFrame())
        if not dispatch_plan.empty:
            disp_cols = [
                "SkuId", "StorageLocationId", "AvgDeliveryQty",
                "CurrentLeadTimeDays", "OptimisedLeadTimeDays",
                "LeadTimeReduction_Pct", "DispatchPriority", "Path",
            ]
            disp_show = dispatch_plan[
                [c for c in disp_cols if c in dispatch_plan.columns]
            ].sort_values("LeadTimeReduction_Pct", ascending=False).head(30)
            st.dataframe(disp_show.round(2), use_container_width=True, height=300)


# -- Footer --------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#aaa;font-size:0.72rem;margin:4px 0 16px">'
    'SupplyGraph (Wasi et al., AAAI 2024) &nbsp;&middot;&nbsp; '
    'M5 (Makridakis et al.) &nbsp;&middot;&nbsp; '
    'SAP BTP AI Core &nbsp;&middot;&nbsp; '
    'Namrath Basavaraju, MSc Data Science, University of Mannheim'
    '</p>',
    unsafe_allow_html=True,
)
