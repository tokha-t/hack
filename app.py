from __future__ import annotations

import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_generator import (
    ASTANA_LATITUDE,
    ASTANA_LONGITUDE,
    BINS_PATH,
    DISTRICTS,
    WASTE_TYPES,
    ensure_data_exists,
)
from src.map_utils import create_route_map
from src.predict import assign_priority, predict_fill_levels
from src.routing import compare_routes
from src.savings import calculate_savings
from src.train_model import METRICS_PATH, MODEL_PATH, train_and_save_model


DEPOT = {"latitude": ASTANA_LATITUDE, "longitude": ASTANA_LONGITUDE}

PRIORITY_COLORS = {
    "Critical": "#dc2626",
    "High": "#f97316",
    "Medium": "#eab308",
    "Skip": "#cbd5e1",
}

CHART_COLORS = {
    "fixed": "#334155",
    "greedy": "#10b981",
    "optimized": "#0ea5e9",
    "blue": "#0ea5e9",
    "green": "#10b981",
    "surface": "#ffffff",
    "surface_2": "#f8fafc",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "text": "#0f172a",
}

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


st.set_page_config(
    page_title="EcoRoute AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, .stDeployButton, [data-testid="stAppDeployButton"] {
            display: none !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(20, 184, 166, 0.18), transparent 32rem),
                radial-gradient(circle at 88% 4%, rgba(14, 165, 233, 0.16), transparent 30rem),
                linear-gradient(180deg, #f4fbf8 0%, #f7fafc 44%, #eef5f7 100%);
            color: #0f172a;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(236, 253, 245, 0.98) 0%, rgba(240, 249, 255, 0.98) 56%, #ffffff 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.26);
            box-shadow: 12px 0 36px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #0f3f3c;
            letter-spacing: 0 !important;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #334155;
        }

        .sidebar-note {
            margin-top: 1rem;
            padding: 0.95rem;
            border: 1px solid rgba(20, 184, 166, 0.24);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.78);
            color: #475467;
            font-size: 0.84rem;
            line-height: 1.38;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }

        .sidebar-note strong {
            display: block;
            margin-bottom: 0.25rem;
            color: #0f766e;
            font-size: 0.9rem;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1360px;
        }

        h1, h2, h3 {
            color: #0f172a;
        }

        h1 {
            font-size: 2.35rem !important;
            line-height: 1.08 !important;
            letter-spacing: 0 !important;
            margin-bottom: 0.6rem !important;
        }

        h2 {
            font-size: 1.6rem !important;
            line-height: 1.18 !important;
        }

        h3 {
            font-size: 1.34rem !important;
            line-height: 1.22 !important;
        }

        .eco-hero {
            position: relative;
            overflow: hidden;
            margin: 0.15rem 0 1rem 0;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 12px;
            background:
                linear-gradient(135deg, rgba(6, 78, 59, 0.97) 0%, rgba(15, 118, 110, 0.96) 55%, rgba(14, 165, 233, 0.92) 100%);
            color: #ffffff;
            box-shadow: 0 22px 54px rgba(15, 76, 92, 0.22);
        }

        .eco-hero::before {
            content: "";
            position: absolute;
            inset: -44% -10% auto auto;
            width: 28rem;
            height: 28rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
        }

        .eco-hero::after {
            content: "";
            position: absolute;
            right: 1.1rem;
            bottom: -2.2rem;
            width: 18rem;
            height: 18rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background:
                radial-gradient(circle, rgba(255, 255, 255, 0.15) 0 2px, transparent 2px);
            background-size: 18px 18px;
            opacity: 0.55;
        }

        .eco-hero-content {
            position: relative;
            z-index: 1;
            max-width: 980px;
        }

        .eco-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.75rem;
            padding: 0.34rem 0.66rem;
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.13);
            color: #e0fbf5;
            font-size: 0.78rem;
            font-weight: 760;
        }

        .eco-hero h1 {
            margin: 0 !important;
            color: #ffffff !important;
            font-size: 2.45rem !important;
            line-height: 1.04 !important;
            letter-spacing: 0 !important;
        }

        .eco-hero p {
            max-width: 760px;
            margin: 0.65rem 0 0 0;
            color: rgba(240, 253, 250, 0.9);
            font-size: 1.02rem;
            line-height: 1.55;
        }

        .hero-stats {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            max-width: 760px;
            margin-top: 1.15rem;
        }

        .hero-stat {
            padding: 0.78rem 0.86rem;
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.13);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14);
        }

        .hero-stat-value {
            color: #ffffff;
            font-size: 1.18rem;
            font-weight: 820;
            line-height: 1.08;
        }

        .hero-stat-label {
            margin-top: 0.24rem;
            color: rgba(240, 253, 250, 0.78);
            font-size: 0.76rem;
            font-weight: 650;
        }

        [data-testid="stMetric"] {
            padding: 0.92rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-top: 3px solid #14b8a6;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stMetricLabel"] p {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 820;
        }

        [data-testid="stMetricDelta"] p {
            font-weight: 760;
        }

        .eco-track {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0.25rem 0 1rem 0;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.84);
            color: #0f766e;
            border: 1px solid rgba(20, 184, 166, 0.24);
            font-size: 0.86rem;
            font-weight: 720;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        }

        .eco-pipeline {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 1.3rem 0;
        }

        .eco-step {
            position: relative;
            padding: 0.95rem 0.98rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
        }

        .eco-step::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0.98rem;
            right: 0.98rem;
            height: 3px;
            border-radius: 0 0 999px 999px;
            background: linear-gradient(90deg, #14b8a6, #0ea5e9);
        }

        .eco-step strong {
            display: block;
            color: #0f172a;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }

        .eco-step span {
            color: #64748b;
            font-size: 0.82rem;
        }

        .scenario-card {
            min-height: 168px;
            padding: 1.02rem;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(248, 250, 252, 0.96) 100%);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }

        .scenario-card h4 {
            margin: 0 0 0.35rem 0;
            color: #0f172a;
        }

        .scenario-card .scenario-meta {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.85rem;
        }

        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem 0.75rem;
        }

        .scenario-value {
            font-size: 1.25rem;
            font-weight: 820;
            color: #0f766e;
        }

        .scenario-label {
            font-size: 0.75rem;
            color: #64748b;
        }

        .map-panel {
            padding: 1.05rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(240, 253, 250, 0.78) 100%);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
            margin: 0.8rem 0 0.6rem 0;
        }

        .map-panel h3 {
            margin: 0 0 0.25rem 0;
            color: #0f172a;
        }

        .map-panel p {
            margin: 0 0 0.8rem 0;
            color: #64748b;
        }

        .map-legend {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.55rem 1rem;
            margin: 0.4rem 0 0.95rem 0;
            color: #475569;
            font-size: 0.82rem;
        }

        .map-legend span {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .legend-dot {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 999px;
            display: inline-block;
            border: 1px solid rgba(15, 23, 42, 0.12);
        }

        .legend-line {
            width: 1.1rem;
            height: 0.22rem;
            border-radius: 999px;
            display: inline-block;
            background: #0f172a;
        }

        .legend-skip { background: #cbd5e1; }
        .legend-medium { background: #eab308; }
        .legend-high { background: #f97316; }
        .legend-critical { background: #dc2626; }
        .legend-depot { background: #2563eb; }

        .map-stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
        }

        .map-stat {
            padding: 0.78rem 0.86rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .map-stat-value {
            font-size: 1.35rem;
            line-height: 1.1;
            font-weight: 820;
            color: #0f172a;
        }

        .map-stat-label {
            margin-top: 0.28rem;
            font-size: 0.78rem;
            color: #64748b;
        }

        [data-testid="stPlotlyChart"] {
            padding: 0.5rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stAlert"],
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div {
            color: #0f172a !important;
        }

        .stDownloadButton > button {
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, #0f766e, #0ea5e9);
            color: #ffffff;
            font-weight: 760;
            box-shadow: 0 12px 28px rgba(14, 165, 233, 0.18);
        }

        .stDownloadButton > button:hover {
            color: #ffffff;
            border: 0;
            filter: brightness(1.03);
        }

        div[data-baseweb="select"] > div,
        [data-testid="stSlider"] {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 1.25rem;
            }

            h1 {
                font-size: 2.15rem !important;
            }

            h2 {
                font-size: 1.38rem !important;
            }

            h3 {
                font-size: 1.16rem !important;
            }

            .eco-hero {
                padding: 1.1rem;
            }

            .eco-hero h1 {
                font-size: 2rem !important;
            }

            .hero-stats {
                grid-template-columns: 1fr;
            }

            .eco-pipeline {
                grid-template-columns: 1fr;
            }

            .map-stat-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig, height: int = 360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={
            "family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            "size": 12,
            "color": CHART_COLORS["text"],
        },
        title={
            "font": {"size": 17, "color": "#172033"},
            "x": 0.02,
            "xanchor": "left",
        },
        margin={"l": 44, "r": 28, "t": 58, "b": 46},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=CHART_COLORS["grid"],
        zeroline=False,
        linecolor=CHART_COLORS["axis"],
        tickfont={"color": "#667085"},
        title_font={"color": "#475467"},
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor=CHART_COLORS["axis"],
        tickfont={"color": "#667085"},
        title_font={"color": "#475467"},
    )
    return fig


def stretch_plotly_chart(fig, config: dict | None = None) -> None:
    chart_config = {**PLOTLY_CONFIG, **(config or {})}
    try:
        st.plotly_chart(fig, width="stretch", config=chart_config)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=chart_config)


def stretch_dataframe(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def stretch_download_button(label: str, **kwargs) -> bool:
    try:
        return st.download_button(label, width="stretch", **kwargs)
    except TypeError:
        return st.download_button(label, use_container_width=True, **kwargs)


@st.cache_data(show_spinner=False)
def load_bins() -> pd.DataFrame:
    return pd.read_csv(BINS_PATH)


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def model_feature_importance_chart():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .assign(
                feature=lambda df: df["feature"]
                .str.replace("categorical__", "", regex=False)
                .str.replace("numeric__", "", regex=False)
            )
            .sort_values("importance", ascending=False)
            .head(10)
            .sort_values("importance", ascending=True)
        )
        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 model features",
            labels={"importance": "Importance", "feature": "Feature"},
            color_discrete_sequence=[CHART_COLORS["blue"]],
        )
        fig.update_traces(
            marker_line_width=0,
            hovertemplate="%{y}<br>Importance %{x:.3f}<extra></extra>",
        )
        return apply_chart_theme(fig, height=420)
    except Exception as exc:
        st.info(f"Feature importance is unavailable right now: {exc}")
        return None


def render_kpis(savings: dict, predicted_df: pd.DataFrame) -> None:
    avg_fill = predicted_df["predicted_fill_pct"].mean() if not predicted_df.empty else 0
    cost_saved = savings["estimated_fuel_cost_saved_kzt"]
    cost_label = f"{cost_saved / 1000:.1f}k KZT" if cost_saved >= 1000 else f"{cost_saved:,.0f} KZT"
    kpis = [
        ("Total bins", f"{savings['total_bins']:,}", None),
        ("Bins selected", f"{savings['selected_bins']:,}", None),
        ("Bins skipped", f"{savings['bins_skipped']:,}", f"{savings['skipped_percent']:.1f}%"),
        ("Avg predicted fill", f"{avg_fill:.1f}%", None),
        ("Optimized route", f"{savings['optimized_route_distance_km']:.1f} km", None),
        ("Distance saved", f"{savings['distance_saved_km']:.1f} km", f"{savings['distance_saved_percent']:.1f}%"),
        ("Time saved", f"{savings['estimated_total_time_saved_minutes']:.0f} min", None),
        ("Fuel saved", f"{savings['estimated_fuel_saved_liters']:.1f} L", None),
        ("CO₂ saved", f"{savings['estimated_co2_saved_kg']:.1f} kg", None),
        ("Cost saved", cost_label, None),
    ]

    for start in range(0, len(kpis), 4):
        columns = st.columns(4)
        for column, (label, value, delta) in zip(columns, kpis[start : start + 4]):
            column.metric(label, value, delta=delta)


def render_recommendation(selected_bins_df: pd.DataFrame, predicted_df: pd.DataFrame, savings: dict) -> None:
    if selected_bins_df.empty:
        st.info(
            "No bins exceed the current threshold. The city can skip this collection cycle or lower the threshold."
        )
        return

    top_districts = selected_bins_df["district"].value_counts().head(2).index.tolist()
    district_phrase = " and ".join(top_districts) if top_districts else "the filtered area"
    recommendation = (
        f"Today, EcoRoute AI recommends collecting {len(selected_bins_df)} out of {len(predicted_df)} bins, "
        f"prioritizing {district_phrase} districts. Estimated savings: "
        f"{savings['distance_saved_km']:.1f} km, "
        f"{savings['estimated_total_time_saved_minutes']:.0f} minutes, "
        f"{savings['estimated_fuel_saved_liters']:.1f} liters of fuel, "
        f"{savings['estimated_co2_saved_kg']:.1f} kg CO₂, and "
        f"{savings['estimated_fuel_cost_saved_kzt']:,.0f} KZT."
    )

    if (selected_bins_df["priority"] == "Critical").any():
        st.warning(recommendation)
    else:
        st.success(recommendation)


def render_critical_alert(selected_bins_df: pd.DataFrame) -> None:
    if selected_bins_df.empty or "priority" not in selected_bins_df.columns:
        return

    critical_counts = selected_bins_df[selected_bins_df["priority"] == "Critical"]["district"].value_counts()
    if critical_counts.empty:
        return

    top_district = critical_counts.index[0]
    top_count = int(critical_counts.iloc[0])
    if top_count >= 5:
        st.warning(
            f"Alert: {top_district} district has {top_count} critical bins and should be prioritized today."
        )


def render_hero(threshold: int | float, district_filter: str, waste_type_filter: str) -> None:
    district_label = district_filter if district_filter != "All" else "All districts"
    waste_label = waste_type_filter if waste_type_filter != "All" else "All waste streams"
    st.markdown(
        f"""
        <section class="eco-hero">
            <div class="eco-hero-content">
                <div class="eco-eyebrow">SmartScape Hackathon · Ecology & Urban Environment</div>
                <h1>EcoRoute AI</h1>
                <p>
                    Predict near-full bins, prioritize the right stops, and build a cleaner truck route
                    for one Astana dispatch zone before the shift starts.
                </p>
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-value">{threshold}%</div>
                    <div class="hero-stat-label">collection threshold</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">{district_label}</div>
                    <div class="hero-stat-label">district filter</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">{waste_label}</div>
                    <div class="hero-stat-label">waste filter</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline(threshold: int | float) -> None:
    st.markdown(
        f"""
        <div class="eco-track">Live operations pipeline · Astana demo</div>
        <div class="eco-pipeline">
            <div class="eco-step">
                <strong>1. Predict</strong>
                <span>RandomForestRegressor forecasts bin fill percentage.</span>
            </div>
            <div class="eco-step">
                <strong>2. Prioritize</strong>
                <span>Bins at or above {threshold}% enter today's collection plan.</span>
            </div>
            <div class="eco-step">
                <strong>3. Optimize</strong>
                <span>Nearest-neighbor route is improved with 2-opt.</span>
            </div>
            <div class="eco-step">
                <strong>4. Quantify</strong>
                <span>Distance, time, fuel, CO₂, and cost savings are estimated.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scenario_metrics(predicted_df: pd.DataFrame, scenario_threshold: int) -> dict:
    scenario_df = assign_priority(predicted_df, threshold=scenario_threshold)
    scenario_selected_df = scenario_df[scenario_df["predicted_fill_pct"] >= scenario_threshold].copy()
    scenario_routes = compare_routes(scenario_df, scenario_selected_df, DEPOT)
    return calculate_savings(scenario_df, scenario_selected_df, scenario_routes)


def render_scenario_cards(predicted_df: pd.DataFrame) -> None:
    st.subheader("Scenario comparison")
    scenarios = [
        ("Conservative", 85, "Collect only the most urgent bins."),
        ("Balanced", 75, "Default operating plan for the demo."),
        ("Aggressive", 65, "Collect earlier to reduce overflow risk."),
    ]
    columns = st.columns(3)

    for column, (name, scenario_threshold, note) in zip(columns, scenarios):
        metrics = scenario_metrics(predicted_df, scenario_threshold)
        with column:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <h4>{name}</h4>
                    <div class="scenario-meta">{scenario_threshold}% threshold · {note}</div>
                    <div class="scenario-grid">
                        <div>
                            <div class="scenario-value">{metrics["selected_bins"]}</div>
                            <div class="scenario-label">bins selected</div>
                        </div>
                        <div>
                            <div class="scenario-value">{metrics["distance_saved_km"]:.1f}</div>
                            <div class="scenario-label">km saved</div>
                        </div>
                        <div>
                            <div class="scenario-value">{metrics["estimated_total_time_saved_minutes"]:.0f}</div>
                            <div class="scenario-label">minutes saved</div>
                        </div>
                        <div>
                            <div class="scenario-value">{metrics["estimated_co2_saved_kg"]:.1f}</div>
                            <div class="scenario-label">kg CO₂ saved</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_map_context(
    predicted_df: pd.DataFrame,
    selected_bins_df: pd.DataFrame,
    savings: dict,
    route_comparison: dict,
) -> None:
    selected_percent = (len(selected_bins_df) / len(predicted_df) * 100) if len(predicted_df) else 0
    st.markdown(
        f"""
        <div class="map-panel">
            <h3>Optimized collection map</h3>
            <p>
                Dense Astana dispatch view with {len(predicted_df)} bins. Skipped bins are intentionally faint,
                selected bins stay vivid, and the route is drawn above the map so the truck path remains readable.
            </p>
            <div class="map-legend">
                <span><i class="legend-line"></i>Optimized route</span>
                <span><i class="legend-dot legend-depot"></i>Depot</span>
                <span><i class="legend-dot legend-critical"></i>Critical</span>
                <span><i class="legend-dot legend-high"></i>High</span>
                <span><i class="legend-dot legend-medium"></i>Medium</span>
                <span><i class="legend-dot legend-skip"></i>Skipped</span>
            </div>
            <div class="map-stat-grid">
                <div class="map-stat">
                    <div class="map-stat-value">{savings["fixed_route_distance_km"]:.1f} km</div>
                    <div class="map-stat-label">fixed all-bin route</div>
                </div>
                <div class="map-stat">
                    <div class="map-stat-value">{savings["optimized_route_distance_km"]:.1f} km</div>
                    <div class="map-stat-label">optimized selected route</div>
                </div>
                <div class="map-stat">
                    <div class="map-stat-value">{selected_percent:.0f}%</div>
                    <div class="map-stat-label">bins selected today</div>
                </div>
                <div class="map-stat">
                    <div class="map-stat-value">{route_comparison["two_opt_improvement_km"]:.1f} km</div>
                    <div class="map-stat-label">2-opt route improvement</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def route_points_to_dataframe(route_points: list[dict]) -> pd.DataFrame:
    rows = []
    for index, point in enumerate(route_points, start=1):
        rows.append(
            {
                "stop_order": index,
                "bin_id": point.get("bin_id", "Depot"),
                "district": point.get("district", "Depot"),
                "priority": point.get("priority", "Depot"),
                "latitude": point.get("latitude"),
                "longitude": point.get("longitude"),
            }
        )
    return pd.DataFrame(rows)


inject_styles()

with st.sidebar:
    st.header("Operations Control")
    threshold = st.slider("Collection threshold (%)", min_value=50, max_value=95, value=75, step=5)
    district_filter = st.selectbox("District", ["All"] + DISTRICTS)
    waste_type_filter = st.selectbox("Waste type", ["All"] + WASTE_TYPES)
    st.markdown(
        """
        <div class="sidebar-note">
            <strong>Demo scale</strong>
            180 live bins represent one dispatch zone for a truck shift. The model trains on
            4,500 synthetic historical observations so the demo stays fast and stable.
        </div>
        """,
        unsafe_allow_html=True,
    )


render_hero(threshold, district_filter, waste_type_filter)
render_pipeline(threshold)

with st.spinner("Preparing data and model..."):
    ensure_data_exists()
    if not MODEL_PATH.exists():
        train_and_save_model()

bins_df = load_bins()
all_predicted_df = predict_fill_levels(bins_df, threshold=threshold)
predicted_df = all_predicted_df.copy()
if district_filter != "All":
    predicted_df = predicted_df[predicted_df["district"] == district_filter]
if waste_type_filter != "All":
    predicted_df = predicted_df[predicted_df["waste_type"] == waste_type_filter]

selected_bins_df = predicted_df[predicted_df["predicted_fill_pct"] >= threshold].copy()
route_comparison = compare_routes(predicted_df, selected_bins_df, DEPOT)
savings = calculate_savings(predicted_df, selected_bins_df, route_comparison)
route_order_df = route_points_to_dataframe(route_comparison["route_points_optimized"])

metrics = load_metrics()
if metrics:
    st.caption(
        f"Model: {metrics.get('model_name', 'RandomForestRegressor')} | "
        f"MAE {metrics.get('mae')} | RMSE {metrics.get('rmse')} | R² {metrics.get('r2')}"
    )

render_recommendation(selected_bins_df, predicted_df, savings)
render_critical_alert(selected_bins_df)
render_kpis(savings, predicted_df)

render_map_context(predicted_df, selected_bins_df, savings, route_comparison)
stretch_plotly_chart(
    create_route_map(
        predicted_df,
        selected_bins_df,
        route_comparison["route_points_optimized"],
        DEPOT,
        threshold,
    )
)

chart_col_1, chart_col_2 = st.columns(2)

with chart_col_1:
    route_chart_df = pd.DataFrame(
        {
            "Route": ["Fixed route", "Selected greedy route", "Selected 2-opt route"],
            "Distance (km)": [
                route_comparison["fixed_route_distance_km"],
                route_comparison["selected_greedy_distance_km"],
                route_comparison["selected_optimized_distance_km"],
            ],
        }
    )
    fig_route = px.bar(
        route_chart_df,
        x="Distance (km)",
        y="Route",
        orientation="h",
        title="Route distance comparison",
        color="Route",
        text="Distance (km)",
        color_discrete_map={
            "Fixed route": CHART_COLORS["fixed"],
            "Selected greedy route": CHART_COLORS["greedy"],
            "Selected 2-opt route": CHART_COLORS["optimized"],
        },
    )
    fig_route.update_traces(
        marker_line_width=0,
        texttemplate="%{x:.1f} km",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.1f} km<extra></extra>",
    )
    fig_route.update_layout(showlegend=False)
    fig_route.update_yaxes(
        categoryorder="array",
        categoryarray=["Selected 2-opt route", "Selected greedy route", "Fixed route"],
    )
    apply_chart_theme(fig_route, height=360)
    fig_route.update_yaxes(title_text=None)
    fig_route.update_layout(margin={"l": 160, "r": 42, "t": 58, "b": 46})
    stretch_plotly_chart(fig_route)

with chart_col_2:
    fig_distribution = px.histogram(
        predicted_df,
        x="predicted_fill_pct",
        nbins=18,
        title="Predicted fill level distribution",
        labels={"predicted_fill_pct": "Predicted fill (%)", "count": "Bins"},
        color_discrete_sequence=[CHART_COLORS["blue"]],
    )
    fig_distribution.add_vrect(
        x0=threshold,
        x1=100,
        fillcolor="#ecfdf3",
        opacity=0.55,
        layer="below",
        line_width=0,
    )
    fig_distribution.add_vline(
        x=threshold,
        line_dash="dash",
        line_width=2,
        line_color=PRIORITY_COLORS["Critical"],
        annotation_text=f"{threshold}% threshold",
        annotation_position="top right",
    )
    fig_distribution.update_traces(
        marker_line_color="#ffffff",
        marker_line_width=1,
        opacity=0.9,
        hovertemplate="Predicted fill %{x}<br>Bins %{y}<extra></extra>",
    )
    apply_chart_theme(fig_distribution, height=360)
    fig_distribution.update_yaxes(title_text="Bins")
    stretch_plotly_chart(fig_distribution)

chart_col_3, chart_col_4 = st.columns(2)

with chart_col_3:
    if selected_bins_df.empty:
        st.info("No selected bins for the district workload chart at the current threshold.")
    else:
        district_workload = (
            selected_bins_df.groupby("district", as_index=False)
            .size()
            .rename(columns={"size": "selected_bins"})
            .sort_values("selected_bins", ascending=True)
        )
        fig_district = px.bar(
            district_workload,
            x="selected_bins",
            y="district",
            orientation="h",
            title="Selected bins by district",
            labels={"district": "District", "selected_bins": "Selected bins"},
            text="selected_bins",
            color_discrete_sequence=[CHART_COLORS["greedy"]],
        )
        fig_district.update_traces(
            marker_line_width=0,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x} selected bins<extra></extra>",
        )
        apply_chart_theme(fig_district, height=360)
        fig_district.update_yaxes(title_text=None)
        fig_district.update_layout(margin={"l": 120, "r": 36, "t": 58, "b": 46})
        stretch_plotly_chart(fig_district)

with chart_col_4:
    priority_summary = (
        predicted_df["priority"]
        .value_counts()
        .reindex(["Critical", "High", "Medium", "Skip"], fill_value=0)
        .reset_index()
    )
    priority_summary.columns = ["priority", "count"]
    priority_summary = priority_summary.sort_values("count", ascending=True)
    fig_priority = px.bar(
        priority_summary,
        x="count",
        y="priority",
        orientation="h",
        title="Priority summary",
        labels={"priority": "Priority", "count": "Bins"},
        text="count",
        color="priority",
        color_discrete_map=PRIORITY_COLORS,
    )
    fig_priority.update_traces(
        marker_line_width=0,
        textposition="outside",
        textfont={"color": CHART_COLORS["text"]},
        cliponaxis=False,
        hovertemplate="%{y}<br>%{x} bins<extra></extra>",
    )
    apply_chart_theme(fig_priority, height=360)
    fig_priority.update_yaxes(title_text=None)
    fig_priority.update_layout(showlegend=False, margin={"l": 96, "r": 36, "t": 58, "b": 46})
    stretch_plotly_chart(fig_priority)

render_scenario_cards(predicted_df)

with st.expander("Why 180 bins and 4,500 training rows?"):
    st.markdown(
        """
        - `180` current bins is a one-shift dispatch zone, not every bin in the whole city. It is dense enough
          to prove routing savings, but still readable on one map.
        - `4,500` training rows are historical-style observations. More synthetic rows can make training slower
          without adding much signal once the model has learned the feature patterns.
        - Hundreds or thousands of live bins become a fleet-planning problem: split by district, truck capacity,
          shift window, and depot, then optimize each truck route separately. One giant route would be slower,
          harder to read, and less realistic for operations.
        """
    )

with st.expander("Model internals: feature importance"):
    feature_fig = model_feature_importance_chart()
    if feature_fig is not None:
        stretch_plotly_chart(feature_fig)

st.subheader("Selected bins for collection")
selected_table_columns = [
    "bin_id",
    "district",
    "waste_type",
    "predicted_fill_pct",
    "priority",
    "latitude",
    "longitude",
]
download_col_1, download_col_2 = st.columns(2)
with download_col_1:
    stretch_download_button(
        "Download selected bins CSV",
        data=selected_bins_df[selected_table_columns].to_csv(index=False).encode("utf-8"),
        file_name="ecoroute_selected_bins.csv",
        mime="text/csv",
    )
with download_col_2:
    stretch_download_button(
        "Download route order CSV",
        data=route_order_df.to_csv(index=False).encode("utf-8"),
        file_name="ecoroute_route_order.csv",
        mime="text/csv",
    )

stretch_dataframe(
    selected_bins_df[selected_table_columns].sort_values("predicted_fill_pct", ascending=False),
    hide_index=True,
)

with st.expander("Truck route order"):
    stretch_dataframe(
        route_order_df,
        hide_index=True,
    )

with st.expander("All predicted bins"):
    stretch_dataframe(
        predicted_df[selected_table_columns].sort_values("predicted_fill_pct", ascending=False),
        hide_index=True,
    )

st.info(
    "How EcoRoute AI works: 1. Predict fill level using ML. 2. Select bins above the collection threshold. "
    "3. Build a route using nearest-neighbor. 4. Improve it with 2-opt. 5. Estimate distance, time, fuel, "
    "cost, and CO₂ savings."
)
