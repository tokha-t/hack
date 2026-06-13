from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.data_generator import ASTANA_LATITUDE, ASTANA_LONGITUDE


PRIORITY_COLORS = {
    "Critical": "#dc2626",
    "High": "#f97316",
    "Medium": "#eab308",
    "Skip": "#cbd5e1",
    "Depot": "#2563eb",
    "Route": "#0f172a",
    "Order": "#0f766e",
}


def priority_color(priority: str) -> str:
    return PRIORITY_COLORS.get(priority, PRIORITY_COLORS["Skip"])


def build_hover_text(row) -> str:
    predicted = row.get("predicted_fill_pct", 0)
    capacity = row.get("capacity_liters", "Unknown")
    activity = row.get("nearby_activity_score", "Unknown")
    return (
        f"<b>{row.get('bin_id', 'Unknown')}</b><br>"
        f"District: {row.get('district', 'Unknown')}<br>"
        f"Waste type: {row.get('waste_type', 'Unknown')}<br>"
        f"Predicted fill: {predicted:.1f}%<br>"
        f"Priority: {row.get('priority', 'Skip')}<br>"
        f"Capacity: {capacity} L<br>"
        f"Activity score: {activity}"
    )


def create_route_map(
    all_bins_df: pd.DataFrame,
    selected_bins_df: pd.DataFrame,
    route_points: list[dict],
    depot: dict,
    threshold: int | float,
) -> go.Figure:
    fig = go.Figure()

    map_center = {"lat": ASTANA_LATITUDE, "lon": ASTANA_LONGITUDE}
    if not all_bins_df.empty:
        map_center = {
            "lat": float(all_bins_df["latitude"].mean()),
            "lon": float(all_bins_df["longitude"].mean()),
        }

    selected_count = len(selected_bins_df)
    map_zoom = 11.05 if len(all_bins_df) > 120 else 11.35 if len(all_bins_df) > 75 else 12.0

    for priority in ["Skip", "Medium", "High", "Critical"]:
        priority_df = all_bins_df[all_bins_df["priority"] == priority] if not all_bins_df.empty else all_bins_df
        if priority_df.empty:
            continue

        if priority == "Skip":
            marker_size = (priority_df["predicted_fill_pct"].clip(10, 75) / 14 + 3.5).tolist()
            opacity = 0.24
        else:
            marker_size = (priority_df["predicted_fill_pct"].clip(35, 100) / 8.5 + 5).tolist()
            opacity = 0.9

        fig.add_trace(
            go.Scattermapbox(
                lat=priority_df["latitude"],
                lon=priority_df["longitude"],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": priority_color(priority),
                    "opacity": opacity,
                },
                name=f"{priority} bins",
                text=[build_hover_text(row) for _, row in priority_df.iterrows()],
                hoverinfo="text",
                showlegend=False,
            )
        )

    if len(route_points) > 1:
        fig.add_trace(
            go.Scattermapbox(
                lat=[point["latitude"] for point in route_points],
                lon=[point["longitude"] for point in route_points],
                mode="lines",
                line={"width": 5, "color": PRIORITY_COLORS["Route"]},
                name="Optimized route",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        stop_points = [point for point in route_points[1:-1]]
        show_labels = selected_count <= 24
        fig.add_trace(
            go.Scattermapbox(
                lat=[point["latitude"] for point in stop_points],
                lon=[point["longitude"] for point in stop_points],
                mode="markers+text" if show_labels else "markers",
                marker={
                    "size": 9,
                    "color": PRIORITY_COLORS["Order"],
                    "opacity": 0.86,
                },
                text=[str(index) for index, _ in enumerate(stop_points, start=1)] if show_labels else None,
                textposition="top center",
                textfont={"size": 10, "color": "#0f172a"},
                name="Route order",
                hovertext=[
                    f"Stop {index}: {point.get('bin_id', 'Unknown')}<br>"
                    f"{point.get('district', 'Unknown')} · {point.get('priority', 'Unknown')}"
                    for index, point in enumerate(stop_points, start=1)
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=[depot["latitude"]],
            lon=[depot["longitude"]],
            mode="markers",
            marker={
                "size": 17,
                "color": PRIORITY_COLORS["Depot"],
            },
            name="Depot",
            text=["Depot"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        mapbox={
            "style": "carto-positron",
            "center": map_center,
            "zoom": map_zoom,
        },
        height=720,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
    )

    return fig
