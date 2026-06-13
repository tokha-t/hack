from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance between two coordinates in kilometers."""
    earth_radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return earth_radius_km * c


def _point_lat(point) -> float:
    if isinstance(point, dict):
        return float(point["latitude"])
    return float(point[0])


def _point_lon(point) -> float:
    if isinstance(point, dict):
        return float(point["longitude"])
    return float(point[1])


def route_distance(route_points: list[dict] | list[tuple]) -> float:
    if len(route_points) < 2:
        return 0.0

    total_distance = 0.0
    for start, end in zip(route_points[:-1], route_points[1:]):
        total_distance += haversine_distance(
            _point_lat(start),
            _point_lon(start),
            _point_lat(end),
            _point_lon(end),
        )
    return total_distance


def _depot_point(depot: dict) -> dict:
    return {
        "bin_id": "Depot",
        "latitude": float(depot["latitude"]),
        "longitude": float(depot["longitude"]),
        "district": "Depot",
        "waste_type": "Depot",
        "priority": "Depot",
        "predicted_fill_pct": None,
    }


def _bin_points(df: pd.DataFrame) -> list[dict]:
    points = []
    for _, row in df.iterrows():
        points.append(
            {
                "bin_id": row.get("bin_id"),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "district": row.get("district", ""),
                "waste_type": row.get("waste_type", ""),
                "priority": row.get("priority", ""),
                "predicted_fill_pct": row.get("predicted_fill_pct"),
            }
        )
    return points


def fixed_route_all_bins(all_bins_df: pd.DataFrame, depot: dict) -> list[dict]:
    depot_route_point = _depot_point(depot)
    if all_bins_df.empty:
        return [depot_route_point]

    sorted_bins = all_bins_df.sort_values("bin_id", ascending=True)
    return [depot_route_point] + _bin_points(sorted_bins) + [depot_route_point.copy()]


def nearest_neighbor_route(selected_bins_df: pd.DataFrame, depot: dict) -> list[dict]:
    depot_route_point = _depot_point(depot)
    if selected_bins_df.empty:
        return [depot_route_point]

    unvisited = _bin_points(selected_bins_df)
    current = depot_route_point
    route = [depot_route_point]

    while unvisited:
        next_index, next_point = min(
            enumerate(unvisited),
            key=lambda item: haversine_distance(
                current["latitude"],
                current["longitude"],
                item[1]["latitude"],
                item[1]["longitude"],
            ),
        )
        route.append(next_point)
        current = next_point
        unvisited.pop(next_index)

    route.append(depot_route_point.copy())
    return route


def two_opt(route_points: list[dict], max_iterations: int = 100) -> list[dict]:
    if len(route_points) <= 4:
        return route_points

    best_route = route_points[:]
    iteration = 0
    improved = True

    # Production deployments could run 2-opt with a time budget instead of a fixed iteration cap.
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(best_route) - 2):
            for k in range(i + 1, len(best_route) - 1):
                a = best_route[i - 1]
                b = best_route[i]
                c = best_route[k]
                d = best_route[k + 1]
                current_edges = haversine_distance(
                    _point_lat(a), _point_lon(a), _point_lat(b), _point_lon(b)
                ) + haversine_distance(
                    _point_lat(c), _point_lon(c), _point_lat(d), _point_lon(d)
                )
                proposed_edges = haversine_distance(
                    _point_lat(a), _point_lon(a), _point_lat(c), _point_lon(c)
                ) + haversine_distance(
                    _point_lat(b), _point_lon(b), _point_lat(d), _point_lon(d)
                )
                if proposed_edges + 1e-9 < current_edges:
                    best_route[i : k + 1] = reversed(best_route[i : k + 1])
                    improved = True
                    break
            if improved:
                break

    return best_route


def compare_routes(all_bins_df: pd.DataFrame, selected_bins_df: pd.DataFrame, depot: dict) -> dict:
    route_points_fixed = fixed_route_all_bins(all_bins_df, depot)
    route_points_greedy = nearest_neighbor_route(selected_bins_df, depot)
    route_points_optimized = (
        two_opt(route_points_greedy, max_iterations=100)
        if len(selected_bins_df) > 2
        else route_points_greedy
    )

    fixed_distance = route_distance(route_points_fixed)
    greedy_distance = route_distance(route_points_greedy)
    optimized_distance = route_distance(route_points_optimized)

    distance_saved = max(0.0, fixed_distance - optimized_distance)
    two_opt_improvement = max(0.0, greedy_distance - optimized_distance)

    return {
        "fixed_route_distance_km": round(fixed_distance, 3),
        "selected_greedy_distance_km": round(greedy_distance, 3),
        "selected_optimized_distance_km": round(optimized_distance, 3),
        "distance_saved_km": round(distance_saved, 3),
        "distance_saved_percent": round((distance_saved / fixed_distance * 100) if fixed_distance else 0.0, 2),
        "two_opt_improvement_km": round(two_opt_improvement, 3),
        "two_opt_improvement_percent": round(
            (two_opt_improvement / greedy_distance * 100) if greedy_distance else 0.0,
            2,
        ),
        "route_points_fixed": route_points_fixed,
        "route_points_greedy": route_points_greedy,
        "route_points_optimized": route_points_optimized,
    }
