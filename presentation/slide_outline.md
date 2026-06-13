# EcoRoute AI — Presentation Outline

## Slide 1 — Title
EcoRoute AI  
Predictive Waste Collection & Route Optimization for Smart Cities  
Track 2 — Ecology & Urban Environment

## Slide 2 — Problem
Fixed garbage truck routes waste fuel, time, and money because many bins are collected before they are full. Busy areas can still overflow because collection is not demand-based.

## Slide 3 — Solution
EcoRoute AI predicts bin fill levels, selects bins that need service, optimizes the truck route, and shows operational savings.

## Slide 4 — How It Works
Bin data → ML model → threshold filter → route optimizer → savings dashboard. The dashboard shows this as a four-step decision pipeline: predict, prioritize, optimize, and quantify.

## Slide 5 — AI/ML Methodology
RandomForestRegressor predicts fill percentage using previous fill, time since collection, district, activity score, day of week, weather, and waste type.  
Show MAE, RMSE, and R² from `models/metrics.json`.

## Slide 6 — Route Optimization Methodology
Fixed baseline route visits all bins sorted by `bin_id`. Nearest-neighbor creates an initial selected-bin route. 2-opt improves the route by removing inefficient path crossings. Haversine distance estimates geographic distance.

## Slide 7 — Demo and Results
Show the 180-bin interactive map, selected bins, optimized route, city manager recommendation, critical district alert, scenario comparison cards, downloadable route order, distance saved, time saved, fuel saved, CO₂ saved, and cost saved.

## Slide 8 — Deployment and Future
Can integrate with smart-bin sensors, municipal trucks, real-time traffic, and city dashboards. Future work includes multi-truck dispatch, IoT integration, live alerts, automated municipal work orders, and traffic-aware routing.
