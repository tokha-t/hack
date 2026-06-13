# EcoRoute AI

Predictive Waste Collection & Route Optimization for Smart Cities

## Track

Track 2 — Ecology & Urban Environment

## Problem

Cities often use fixed waste collection routes, which means trucks visit bins that may only be partially full. This wastes fuel, time, driver resources, municipal budget, and increases emissions. At the same time, high-demand areas may overflow if they are not prioritized.

## Solution

EcoRoute AI predicts waste bin fill levels and optimizes collection routes. The system selects only bins that are likely to need service and generates an efficient route for the truck.

## Key Features

- ML-based bin fill-level prediction
- Realistic synthetic smart-bin dataset
- Collection threshold control
- Greedy nearest-neighbor route optimization
- 2-opt route improvement
- Interactive Plotly map dashboard with a clean OpenStreetMap-derived basemap
- City manager recommendation
- Critical-bin district alert
- AI + route optimization decision pipeline
- Conservative, balanced, and aggressive scenario comparison
- Downloadable selected-bin and route-order CSVs
- Savings dashboard for distance, time, fuel, CO₂, and cost
- Model feature importance chart
- Clear deployment path for smart cities

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- plotly
- joblib

## AI/ML Methodology

EcoRoute AI uses a regression model to predict each bin's fill percentage. The target is `target_fill_pct`, and the model learns from operational features such as previous fill level, hours since collection, district, waste type, nearby activity score, weather, capacity, and day of week.

The ML pipeline uses:

- `RandomForestRegressor`
- `ColumnTransformer`
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features
- Passthrough numeric features

Metrics tracked in `models/metrics.json`:

- MAE
- RMSE
- R²

## Route Optimization Methodology

EcoRoute AI compares a naive fixed route against a demand-based optimized route:

- Haversine distance estimates geographic distance between bins.
- Fixed baseline route visits all bins sorted by `bin_id` ascending.
- Nearest-neighbor creates an initial route for selected bins.
- 2-opt improves the selected-bin route with `max_iterations=100`.
- Savings compare the optimized route against the fixed all-bin baseline.

## Dataset

Real municipal smart-bin sensor data is often not public, so the prototype uses a realistic synthetic dataset based on practical waste collection factors. The system is designed so real IoT sensor data can replace the synthetic dataset without changing the pipeline.

The default demo city is Astana, Kazakhstan, centered at:

- Latitude: `51.1694`
- Longitude: `71.4491`

Generated files:

- `data/training_data.csv`
- `data/bins.csv`
- `data/sample_predictions.csv`

Default demo scale:

- `180` current bins represent one dense dispatch zone for a single truck shift.
- `4,500` historical-style training observations keep the model stable while preserving fast local startup.
- City-wide deployments with hundreds or thousands of live bins should be split into district or fleet routes instead of one unreadable mega-route.

## How to Run

```bash
python -m venv venv
```

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app automatically generates data and trains the model if required.

## Project Structure

```text
ecoroute-ai/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── bins.csv
│   ├── training_data.csv
│   └── sample_predictions.csv
├── models/
│   ├── fill_level_model.pkl
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── routing.py
│   ├── savings.py
│   └── map_utils.py
├── assets/
│   └── screenshots/
└── presentation/
    └── slide_outline.md
```

Note: `*.pkl` files are ignored by Git, but the app works from a clean clone because it automatically trains `models/fill_level_model.pkl` if missing.

## Demo Flow

1. Open the Streamlit app.
2. View predicted bin fill levels.
3. Adjust collection threshold.
4. See selected and skipped bins.
5. View optimized route on the map.
6. Compare fixed vs optimized route.
7. Compare conservative, balanced, and aggressive threshold scenarios.
8. Download selected bins or the exact truck route order as CSV.
9. View estimated distance, time, fuel, CO₂, and cost savings.

## Results

Latest training run:

- Model: `RandomForestRegressor`
- Train rows: `3600`
- Test rows: `900`
- MAE: `4.534`
- RMSE: `5.78`
- R²: `0.929`

At the default `75%` threshold, the generated demo selected `29` of `180` bins for collection and produced an optimized selected-bin route of `38.387 km`.

Default scenario comparison:

- Conservative `85%`: `12` bins selected, `283.8 km` saved
- Balanced `75%`: `29` bins selected, `266.2 km` saved
- Aggressive `65%`: `52` bins selected, `246.6 km` saved

## Future Improvements

- Real IoT smart-bin sensor integration
- Live traffic-aware routing
- Multi-truck route optimization
- Dynamic collection scheduling
- Municipal waste management API integration
- Citizen reporting app
- Historical analytics dashboard for city managers
