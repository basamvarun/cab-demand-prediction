# Geospatial Demand Forecasting Engine for Urban Mobility (NYC)

A high-performance, full-stack machine learning system designed to predict taxi demand across New York City’s 260+ taxi zones using historical trip data and real-time weather telemetry.

---

## 🚀 Overview
This project addresses the "Cold Start" and "Relocation" problems for ride-hailing services by providing drivers and fleet managers with real-time, data-driven insights into high-demand urban clusters. Using a Gradient Boosted Decision Tree (GBDT) architecture, the system forecasts demand with high temporal and spatial precision.

### 🧠 Core Features
- **Predictive Intelligence**: Uses **LightGBM** to forecast cab demand with seasonal and environmental awareness.
- **Geospatial Dashboard**: Interactive 2D/3D visualization built with **CesiumJS** for zone-wise demand heatmaps.
- **Real-time API**: High-concurrency **Flask REST API** providing sub-100ms inference.
- **Weather Integration**: Dynamic feature engineering incorporating real-time weather data (temperature, precipitation, etc.) to improve accuracy by [X]%.

---

## 🛠️ Tech Stack
| Layer | Technologies |
| :--- | :--- |
| **ML & Data** | Python, LightGBM, Pandas, NumPy, Scikit-Learn, Parquet |
| **Backend** | Flask, Flask-CORS, Joblib, Holidays API |
| **Frontend** | JavaScript (ES6+), CesiumJS, HTML5, CSS3 (Vanilla) |
| **Data Source** | NYC TLC Open Dataset, Meteostat (Historical Weather) |

---


## 🏗️ Project Structure
```text
├── Backend/
│   ├── API.py              # Flask REST Server
│   ├── train_ml.py         # Model training & Feature selection
│   ├── accuracy.py         # Metric evaluation (MAE, RMSE)
│   └── models_ml/          # Trained weights (excluded from Git due to size)
├── Frontend/
│   ├── index.html          # Core dashboard
│   ├── script.js           # Cesium map logic & API interaction
│   └── style.css           # Premium UI/UX design
├── Dataset/                # Geospatial boundaries (GeoJSON)
└── nyc_weather_2024.csv    # Temporal environmental data
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+
- A valid [Cesium Access Token](https://ion.cesium.com/) (Required for map rendering)

### 2. Backend Setup
```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the inference server
./start_backend.sh
```

### 3. Frontend Setup
Run a local web server from the project root:
```bash
python3 -m http.server 8082
```
Navigate to: `http://localhost:8082/Frontend/index.html`

---

## 📦 Model Weights Notice
The trained **LGBM model file (353MB)** is currently excluded from Git to comply with standard repository limits (100MB). To push weights larger than 100MB, please use **Git LFS (Large File Storage)** or contact the architect for a secure download link.

---
*Developed for excellence in urban mobility analytics.*
