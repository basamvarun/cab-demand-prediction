# 🚖 Cab Demand Prediction using LightGBM & Multi-Output Regressor

This project predicts cab demand across multiple regions of a city using machine learning. It integrates real-time weather data and location-based features from mapping APIs to enhance accuracy. The model uses **LightGBM** wrapped in a **Multi-Output Regressor** to handle simultaneous prediction for multiple zones.

---

## 📌 Objectives

- Predict the number of cab bookings for multiple zones in a city
- Leverage external features such as weather and events to improve model performance
- Build a scalable, modular machine learning pipeline

---

## 📊 Key Features

- LightGBM-based multi-target regression
- Real-time weather data integration via **OpenWeatherMap API**
- Map and location enrichment via **OpenStreetMap** or **Mapbox API**
- Exploratory data analysis, training, and evaluation notebooks
- Visualization of predicted vs actual cab demand

---

## 🛠️ Tech Stack

- **Language**: Python
- **ML Models**: LightGBM, MultiOutputRegressor (`sklearn`)
- **APIs**:
  - Weather: OpenWeatherMap API
  - Map/Event Data: OpenStreetMap / Mapbox / HERE Maps (optional)
- **Libraries**:
  - `pandas`, `numpy` – Data processing
  - `lightgbm`, `scikit-learn` – ML modeling
  - `requests`, `dotenv` – API integration
  - `matplotlib`, `seaborn` – Visualization

---

