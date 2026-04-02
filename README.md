NYC Cab Demand Prediction
This project aims to predict the cab demand in different zones of New York City using historical trip data, weather information, and various machine learning models. It helps to optimize cab availability and improve service efficiency.

Project Structure
Cab-Demand-Prediction/
├── Backend/
│   ├── API.py
│   ├── train_ml.py
│   ├── load_data.py
│   ├── accuracy.py
│   └── fetch_historical_weather.py
├── Frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── Dataset/
│   ├── taxi_zone_lookup.csv
│   └── zones.geojson
├── .gitignore
└── README.md

💡 Features
🔍 Predicts cab demand per zone using historical FHV data
🌦 Integrates weather data to improve prediction accuracy
📊 Evaluates performance with MAE, RMSE, and zone-wise metrics
🖥️ Simple frontend visualization with 3D Map using CesiumJS
🚀 REST API for prediction (Flask-based backend)

🔗 Dataset Source
We use NYC’s open public dataset for cab rides:

👉 TLC Trip Record Data

The dataset includes detailed trip-level records like pickup time, location, and more.

🛠️ Tech Stack
Python – Data preprocessing & ML model training
LightGBM – Machine learning model for demand prediction
Pandas / NumPy – Data analysis
Parquet – Efficient data storage
Flask – Backend API
HTML / CSS / JS / CesiumJS – Frontend interface

🧹 Excluded from GitHub
To keep the repository lightweight and within GitHub’s limits:

📁 Dataset/ – Raw trip data files
📁 Backend/models_ml/ – Trained ML models
📁 Backend/processed_data_ml/ – Processed intermediate data
📁 Documentation/ – PDF/DOCX reports and presentations
📄 Large files over 100 MB (handled with .gitignore)
