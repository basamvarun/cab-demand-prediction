# --- train_ml.py ---
import pandas as pd
import numpy as np
import lightgbm as lgb

# from sklearn.model_selection import TimeSeriesSplit # Not used if CV skipped
# from sklearn.model_selection import cross_val_score # Not used if CV skipped
from sklearn.multioutput import MultiOutputRegressor
import os
import joblib
import warnings
import traceback

# Suppress specific LightGBM warnings (optional)
warnings.filterwarnings("ignore", message="Found `n_estimators` in params")
warnings.filterwarnings(
    "ignore", message="[LightGBM] [Warning] feature_fraction is set"
)
warnings.filterwarnings(
    "ignore", message="[LightGBM] [Warning] bagging_fraction is set"
)
warnings.filterwarnings("ignore", message="[LightGBM] [Warning] bagging_freq is set")

# --- Configuration ---
# Assuming the script is run from the 'Backend' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data_ml/")
MODEL_DIR = os.path.join(BASE_DIR, "models_ml/")

PROCESSED_TARGET_FILE = os.path.join(OUTPUT_DIR, "target_demand_ml.parquet")
PROCESSED_FEATURES_FILE = os.path.join(
    OUTPUT_DIR, "scaled_features_ml.parquet"
)  # Contains combined scaled features
ZONE_ORDER_FILE = os.path.join(MODEL_DIR, "zone_order_ml.joblib")
MODEL_SAVE_PATH = os.path.join(
    MODEL_DIR, "lgbm_demand_model.joblib"
)  # Saving the wrapped model

# --- Parameters ---
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_leaves": 31,
    "max_depth": -1,
    "seed": 42,
    "n_jobs": -1,
    "verbose": -1,
    "boosting_type": "gbdt",
}
WRAPPER_N_JOBS = (
    -1
)  # Use all available cores for training models per target in parallel for final fit

print("--- Model Training Script ---")
print(f"Loading data from: {os.path.abspath(OUTPUT_DIR)}")
print(f"Saving model components to: {os.path.abspath(MODEL_DIR)}")

# --- Load Processed Data ---
print("\nLoading processed data (features and target)...")
try:
    df_target = pd.read_parquet(PROCESSED_TARGET_FILE)
    df_scaled_features = pd.read_parquet(
        PROCESSED_FEATURES_FILE
    )  # Now includes weather
    zone_order = joblib.load(ZONE_ORDER_FILE)
    NUM_ZONES = len(zone_order)

    # --- Data Validation and Alignment ---
    # Ensure indices match (should be UTC from load_data)
    print(
        f"Aligning target index ({len(df_target)}) to features index ({len(df_scaled_features)})..."
    )
    df_target = df_target.reindex(df_scaled_features.index)
    print(f"Re-ordering target columns to match zone_order ({len(zone_order)})...")
    df_target = df_target[zone_order]  # Ensure target columns are in the expected order

    print("Data loaded successfully.")
    print(f"Features shape: {df_scaled_features.shape}")
    print(f"Target shape: {df_target.shape}")

except FileNotFoundError as fnf:
    print(f"ERROR: Required file not found: {fnf}. Run load_data.py first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()
    exit()

# Further data validation
if df_scaled_features.empty or df_target.empty:
    print("ERROR: Loaded features or target data is empty.")
    exit()
if len(df_scaled_features) != len(df_target):
    print("ERROR: Mismatch between number of feature rows and target rows.")
    exit()
if df_scaled_features.isnull().values.any():
    print(
        "ERROR: NaNs found in scaled features data after loading. Check load_data.py."
    )
    exit()
if df_target.isnull().values.any():
    print("ERROR: NaNs found in target data after loading/reindex. Check load_data.py.")
    exit()


# --- Prepare Data for Model ---
X = df_scaled_features  # Features (scaled, includes time + weather)
y = df_target  # Target (original demand counts, multi-column)


# --- Define the Base Estimator ---
print("\nDefining Base Estimator (LightGBM)...")
base_lgbm_estimator = lgb.LGBMRegressor(**LGBM_PARAMS)
print(f"Base Estimator Type: {type(base_lgbm_estimator)}")


# --- CROSS-VALIDATION BLOCK - SKIPPED ---
print(f"\nSkipping Time Series Cross-Validation step.")


# --- Train Final Model on ALL 2024 Data ---
print(
    "\nAttempting to train final Multi-Output LightGBM model (with weather features)..."
)

# Create the final wrapped model instance using the base estimator blueprint
final_model_wrapped = MultiOutputRegressor(base_lgbm_estimator, n_jobs=WRAPPER_N_JOBS)
print(f"Final Model Type: {type(final_model_wrapped)}")
print(f"Using n_jobs={WRAPPER_N_JOBS} for parallel training of zone models.")

try:
    print(f"Fitting final model with X shape {X.shape} and y shape {y.shape}...")
    start_time = pd.Timestamp.now()
    final_model_wrapped.fit(X, y)
    end_time = pd.Timestamp.now()
    print(f"Final model training complete. Duration: {end_time - start_time}")

    # --- Save the Trained WRAPPED Model ---
    print(f"\nSaving final model to {MODEL_SAVE_PATH}...")
    joblib.dump(final_model_wrapped, MODEL_SAVE_PATH)
    print(f"Final wrapped LightGBM model saved successfully.")

except Exception as fit_err:
    # Catch errors during the final model fitting
    print(f"---!!! FINAL FITTING FAILED !!!---")
    print(f"Error during final fit: {fit_err}")
    traceback.print_exc()  # Print detailed traceback
    print("------------------------------------")
    print("Model training failed. Model file was NOT saved.")
    exit()  # Exit script if final training fails

print("\n--- Model Training Script End ---")
print("Model training completed successfully.")
