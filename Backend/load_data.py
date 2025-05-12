# --- load_data.py (Revised with more Debugging and Simpler Fill) ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import holidays
import os
import joblib
import traceback

# import pyarrow # Make sure pyarrow is installed if reading parquet

# --- Configuration ---
# !! ADJUST THESE PATHS !!
# Assuming the script is run from the 'Code' directory
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Gets directory of the script (Backend)
CODE_DIR = os.path.dirname(BASE_DIR)  # Gets the parent 'Code' directory
DATA_DIR = os.path.join(CODE_DIR, "Dataset")  # Path to Dataset relative to Code dir
ZONE_LOOKUP_PATH = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
HISTORICAL_WEATHER_PATH = os.path.join(
    CODE_DIR, "nyc_weather_2024_hourly_meteostat.csv"
)  # Assume weather CSV is in Code dir
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data_ml/")  # Subdir within Backend
MODEL_DIR = os.path.join(BASE_DIR, "models_ml/")  # Subdir within Backend


WEATHER_COLS_TO_USE = [
    "temperature",
    "precipitation",
    "snow_depth",
    "humidity",
    "wind_speed",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Data Loading and Processing Script ---")
print(f"Using Base Directory: {BASE_DIR}")
print(f"Using Data Directory: {DATA_DIR}")
print(f"Outputting Processed Data to: {OUTPUT_DIR}")
print(f"Outputting Model Assets to: {MODEL_DIR}")

# --- Load Taxi Zone Lookup ---
print("\nLoading Taxi Zone Lookup CSV...")
try:
    zone_lookup_df = pd.read_csv(ZONE_LOOKUP_PATH)
    # Use raw string r"(\d+)" for regex
    zone_lookup_df["LocationID"] = (
        zone_lookup_df["LocationID"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    zone_lookup_df = zone_lookup_df[["LocationID", "Borough", "Zone"]].drop_duplicates(
        subset=["LocationID"]
    )
    VALID_ZONE_IDS = sorted(list(zone_lookup_df["LocationID"].unique()))
    NUM_ZONES = len(VALID_ZONE_IDS)
    if NUM_ZONES == 0:
        print("ERROR: No valid zones loaded...")
        exit()
    print(f"Loaded {NUM_ZONES} unique zones.")
except FileNotFoundError:
    print(f"ERROR: Zone lookup file not found at {ZONE_LOOKUP_PATH}.")
    exit()
except Exception as e:
    print(f"Error loading zone lookup: {e}")
    traceback.print_exc()
    exit()

# --- Load and Process Taxi Trip Data (2024) ---
print("\nLoading and Processing Taxi Trip Data (2024)...")
all_monthly_data = []
parquet_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".parquet") and "fhvhv" in f and "2024" in f
]
if not parquet_files:
    print(f"ERROR: No 2024 Parquet files found in {DATA_DIR}...")
    exit()
print(f"Found {len(parquet_files)} parquet files. Loading...")
parquet_cols_to_read = ["pickup_datetime", "PULocationID"]
for i, file_path in enumerate(parquet_files):
    print(f"Processing file: {os.path.basename(file_path)}...")
    try:
        df_month = pd.read_parquet(file_path, columns=parquet_cols_to_read)
        df_month.dropna(subset=parquet_cols_to_read, inplace=True)
        df_month["PULocationID"] = pd.to_numeric(
            df_month["PULocationID"], errors="coerce"
        )
        df_month.dropna(subset=["PULocationID"], inplace=True)
        df_month["PULocationID"] = df_month["PULocationID"].astype(int)
        df_month = df_month[df_month["PULocationID"].isin(VALID_ZONE_IDS)]
        df_month["pickup_datetime"] = pd.to_datetime(df_month["pickup_datetime"])
        # Filter strictly by year - handle potential timezone issues if data spans year boundary precisely
        df_month = df_month[df_month["pickup_datetime"].dt.year == 2024]
        df_month.dropna(subset=["pickup_datetime"], inplace=True)
        if not df_month.empty:
            all_monthly_data.append(df_month[["pickup_datetime", "PULocationID"]])
    except Exception as e:
        print(f"  Warning: Error processing file {file_path}: {e}")
if not all_monthly_data:
    print("ERROR: No data loaded...")
    exit()
print("Concatenating monthly taxi data...")
df_trips = pd.concat(all_monthly_data, ignore_index=True)
del all_monthly_data
print(f"Total valid trips: {len(df_trips)}")
print("Aggregating demand...")
df_trips["pickup_hour"] = df_trips["pickup_datetime"].dt.floor("h")
df_demand = (
    df_trips.groupby(["pickup_hour", "PULocationID"]).size().reset_index(name="demand")
)
del df_trips
print("Pivoting demand data...")
df_pivot = df_demand.pivot_table(
    index="pickup_hour", columns="PULocationID", values="demand", fill_value=0
)

# --- Timezone Handling and Indexing ---
print("\nApplying Timezone (Converting to UTC) and Creating Full 2024 Index...")
try:
    if df_pivot.index.tz is None:
        print("Taxi index naive, localizing directly to UTC...")
        df_pivot.index = df_pivot.index.tz_localize(
            "UTC", ambiguous="raise", nonexistent="raise"
        )  # Be strict
    elif df_pivot.index.tz != "UTC":
        print(f"Converting taxi index ({df_pivot.index.tz}) to UTC...")
        df_pivot.index = df_pivot.index.tz_convert("UTC")
    else:
        print("Taxi index already UTC.")
except Exception as tz_err:
    print(f"ERROR: TZ handling failed for taxi data: {tz_err}.")
    traceback.print_exc()
    exit()
start_2024_utc = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
end_2024_utc = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
all_hours_utc = pd.date_range(
    start=start_2024_utc, end=end_2024_utc, freq="h"
)  # Use 'h'
df_pivot = df_pivot.reindex(all_hours_utc, fill_value=0)
df_pivot = df_pivot.reindex(columns=VALID_ZONE_IDS, fill_value=0)
df_pivot.fillna(0, inplace=True)
df_pivot = df_pivot.astype(int)
print(f"Pivoted demand DataFrame shape (UTC index): {df_pivot.shape}")


# --- Generate Time Features ---
print("\nGenerating time-based features (from UTC index)...")
df_features = pd.DataFrame(index=df_pivot.index)  # Use the demand index
df_features["hour"] = df_features.index.hour
df_features["dayofweek"] = df_features.index.dayofweek
df_features["dayofmonth"] = df_features.index.day
df_features["month"] = df_features.index.month
df_features["quarter"] = df_features.index.quarter
df_features["dayofyear"] = df_features.index.dayofyear
df_features["weekofyear"] = df_features.index.isocalendar().week.astype(int)
df_features["is_weekend"] = df_features["dayofweek"].isin([5, 6]).astype(int)
hist_years = df_features.index.year.unique()
us_holidays = holidays.US(years=hist_years)
df_features["is_holiday"] = df_features.index.date
df_features["is_holiday"] = (
    df_features["is_holiday"].map(lambda dt: 1 if dt in us_holidays else 0).astype(int)
)
seconds_in_day = 24 * 60 * 60
seconds_in_year = 365.2425 * seconds_in_day  # Leap year approx
timestamps_sec = df_features.index.astype(np.int64) // 10**9
df_features["hour_sin"] = np.sin(timestamps_sec * (2 * np.pi / seconds_in_day))
df_features["hour_cos"] = np.cos(timestamps_sec * (2 * np.pi / seconds_in_day))
df_features["month_sin"] = np.sin(timestamps_sec * (2 * np.pi / seconds_in_year))
df_features["month_cos"] = np.cos(timestamps_sec * (2 * np.pi / seconds_in_year))
print(f"Time features generated. Shape: {df_features.shape}")


# --- Load and Merge Historical Weather Data ---
print("\nLoading historical weather data (2024)...")
try:
    df_weather_hist = pd.read_csv(
        HISTORICAL_WEATHER_PATH, index_col=0, parse_dates=True
    )

    # Convert weather index to UTC
    if df_weather_hist.index.tz is None:
        print("Weather index naive, localizing to UTC...")
        df_weather_hist.index = df_weather_hist.index.tz_localize(
            "UTC", ambiguous="raise", nonexistent="raise"
        )
    elif df_weather_hist.index.tz != "UTC":
        print(f"Converting weather index ({df_weather_hist.index.tz}) to UTC...")
        df_weather_hist.index = df_weather_hist.index.tz_convert("UTC")
    else:
        print("Weather index is already UTC.")

    # Check for missing weather columns
    missing_weather_cols = [
        col for col in WEATHER_COLS_TO_USE if col not in df_weather_hist.columns
    ]
    if missing_weather_cols:
        print(f"ERROR: Weather columns missing: {missing_weather_cols}")
        exit()
    df_weather_hist = df_weather_hist[WEATHER_COLS_TO_USE]
    print(f"Selected weather columns: {list(df_weather_hist.columns)}")

    # --- Specific NaN Handling for Weather DF ---
    if "snow_depth" in df_weather_hist.columns:
        print("Filling NaNs in 'snow_depth' with 0...")
        df_weather_hist["snow_depth"].fillna(0, inplace=True)
    print("Interpolating remaining NaNs in weather data...")
    df_weather_hist = df_weather_hist.interpolate(method="time")
    print("Forward/Backward filling any remaining start/end NaNs...")
    df_weather_hist = df_weather_hist.ffill().bfill()
    # ** DEBUG: Check for NaNs in weather BEFORE merge **
    weather_nans = df_weather_hist.isnull().sum()
    print("NaN counts in weather data BEFORE merge:\n", weather_nans[weather_nans > 0])
    if df_weather_hist.isnull().values.any():
        print("ERROR: NaNs still in weather DF before merge!")
        exit()

    # ** DEBUG: Check index alignment BEFORE merge **
    print(f"\n--- Index Debug ---")
    print(
        f"Time Features Index - Start: {df_features.index.min()}, End: {df_features.index.max()}, Length: {len(df_features.index)}, Freq: {df_features.index.freq}"
    )
    print(
        f"Weather Hist Index  - Start: {df_weather_hist.index.min()}, End: {df_weather_hist.index.max()}, Length: {len(df_weather_hist.index)}, Freq: {df_weather_hist.index.freq}"
    )
    index_equal_check = df_features.index.equals(df_weather_hist.index)
    print(f"Index alignment check before concat: {index_equal_check}")
    if not index_equal_check:
        print(
            "Warning: Indices are not perfectly equal before concat. Reindexing weather to match features index."
        )
        # Ensure weather index covers the full range needed by features index
        df_weather_hist = df_weather_hist.reindex(df_features.index)
        print("Re-running NaN handling on weather data after reindex...")
        if "snow_depth" in df_weather_hist.columns:
            df_weather_hist["snow_depth"].fillna(0, inplace=True)
        df_weather_hist = df_weather_hist.interpolate(method="time").ffill().bfill()
        weather_nans_after_reindex = df_weather_hist.isnull().sum()
        print(
            "NaN counts in weather data AFTER reindex+fill:\n",
            weather_nans_after_reindex[weather_nans_after_reindex > 0],
        )
        if df_weather_hist.isnull().values.any():
            print("ERROR: NaNs in weather DF after reindex & fill!")
            exit()
        print(f"Weather index length after reindex: {len(df_weather_hist.index)}")
    print(f"--- End Index Debug ---\n")

    # --- Combine time features and weather features ---
    print("Merging time and weather features...")
    # Use join instead of concat for potentially better index handling
    # Ensure index names are different or None to avoid issues if merging on index
    df_features.index.name = "timestamp_feat"
    df_weather_hist.index.name = "timestamp_weather"
    df_features_combined = df_features.join(
        df_weather_hist, how="left"
    )  # Left join ensures all time features rows are kept
    df_features_combined.index.name = "timestamp"  # Reset index name

    # ** DEBUG: Check NaNs AFTER join **
    join_nans = df_features_combined.isnull().sum()
    print("NaN counts AFTER join:\n", join_nans[join_nans > 0])

    # Reindex based on the primary data index (df_pivot) for final alignment (safety step after join)
    # This shouldn't be necessary if the join was on the correct index (df_features.index == df_pivot.index)
    # df_features_combined = df_features_combined.reindex(df_pivot.index)
    # reindex_nans = df_features_combined.isnull().sum()
    # print("NaN counts AFTER reindex (post-join):\n", reindex_nans[reindex_nans > 0])

    # Handle potential NaNs from the join/reindex
    if df_features_combined.isnull().values.any():
        print("Warning: NaNs found after join. Filling ALL remaining NaNs with 0...")
        nan_cols_final = df_features_combined.columns[
            df_features_combined.isnull().any()
        ].tolist()
        print(f"Columns with NaNs to fill: {nan_cols_final}")
        # ** SIMPLIFIED FILL: Fill ALL NaNs with 0 **
        df_features_combined.fillna(0, inplace=True)

    # Final check for NaNs before scaling
    if df_features_combined.isnull().values.any():
        print("ERROR: NaNs remain in combined features after final fill(0).")
        exit()
    else:
        print("No NaNs found in combined features after final fill(0).")

    print(f"Combined features shape before scaling: {df_features_combined.shape}")
    FEATURE_NAMES = df_features_combined.columns.tolist()

except FileNotFoundError:
    print(f"ERROR: Historical weather file not found at {HISTORICAL_WEATHER_PATH}")
    exit()
except KeyError as e:
    print(
        f"ERROR: Column missing during processing: {e}. Check WEATHER_COLS_TO_USE or CSV content."
    )
    traceback.print_exc()
    exit()
except Exception as e:
    print(f"Error loading/merging historical weather data: {e}")
    traceback.print_exc()
    exit()


# --- Scaling ---
print("\nScaling combined features...")
feature_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(df_features_combined)
# Preserve index and columns during scaling
df_scaled_features = pd.DataFrame(
    scaled_features, index=df_features_combined.index, columns=FEATURE_NAMES
)


# --- Saving ---
print("\nSaving processed data and assets...")
df_target = df_pivot  # Already aligned to the correct UTC index
target_file = os.path.join(OUTPUT_DIR, "target_demand_ml.parquet")
df_target.to_parquet(target_file)
print(f"Target saved (Shape: {df_target.shape})")
features_file = os.path.join(OUTPUT_DIR, "scaled_features_ml.parquet")
df_scaled_features.to_parquet(features_file)
print(f"Features saved (Shape: {df_scaled_features.shape})")
scaler_filename_features = os.path.join(MODEL_DIR, "feature_scaler_ml.joblib")
joblib.dump(feature_scaler, scaler_filename_features)
print(f"Scaler saved.")
zone_order_file = os.path.join(MODEL_DIR, "zone_order_ml.joblib")
joblib.dump(VALID_ZONE_IDS, zone_order_file)
print(f"Zone order saved.")
feature_names_file = os.path.join(MODEL_DIR, "feature_names_ml.joblib")
joblib.dump(FEATURE_NAMES, feature_names_file)
print(f"Feature names saved.")

print("\n--- Data processing (with weather) complete. ---")
