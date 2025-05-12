# --- fetch_historical_weather.py (Using meteostat) ---
import pandas as pd
from datetime import datetime
from meteostat import Point, Hourly  # Import Point and Hourly
import os

# Define the location (New York City)
# You can find station IDs near NYC using meteostat's Stations().nearby(LAT, LON).fetch()
# Example: Central Park is often '72503' but check availability. LGA might be '72503'. JFK '74486'.
# Using coordinates is often easier to start.
LAT = 40.7128
LON = -74.0060
ALTITUDE = 10  # Approximate altitude in meters for NYC

location = Point(LAT, LON, ALTITUDE)

OUTPUT_FILE = "nyc_weather_2024_hourly_meteostat.csv"

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31, 23, 59, 59)  # End of Dec 31st

print(f"Fetching historical weather using meteostat from {start_date} to {end_date}...")

try:
    # Fetch hourly data for the period
    # This single call gets the whole period
    data = Hourly(location, start_date, end_date)
    df_weather = data.fetch()

    if df_weather.empty:
        print("ERROR: Meteostat fetch returned no data. Check location or date range.")
        exit()

    print(f"Fetched {len(df_weather)} hourly records.")

    # --- Select and Rename Columns ---
    # Meteostat column names: temp, dwpt, rhum, prcp, snow,wdir, wspd, pres, tsun, coco
    # Rename to match what your load_data expects (or adjust load_data)
    df_weather = df_weather.rename(
        columns={
            "temp": "temperature",
            "prcp": "precipitation",  # Precipitation amount in mm
            "snow": "snow_depth",  # Snow depth on ground in mm (might need 'coco' for snowfall intensity)
            "rhum": "humidity",
            "wspd": "wind_speed",
            # Add/rename others as needed (e.g., 'coco' for weather condition code)
        }
    )

    # Select the columns you need
    cols_to_keep = [
        "temperature",
        "precipitation",
        "snow_depth",
        "humidity",
        "wind_speed",
    ]  # Adjust as needed
    df_weather = df_weather[cols_to_keep]

    # Meteostat might have missing hours, check and interpolate
    df_weather = df_weather.interpolate(method="time").ffill().bfill()

    df_weather.to_csv(OUTPUT_FILE)
    print(f"Historical weather data saved to {OUTPUT_FILE}")

except Exception as e:
    print(f"ERROR during meteostat fetch or processing: {e}")
    import traceback

    traceback.print_exc()
