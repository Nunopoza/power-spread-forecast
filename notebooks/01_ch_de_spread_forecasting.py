"""
Swiss Power Spread Forecast (2025 Data)
---------------------------------------
Pipeline completo para limpiar, unir y modelar datos de precios, demanda y generación solar/eólica.
Adaptado a los CSVs de ENTSO-E descargados en 2025 (nueva plataforma).

Autor: Nuno Poza
Repositorio: github.com/Nunopoza/power-spread-forecast
"""

# ====================================================
# 1. Imports
# ====================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os


# ====================================================
# 2. Helper functions
# ====================================================

def clean_price_csv(path, value_col_name):
    """Limpia CSVs de precios ENTSO-E (nueva versión 2025)."""
    df = pd.read_csv(path, sep=",", decimal=".", usecols=["MTU (UTC)", "Day-ahead Price (EUR/MWh)"])
    df.columns = ["MTU", value_col_name]
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")
    df["datetime"] = df["MTU"].astype(str).str.split(" - ").str[0]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.set_index("datetime").drop(columns=["MTU"]).dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def clean_generation_csv(path, value_col_name):
    """Limpia CSVs de generación solar/eólica ENTSO-E (2025)."""
    df = pd.read_csv(path, sep=",", decimal=".", usecols=["MTU (UTC)", "Day-ahead (MW)"])
    df.columns = ["MTU", value_col_name]
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")
    df["datetime"] = df["MTU"].astype(str).str.split(" - ").str[0]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.set_index("datetime").drop(columns=["MTU"]).dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# ====================================================
# 3. Load data
# ====================================================

print("Loading CSV files from data/raw/...")

# --- Price (CH) ---
price_ch = clean_price_csv("data/raw/GUI_ENERGY_PRICES_202501010000-202601010000.csv", "price_ch")

# --- Load (CH) ---
print("Loading load_ch ...")
load_ch = pd.read_csv(
    "data/raw/GUI_TOTAL_LOAD_DAYAHEAD_202501010000-202601010000.csv",
    sep=",",
    decimal="."
)

expected_cols = ["MTU (UTC)", "Day-ahead Total Load Forecast (MW)"]
missing = [c for c in expected_cols if c not in load_ch.columns]
if missing:
    raise ValueError(f"Missing expected columns in load_ch CSV: {missing}. Found: {list(load_ch.columns)}")

load_ch = load_ch[["MTU (UTC)", "Day-ahead Total Load Forecast (MW)"]]
load_ch.columns = ["MTU", "load_ch"]
load_ch["datetime"] = load_ch["MTU"].astype(str).str.split(" - ").str[0]
load_ch["datetime"] = pd.to_datetime(load_ch["datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
load_ch = load_ch.set_index("datetime").drop(columns=["MTU"]).dropna()
load_ch = load_ch[~load_ch.index.duplicated(keep="first")].sort_index()
print(f"load_ch loaded: {len(load_ch)} rows, {load_ch.index.min()} → {load_ch.index.max()}")

# --- Solar (DE) ---
solar_de = clean_generation_csv("data/raw/GUI_WIND_SOLAR_GENERATION_FORECAST_SOLAR_202501010000-202601010000.csv", "solar_de")

# --- Wind (DE) ---
wind_on = clean_generation_csv("data/raw/GUI_WIND_SOLAR_GENERATION_FORECAST_ONSHORE_202501010000-202601010000.csv", "wind_on")
wind_off = clean_generation_csv("data/raw/GUI_WIND_SOLAR_GENERATION_FORECAST_OFFSHORE_202501010000-202601010000.csv", "wind_off")
wind_de = pd.concat([wind_on, wind_off], axis=1).sum(axis=1).to_frame("wind_de")


# ====================================================
# 4. Clean & align indices
# ====================================================

for name, df in [("price_ch", price_ch), ("load_ch", load_ch), ("solar_de", solar_de), ("wind_de", wind_de)]:
    print(f"{name}: {df.index.min()} → {df.index.max()} ({len(df)} rows)")

start = max(df.index.min() for df in [price_ch, load_ch, solar_de, wind_de])
end = min(df.index.max() for df in [price_ch, load_ch, solar_de, wind_de])
print(f"Common range: {start} → {end}")

price_ch = price_ch.loc[start:end]
load_ch = load_ch.loc[start:end]
solar_de = solar_de.loc[start:end]
wind_de = wind_de.loc[start:end]

data = pd.concat([price_ch, load_ch, solar_de, wind_de], axis=1, join="inner").dropna()
if data.empty:
    raise ValueError(" Dataset is empty after merging. Check date overlap between CSVs.")
print(" Data merged successfully:", data.shape)

# ====================================================
# 5. Feature engineering
# ====================================================

data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["month"] = data.index.month
data["spread"] = data["price_ch"] - data["price_ch"].rolling(24, min_periods=1).mean()

features = ["hour", "dayofweek", "month", "load_ch", "solar_de", "wind_de"]
target = "spread"

X = data[features]
y = data[target]

# ====================================================
# 6. Model training
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:")
print(f"MAE: {mae:.2f} €/MWh")
print(f"R²: {r2:.3f}")

# ====================================================
# 7. Visualization
# ====================================================

# ====================================================
# 7. Visualization (Enhanced)
# ====================================================

import matplotlib.dates as mdates

plt.style.use("seaborn-v0_8-darkgrid")
plt.figure(figsize=(14, 6))

# Plot
plt.plot(
    y_test.index, y_test,
    label="Actual Spread", color="#1f77b4", linewidth=1.5, alpha=0.9
)
plt.plot(
    y_test.index, y_pred,
    label="Predicted Spread", color="#ff7f0e", linewidth=1.5, alpha=0.8
)

# Titles and labels
plt.title("Swiss Power Day-Ahead Spread Forecast", fontsize=16, weight="bold", pad=20)
plt.suptitle("Model: Random Forest | Period: Jan–Dec 2025", fontsize=10, color="gray")

plt.xlabel("Date", fontsize=12)
plt.ylabel("Spread (€/MWh)", fontsize=12)

# Date formatting
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)

# Grid and legend
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True, loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("figures/spread_forecast.png", dpi=200)
plt.savefig("figures/feature_importance.png", dpi=200)
plt.show()

# Feature importance plot
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 4))
importances.plot(
    kind="barh",
    color="#2ca02c",
    edgecolor="black"
)
plt.title("Feature Importance – Random Forest", fontsize=14, weight="bold")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("figures/spread_forecast.png", dpi=200)
plt.savefig("figures/feature_importance.png", dpi=200)
plt.show()


# ====================================================
# 8. Save results
# ====================================================

os.makedirs("data/processed", exist_ok=True)
results = pd.DataFrame({"actual": y_test.values, "predicted": y_pred}, index=y_test.index)
results.to_csv("data/processed/spread_predictions.csv")
print(" Results saved to data/processed/spread_predictions.csv")
