import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.dates as mdates
import os

# ====================================================
# Helper functions
# ====================================================

def clean_price_csv(path, value_col_name):
    """Clean ENTSO-E day-ahead price CSVs (CH or DE_LU)."""
    df = pd.read_csv(path, sep=",", decimal=".")
    df = df[["MTU (UTC)", "Day-ahead Price (EUR/MWh)"]]
    df["datetime"] = df["MTU (UTC)"].astype(str).str.split(" - ").str[0]

    for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]:
        df["datetime"] = pd.to_datetime(df["datetime"], format=fmt, errors="coerce")
        if df["datetime"].notna().sum() > 0:
            break

    df = df.groupby("datetime")["Day-ahead Price (EUR/MWh)"].mean().to_frame(value_col_name)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Convert to hourly if 15-min
    freq = pd.infer_freq(df.index[:10])
    if freq == "15T" or (df.index.to_series().diff().dt.total_seconds().median() == 900):
        df = df.resample("1h").mean()

    return df


def clean_load_csv(path, value_col_name):
    """Clean and resample ENTSO-E total load forecast data (CH / DE-LU)."""
    import re

    # Leer CSV (coma separador, decimales con punto)
    df = pd.read_csv(path, sep=",", decimal=".", quotechar='"', skip_blank_lines=True)

    # Buscar la columna de pronóstico
    col_candidates = [c for c in df.columns if "Day-ahead" in c and "Load" in c]
    if not col_candidates:
        raise ValueError(f"No Day-ahead load column found in {path}")
    col_name = col_candidates[0]

    # Parsear fecha de inicio del intervalo
    df["datetime"] = (
        df["MTU (UTC)"]
        .astype(str)
        .apply(lambda x: re.split(r"\s*-\s*", x)[0].strip())
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M", errors="coerce", dayfirst=True)

    # Convertir a numérico
    df[value_col_name] = pd.to_numeric(df[col_name], errors="coerce")

    # Filtrar filas válidas
    df = df.dropna(subset=["datetime", value_col_name]).set_index("datetime").sort_index()

    # Detectar si es cuarto de hora y reagrupar a hora
    median_step = df.index.to_series().diff().dt.total_seconds().median()
    if median_step and median_step < 3600:
        # Resamplear solo la columna numérica
        df = df[[value_col_name]].resample("1h").mean()

    # Eliminar duplicados y devolver solo la columna
    df = df[~df.index.duplicated(keep="first")]
    return df[[value_col_name]]



def clean_generation_csv(path, value_col_name):
    """Clean ENTSO-E wind/solar forecast CSVs (DE_LU)."""
    df = pd.read_csv(path, sep=",", decimal=".")
    df = df[["MTU (UTC)", "Day-ahead (MW)"]]
    df["datetime"] = df["MTU (UTC)"].astype(str).str.split(" - ").str[0]

    for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]:
        df["datetime"] = pd.to_datetime(df["datetime"], format=fmt, errors="coerce")
        if df["datetime"].notna().sum() > 0:
            break

    df = df.set_index("datetime")[["Day-ahead (MW)"]]
    df.columns = [value_col_name]

    # Clean bad values
    df[value_col_name] = (
        df[value_col_name]
        .astype(str)
        .str.replace("n/e", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")

    # Resample 15-min → 1h
    freq = pd.infer_freq(df.index[:10])
    if freq == "15T" or (df.index.to_series().diff().dt.total_seconds().median() == 900):
        df = df.resample("1h").mean()

    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna()
    return df


# ====================================================
# Load and prepare data
# ====================================================

print("Loading price data...")

price_ch = clean_price_csv("data/raw/GUI_ENERGY_PRICES_CH_202501010000-202601010000.csv", "price_ch")
price_de = clean_price_csv("data/raw/GUI_ENERGY_PRICES_DE_202501010000-202601010000.csv", "price_de")

spread = price_ch.join(price_de, how="inner")
spread["spread"] = spread["price_ch"] - spread["price_de"]
spread = spread.dropna()

print(f"price_ch: {price_ch.index.min()} → {price_ch.index.max()} ({len(price_ch)} rows)")
print(f"price_de: {price_de.index.min()} → {price_de.index.max()} ({len(price_de)} rows)")
print(f"Common range: {spread.index.min()} → {spread.index.max()} ({len(spread)} rows)")

print("\nLoading fundamental data...")

load_ch = clean_load_csv("data/raw/GUI_TOTAL_LOAD_DAYAHEAD_202501010000-202601010000.csv", "load_ch")
load_de = clean_load_csv("data/raw/GUI_TOTAL_LOAD_DAYAHEAD_202501010000-202601010000 (1).csv", "load_de")

solar_de = clean_generation_csv("data/raw/GUI_WIND_SOLAR_GENERATION_FORECAST_SOLAR_202501010000-202601010000 (1).csv", "solar_de")
wind_on = clean_generation_csv("data/raw/GUI_WIND_SOLAR_GENERATION_FORECAST_ONSHORE_202501010000-202601010000 (1).csv", "wind_on")
wind_off = clean_generation_csv("data/raw/GUI_WIND_SOLAR_GENERATION_FORECAST_OFFSHORE_202501010000-202601010000 (1).csv", "wind_off")

wind_de = pd.concat([wind_on, wind_off], axis=1).sum(axis=1).to_frame("wind_de")

# ====================================================
# Merge everything
# ====================================================

dfs = [spread, load_ch, load_de, solar_de, wind_de]
for i in range(len(dfs)):
    dfs[i].index = pd.to_datetime(dfs[i].index).round("1h")

common_index = dfs[0].index
for df in dfs[1:]:
    common_index = common_index.intersection(df.index)

data = pd.concat([df.loc[common_index] for df in dfs], axis=1).dropna()

print(f"\nData merged successfully: {data.shape[0]} rows")
print(f"Date range: {data.index.min()} → {data.index.max()}\n")

# ====================================================
# Feature engineering
# ====================================================

data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["month"] = data.index.month

features = ["hour", "dayofweek", "month", "load_ch", "load_de", "solar_de", "wind_de"]
target = "spread"

X = data[features]
y = data[target]

# === DEBUG: check dataset sizes ===
print("\nDataset overview before merge:\n")
for name, df in [
    ("price_ch", price_ch),
    ("price_de", price_de),
    ("load_ch", load_ch),
    ("load_de", load_de),
    ("solar_de", solar_de),
    ("wind_on", wind_on),
    ("wind_off", wind_off),
]:
    print(f"{name:10s} → {len(df):6d} rows | {df.index.min()} → {df.index.max()}")

# ====================================================
# Train/test split
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ====================================================
# Model training
# ====================================================

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ====================================================
# Evaluation
# ====================================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f} €/MWh")
print(f"R²: {r2:.3f}")

# ====================================================
# Visualization
# ====================================================

plt.style.use("seaborn-v0_8-darkgrid")
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label="Actual Spread (CH–DE)", color="#1f77b4", linewidth=1.5)
plt.plot(y_test.index, y_pred, label="Predicted Spread", color="#ff7f0e", linewidth=1.5)
plt.title("Swiss–German Day-Ahead Power Spread Forecast (2025)", fontsize=16, weight="bold")
plt.xlabel("Date")
plt.ylabel("Spread (€/MWh)")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(8, 4))
importances.plot(kind="barh", color="#2ca02c", edgecolor="black")
plt.title("Feature Importance – Random Forest", fontsize=14, weight="bold")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# ====================================================
# Save results
# ====================================================

os.makedirs("data/processed", exist_ok=True)
results = pd.DataFrame({"datetime": y_test.index, "actual": y_test, "predicted": y_pred})
results.to_csv("data/processed/spread_predictions.csv", index=False)

print("\nForecasting complete. Results saved to data/processed/spread_predictions.csv")


