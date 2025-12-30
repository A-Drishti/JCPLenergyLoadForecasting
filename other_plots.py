# other_plots.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------
# CONFIG
# ---------------------------------------
PRED_DIR = "out_crnn_fix1"

WEATHER_DIR = r"2025-Data/Data/Predict With - X/Weather/Newark"
ENERGY_DIR = r"2025-Data/Data/Predict With - X/Energy"

OUT_PLOT_DIR = "plots/other"
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUT_PLOT_DIR, name), dpi=200, bbox_inches="tight")
    plt.close()


# LOAD CRNN PREDICTIONS
def load_predictions():
    preds = {}
    for f in os.listdir(PRED_DIR):
        if f.endswith("_predictions_CRNN.csv"):
            rate = f.replace("_predictions_CRNN.csv", "")
            df = pd.read_csv(os.path.join(PRED_DIR, f))

            # find timestamp column
            dt_col = [c for c in df.columns if "time" in c.lower()][0]

            df["timestamp"] = pd.to_datetime(df[dt_col])
            df = df.sort_values("timestamp")
            preds[rate] = df

            print(f"Loaded CRNN predictions for {rate} → {df.shape}")

    return preds

# WEATHER CLEANER
def clean_weather_df(df_raw):
    df = df_raw.copy()

    # normalize datetime column
    dt_col = [c for c in df.columns if c.lower() in ("time", "date", "timestamp")][0]
    df["timestamp"] = pd.to_datetime(df[dt_col])

    # clean columns with units
    def strip_unit(col, unit):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(unit, "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    strip_unit("Temperature", "F")
    strip_unit("Dew Point", "F")
    strip_unit("Humidity", "%")
    strip_unit("Wind Speed", "mph")
    strip_unit("Wind Gust", "mph")
    strip_unit("Pressure", "in")
    strip_unit("Precipitation", "in")

    return df

# LOAD MULTI-YEAR WEATHER
def load_multi_year_weather():
    files = sorted([f for f in os.listdir(WEATHER_DIR) if f.endswith(".csv")])

    if not files:
        raise RuntimeError("No weather CSV files found in Weather FOLDER/")

    all_years = []

    for f in files:
        print("Loading weather file:", f)
        df = pd.read_csv(os.path.join(WEATHER_DIR, f))
        df = clean_weather_df(df)
        all_years.append(df)

    weather = pd.concat(all_years, ignore_index=True)
    weather = weather.sort_values("timestamp")

    # remove duplicates, preserve best resolution data
    weather = weather.drop_duplicates(subset=["timestamp"])

    print("Final merged weather shape:", weather.shape)
    print("Weather years covered:",
          weather["timestamp"].dt.year.min(), "→", weather["timestamp"].dt.year.max())

    return weather

# LOAD ENERGY MONTHLY DATA
# def load_energy():
#     all_energy = {}

#     for f in os.listdir(ENERGY_DIR):
#         if f.endswith(".csv"):
#             name = f.replace(".csv", "")
#             df = pd.read_csv(os.path.join(ENERGY_DIR, f))

#             # Standardize month column
#             month_col = df.columns[0]
#             df[month_col] = df[month_col].astype(str).str.strip()

#             # Case 1: Format like "Jun-25" → convert to "Jun-2025"
#             if df[month_col].str.contains(r"^[A-Za-z]{3}-\d{2}$").any():
#                 df[month_col] = pd.to_datetime(
#                     df[month_col].str.replace(
#                         r"-", "-20", regex=True
#                     ),
#                     format="%b-%Y"
#                 )
#             else:
#                 # Try normal parser
#                 df[month_col] = pd.to_datetime(df[month_col], errors="coerce")

#             # Report any failed rows
#             bad = df[df[month_col].isna()]
#             if len(bad) > 0:
#                 print(f"⚠ WARNING: {len(bad)} rows had invalid month values in {name}")

#             df = df.sort_values(month_col)
#             df = df.rename(columns={month_col: "Month"})

#             all_energy[name] = df
#             print("Loaded energy file:", name)

#     return all_energy
def load_energy():
    all_energy = {}

    for f in os.listdir(ENERGY_DIR):
        if f.endswith(".csv"):
            name = f.replace(".csv", "")
            df = pd.read_csv(os.path.join(ENERGY_DIR, f))

            month_col = df.columns[0]
            df[month_col] = df[month_col].astype(str).str.strip()

            # Fix formats like "Jun-25"
            if df[month_col].str.contains(r"^[A-Za-z]{3}-\d{2}$").any():
                df[month_col] = pd.to_datetime(
                    df[month_col].str.replace(r"-", "-20", regex=True),
                    format="%b-%Y",
                    errors="coerce"
                )
            else:
                df[month_col] = pd.to_datetime(df[month_col], errors="coerce")

            # Warning for any failed conversions
            if df[month_col].isna().sum() > 0:
                print(f"⚠ WARNING: {df[month_col].isna().sum()} bad month entries in {name}")

            df = df.sort_values(month_col)
            df = df.rename(columns={month_col: "Month"})

            all_energy[name] = df
            print("Loaded energy file:", name)

    return all_energy


# RATE CLASS VOLATILITY
def rate_class_volatility(predictions):
    stats = []

    for rate, df in predictions.items():
        err = df["true"] - df["pred"]

        MAD = np.mean(np.abs(err - err.mean()))  # replacement for deprecated Series.mad()

        stats.append({
            "Rate": rate,
            "StdDev": err.std(),
            "MAD": MAD,
            "IQR": err.quantile(0.75) - err.quantile(0.25),
            "RollingVol(24h)": err.rolling(24).std().mean()
        })

    result = pd.DataFrame(stats)
    result.to_csv(os.path.join(OUT_PLOT_DIR, "rate_class_volatility_summary.csv"), index=False)
    print(result)

    plt.figure(figsize=(9,5))
    sns.barplot(data=result, x="Rate", y="StdDev")
    plt.title("Prediction Error Volatility (StdDev) per Rate Class")
    savefig("rate_class_volatility_std.png")

    return result

# WEATHER EFFECT ON VOLATILITY
def weather_volatility_analysis(pred, weather, rate):
    merged = pd.merge_asof(
        pred.sort_values("timestamp"),
        weather.sort_values("timestamp"),
        on="timestamp"
    )

    merged["error"] = merged["true"] - merged["pred"]

    # correlation matrix only numerical fields
    num_cols = merged.select_dtypes(include=[np.number]).columns
    corr = merged[num_cols].corr()["error"].sort_values(ascending=False)

    plt.figure(figsize=(10,8))
    sns.heatmap(
        merged[
            ["error","Temperature","Dew Point","Humidity","Wind Speed",
             "Wind Gust","Pressure","Precipitation"]
        ].corr(),
        annot=True, cmap="coolwarm", fmt=".2f"
    )
    plt.title(f"Weather correlation with CRNN error – {rate}")
    savefig(f"{rate}_weather_correlation.png")

    return corr


# ENERGY EFFECT ON VOLATILITY
def energy_volatility_analysis(pred, energy, rate):

    df = pred.copy()
    df["Month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    # incorporate all energy datasets
    for name, edf in energy.items():
        df = df.merge(edf, on="Month", how="left", suffixes=("","_"+name))

    df["error"] = df["true"] - df["pred"]

    corr = df.corr(numeric_only=True)["error"].sort_values(ascending=False)

    plt.figure(figsize=(8,6))
    corr.head(12).plot(kind="bar")
    plt.title(f"Energy correlations with CRNN error – {rate}")
    savefig(f"{rate}_energy_correlations.png")

    return corr

# FEATURE IMPORTANCE (WEATHER + ENERGY)
def combined_feature_importance(pred, weather, energy, rate):

    df = pd.merge_asof(
        pred.sort_values("timestamp"),
        weather.sort_values("timestamp"),
        on="timestamp"
    )

    df["Month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    for name, edf in energy.items():
        df = df.merge(edf, on="Month", how="left", suffixes=("","_"+name))

    df["error"] = df["true"] - df["pred"]

    X = df.drop(columns=["timestamp","true","pred","error"])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df["error"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_scaled, y)

    imp = pd.Series(model.feature_importances_, index=X.columns)
    top = imp.sort_values(ascending=False).head(20)

    plt.figure(figsize=(10,6))
    top.plot(kind="bar")
    plt.title(f"Feature Importance (Weather + Energy) – {rate}")
    savefig(f"{rate}_feature_importance.png")

    return top

# MAIN
if __name__ == "__main__":
    print("\nLoading CRNN Predictions...")
    preds = load_predictions()

    print("\nLoading Multi-Year Weather...")
    weather = load_multi_year_weather()

    print("\nLoading Energy Data...")
    energy = load_energy()

    print("\nComputing Volatility per Rate Class...")
    vol_df = rate_class_volatility(preds)

    print("\nRunning Impact Analysis...")
    for rate, pred in preds.items():
        weather_volatility_analysis(pred, weather, rate)
        energy_volatility_analysis(pred, energy, rate)
        combined_feature_importance(pred, weather, energy, rate)

    print("\nDONE — all plots saved to:", OUT_PLOT_DIR)
