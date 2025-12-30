import os
import gc
import pickle
import traceback
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import holidays


# ===========================================================
#  FEATURE CREATION
# ===========================================================
def make_features(df):
    df = df.copy()
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # US holidays
    us_holidays = holidays.US()
    # normalize → keeps dtype = datetime64[ns], avoids dtype errors
    df["is_holiday"] = df["timestamp"].dt.normalize().isin(us_holidays).astype(int)

    # Leakage-free: lags + rolling
    df["lag_1"] = df["load"].shift(1)
    df["lag_24"] = df["load"].shift(24)
    df["lag_168"] = df["load"].shift(168)

    df["roll24_mean"] = df["load"].rolling(24).mean()
    df["roll24_std"] = df["load"].rolling(24).std()

    return df.dropna().reset_index(drop=True)


# ===========================================================
#  EVALUATION
# ===========================================================
def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}


# ===========================================================
#  PLOT SAVING
# ===========================================================
def save_plot(df, y_true, y_pred, title, out_path, max_points=500):
    plt.figure(figsize=(14, 5))
    df_plot = df.iloc[-max_points:]
    plt.plot(df_plot["timestamp"], y_true[-max_points:], label="Actual")
    plt.plot(df_plot["timestamp"], y_pred[-max_points:], label="Prediction")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ===========================================================
#  MAIN PIPELINE
# ===========================================================
def run_baseline(data_root):

    print("Loading cleaned rate class files…")
    files = [f for f in os.listdir(data_root) if f.endswith(".xlsx")]
    print(f"Found {len(files)} files")

    # Load all datasets
    data_dict = {}
    for f in files:
        df = pd.read_excel(os.path.join(data_root, f))
        df.columns = ["timestamp", "load"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        data_dict[f.replace("_cleaned.xlsx", "")] = df

    # Create save directories
    save_dir_models = os.path.join(data_root, "saved_models")
    save_dir_plots = os.path.join(data_root, "plots")
    os.makedirs(save_dir_models, exist_ok=True)
    os.makedirs(save_dir_plots, exist_ok=True)
    os.makedirs(os.path.join(save_dir_plots, "LinearReg"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_plots, "SARIMAX"), exist_ok=True)

    results_summary = []
    problem_logs = []

    # ===========================================================
    # PROCESS EACH RATE CLASS
    # ===========================================================
    for name, df in tqdm(data_dict.items(), desc="Rate classes"):
        try:
            print(f"\n🔹 Processing rate class: {name}")

            feat = make_features(df)

            # Strict YEAR split
            train_df = feat[(feat["year"] >= 2022) & (feat["year"] <= 2023)].copy()
            test_df = feat[feat["year"] == 2024].copy()

            MIN_TRAIN_DAYS = 25
            MIN_TRAIN_LEN = MIN_TRAIN_DAYS * 24  # 600 hours

            if len(train_df) < MIN_TRAIN_LEN:
                msg = f"Skipping {name}: insufficient training data ({len(train_df)} < {MIN_TRAIN_LEN})"
                print(msg)
                problem_logs.append({"rate_class": name, "error": msg})
                continue


            if len(test_df) < 24 * 7:
                msg = f"Skipping {name}: insufficient test data"
                print(msg)
                problem_logs.append({"rate_class": name, "error": msg})
                continue

            # Features
            X_cols = [c for c in feat.columns if c not in ["timestamp", "load", "year"]]

            X_train, y_train = train_df[X_cols], train_df["load"]
            X_test, y_test = test_df[X_cols], test_df["load"]

            # =======================================================
            #  LINEAR REGRESSION (no leakage)
            # =======================================================
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # fit only on training
            X_test_scaled = scaler.transform(X_test)         # transform test safely

            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred_lr = lr.predict(X_test_scaled)

            metrics_lr = evaluate(y_test, y_pred_lr)
            print("  • Linear Regression:", metrics_lr)

            # Save LR model + scaler
            with open(os.path.join(save_dir_models, f"{name}_LR.pkl"), "wb") as f:
                pickle.dump({"model": lr, "scaler": scaler}, f)

            # Save plot
            save_plot(
                test_df,
                y_test.values,
                y_pred_lr,
                f"{name} — Linear Regression (2024)",
                os.path.join(save_dir_plots, "LinearReg", f"{name}_LR.png")
            )

            # =======================================================
            #  SARIMAX
            # =======================================================
            train_series = train_df.set_index("timestamp")["load"].asfreq("H")
            test_series = test_df.set_index("timestamp")["load"].asfreq("H")

            y_pred_arima = None
            metrics_arima = None

            try:
                model = sm.tsa.statespace.SARIMAX(
                    train_series,
                    order=(2, 1, 2),
                    seasonal_order=(1, 1, 1, 24),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                arima_fit = model.fit(disp=False)
                y_pred_arima = arima_fit.forecast(steps=len(test_series))
                metrics_arima = evaluate(test_series.values, y_pred_arima)

                print("  • SARIMAX:", metrics_arima)

                # Save model
                with open(os.path.join(save_dir_models, f"{name}_SARIMAX.pkl"), "wb") as f:
                    pickle.dump(arima_fit, f)

                # Save plot
                save_plot(
                    test_df,
                    test_series.values,
                    y_pred_arima,
                    f"{name} — SARIMAX (2024)",
                    os.path.join(save_dir_plots, "SARIMAX", f"{name}_SARIMAX.png")
                )

            except Exception as e_ar:
                tb = traceback.format_exc()
                msg = f"SARIMAX failed for {name}: {e_ar}"
                print("  " + msg)
                problem_logs.append({"rate_class": name, "error": msg, "traceback": tb})

            # Store results
            results_summary.append({
                "rate_class": name,
                "model": "Linear Regression",
                **metrics_lr
            })

            if metrics_arima is not None:
                results_summary.append({
                    "rate_class": name,
                    "model": "SARIMAX",
                    **metrics_arima
                })

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error in {name}: {e}")
            problem_logs.append({"rate_class": name, "error": str(e), "traceback": tb})

        finally:
            gc.collect()

    # Save results to CSV
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(data_root, "Baseline_Results.csv"), index=False)

    if problem_logs:
        pd.DataFrame(problem_logs).to_csv(os.path.join(data_root, "Baseline_Problems.csv"), index=False)

    print("\nSaved results to Baseline_Results.csv")
    print("Saved problem logs to Baseline_Problems.csv")


# ===========================================================
#  RUN LOCALLY
# ===========================================================
if __name__ == "__main__":
    DATA_PATH = r"./2025-Data/Data/cleaned_rate_class_data"
    run_baseline(DATA_PATH)
