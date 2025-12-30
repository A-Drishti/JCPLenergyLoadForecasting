import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# CONFIGURATION
# ------------------------
results_folder = r"out_crnn_fix1"   # update if needed
output_folder = results_folder

print(f"\n📁 Reading results from: {results_folder}\n")

# ------------------------
# STORAGE STRUCTURES
# ------------------------
metrics_rows = []
prediction_dfs = {}

# ------------------------
# SCAN FOLDER FOR FILES
# ------------------------
for filename in os.listdir(results_folder):

    file_path = os.path.join(results_folder, filename)

    if filename.endswith("unified_metrics_summary.csv"):
        print(f"📄 Loading metrics summary → {filename}")

        try:
            df = pd.read_csv(file_path)

            # Normalize column names
            df.columns = [col.replace(" ", "").replace("%", "") for col in df.columns]

            metrics_rows.append({
                "dataset": "FULL_MODEL",
                "MAE": float(df["MAE"].iloc[0]),
                "RMSE": float(df["RMSE"].iloc[0]),
                "MAPE": float(df["MAPE"].iloc[0]),
                "SMAPE": float(df["SMAPE"].iloc[0])
            })

        except Exception as e:
            print(f"⚠ Error reading metrics: {e}")

    elif filename.endswith("_predictions_CRNN.csv"):
        print(f"📄 Loading prediction file → {filename}")

        try:
            df_pred = pd.read_csv(file_path)
            df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])

            rate_class = filename.replace("_predictions_CRNN.csv", "")
            prediction_dfs[rate_class] = df_pred

        except Exception as e:
            print(f"⚠ Error reading predictions: {e}")

    elif filename == "kfold_results.csv":
        print(f"📄 Loading kfold results → {filename}")

        df_kfold = pd.read_csv(file_path)
        df_kfold.columns = [col.replace("%", "").replace(" ", "") for col in df_kfold.columns]

        for _, row in df_kfold.iterrows():
            metrics_rows.append({
                "dataset": row.get("rate_class", "unknown"),
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "MAPE": row["MAPE"],
                "SMAPE": row["SMAPE"]
            })

print("\n✔ File scanning complete.")

# ------------------------
# BUILD SUMMARY DATAFRAME
# ------------------------
if not metrics_rows:
    raise ValueError("❌ No usable metric data found. Check files!")

df_summary = pd.DataFrame(metrics_rows)

print("\n📌 Summary of Loaded Metrics:\n", df_summary)

# ------------------------
# SAVE SUMMARY
# ------------------------
summary_csv = os.path.join(output_folder, "CRNN_Aggregated_Model_Summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"\n💾 Summary saved: {summary_csv}")


# ------------------------
# PLOTS
# ------------------------

# ---- Bar Chart of Mean Metrics ----
df_mean = df_summary.groupby("dataset").mean().reset_index()

plt.figure(figsize=(10, 5))
df_mean.set_index("dataset")[["MAE", "RMSE", "MAPE", "SMAPE"]].plot(kind='bar')
plt.title("CRNN Model Performance Across Datasets")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "CRNN_barplot.png"))
plt.show()

# ---- Heatmap ----
plt.figure(figsize=(6, 5))
sns.heatmap(df_mean.set_index("dataset"), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("CRNN Performance Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "CRNN_heatmap.png"))
plt.show()

print("\n📊 Performance plots saved.")

# ---- Time-Series Prediction Plots ----
for rate_class, df_pred in prediction_dfs.items():
    plt.figure(figsize=(12, 5))
    plt.plot(df_pred["timestamp"], df_pred["true"], label="True", linewidth=2)
    plt.plot(df_pred["timestamp"], df_pred["pred"], label="Predicted", linestyle="--")
    plt.title(f"CRNN Predictions Over Time: {rate_class}")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{rate_class}_timeseries.png"))
    plt.close()

print("\n📈 Time-series prediction plots generated.")

print("\n🎉 ALL DONE — Your CRNN result analysis visuals are ready!")
