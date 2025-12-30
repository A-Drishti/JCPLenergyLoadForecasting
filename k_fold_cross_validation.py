import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# SMAPE FUNCTION
def smape(true, pred):
    denom = (np.abs(true) + np.abs(pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return np.mean(np.abs(true - pred) / denom) * 100.0

# Compute metrics
def compute_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))
    mape = float(np.mean(np.abs((true - pred) / np.maximum(np.abs(true), 1e-6))) * 100.0)
    smp = float(smape(true, pred))

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE%": mape,
        "SMAPE%": smp
    }

# K-FOLD VALIDATION
def run_kfold_cv(prediction_file, k_folds=5):

    print(f"\nProcessing: {prediction_file}")

    df = pd.read_csv(prediction_file)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Must contain true and pred
    if not {"true", "pred"}.issubset(df.columns):
        raise ValueError(f"{prediction_file} missing required columns.")

    # Sort by timestamp if available
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    else:
        df = df.sort_index()

    y = df["true"].values
    yhat = df["pred"].values

    tscv = TimeSeriesSplit(n_splits=k_folds)

    fold_results = []
    fold_id = 1

    for train_idx, test_idx in tscv.split(y):
        true_fold = y[test_idx]
        pred_fold = yhat[test_idx]

        metrics = compute_metrics(true_fold, pred_fold)
        metrics["fold"] = fold_id
        fold_results.append(metrics)

        print(f"Fold {fold_id}: {metrics}")
        fold_id += 1

    return pd.DataFrame(fold_results)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing *_predictions_CRNN.csv files")
    parser.add_argument("--k_folds", type=int, default=5)
    args = parser.parse_args()

    pred_files = [f for f in os.listdir(args.pred_dir)
                  if f.endswith("_predictions_CRNN.csv")]

    if not pred_files:
        raise ValueError("No prediction CSV files found in directory.")

    all_results = []

    for f in pred_files:
        full_path = os.path.join(args.pred_dir, f)
        df_res = run_kfold_cv(full_path, k_folds=args.k_folds)
        df_res["rate_class"] = f.replace("_predictions_CRNN.csv", "")
        all_results.append(df_res)

    all_results = pd.concat(all_results, ignore_index=True)

    # Save output
    out_file = os.path.join(args.pred_dir, "kfold_results.csv")
    all_results.to_csv(out_file, index=False)

    print("\n============================================")
    print("FINAL K-FOLD RESULTS \n")
    print(all_results)

    print("\nAggregated Metrics (mean ± std):")
    summary = all_results.groupby("rate_class")[["MAE", "RMSE", "MAPE%", "SMAPE%"]].agg(["mean", "std"])
    print(summary)

    summary_file = os.path.join(args.pred_dir, "kfold_summary.csv")
    summary.to_csv(summary_file)
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()

'''
>>>python k_fold_cross_validation.py --pred_dir ./out_crnn_fix1 --k_folds 5

Processing: ./out_crnn_fix1\CIEP_predictions_CRNN.csv
Fold 1: {'MAE': 20.228373412256264, 'RMSE': 26.602452260176253, 'MAPE%': 4.4734872376319155, 'SMAPE%': 4.425245153211104, 'fold': 1}
Fold 2: {'MAE': 20.9955381545961, 'RMSE': 26.803841149416918, 'MAPE%': 4.12406239712637, 'SMAPE%': 4.102478673504179, 'fold': 2}
Fold 3: {'MAE': 20.02865902506964, 'RMSE': 25.771215445440163, 'MAPE%': 3.5517941555414456, 'SMAPE%': 3.5257739068334257, 'fold': 3}
Fold 4: {'MAE': 17.64653454735376, 'RMSE': 22.951523402419223, 'MAPE%': 3.624207780748868, 'SMAPE%': 3.582805909992416, 'fold': 4}
Fold 5: {'MAE': 18.402951448467967, 'RMSE': 23.73416864353286, 'MAPE%': 3.7617793494593688, 'SMAPE%': 3.7288177134154603, 'fold': 5}

Processing: ./out_crnn_fix1\RSCP_predictions_CRNN.csv
Fold 1: {'MAE': 50.66576114206128, 'RMSE': 77.00151660562115, 'MAPE%': 105574014.3381211, 'SMAPE%': 3.320764429973635, 'fold': 1}
Fold 2: {'MAE': 69.57793175487465, 'RMSE': 95.48851940019217, 'MAPE%': 3.124303994981371, 'SMAPE%': 3.1366045962576994, 'fold': 2}
Fold 3: {'MAE': 85.24959937325906, 'RMSE': 117.33620509613995, 'MAPE%': 3.0448736833265313, 'SMAPE%': 3.0610946930261953, 'fold': 3}
Fold 4: {'MAE': 43.541624512534824, 'RMSE': 57.855295668884935, 'MAPE%': 2.6578047361036043, 'SMAPE%': 2.646311748991784, 'fold': 4}
Fold 5: {'MAE': 46.313830222841226, 'RMSE': 59.472866478401485, 'MAPE%': 2.5951357967181585, 'SMAPE%': 2.602245418285, 'fold': 5}

============================================
FINAL K-FOLD RESULTS

         MAE        RMSE         MAPE%    SMAPE%  fold rate_class
0  20.228373   26.602452  4.473487e+00  4.425245     1       CIEP
1  20.995538   26.803841  4.124062e+00  4.102479     2       CIEP
2  20.028659   25.771215  3.551794e+00  3.525774     3       CIEP
3  17.646535   22.951523  3.624208e+00  3.582806     4       CIEP
4  18.402951   23.734169  3.761779e+00  3.728818     5       CIEP
5  50.665761   77.001517  1.055740e+08  3.320764     1       RSCP
6  69.577932   95.488519  3.124304e+00  3.136605     2       RSCP
7  85.249599  117.336205  3.044874e+00  3.061095     3       RSCP
8  43.541625   57.855296  2.657805e+00  2.646312     4       RSCP
9  46.313830   59.472866  2.595136e+00  2.602245     5       RSCP

Aggregated Metrics (mean ± std):
                  MAE                  RMSE                   MAPE%                  SMAPE%
                 mean        std       mean       std          mean           std      mean       std
rate_class
CIEP        19.460411   1.385429  25.172640   1.73681  3.907066e+00  3.857640e-01  3.873024  0.381838
RSCP        59.069749  17.820421  81.430881  25.22098  2.111481e+07  4.721413e+07  2.953404  0.315330

Saved: ./out_crnn_fix1\kfold_summary.csv
'''