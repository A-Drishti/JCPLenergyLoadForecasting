# # Run
# python crnn_reduce_error.py --data_path ./2025-Data/Data/cleaned_rate_class_data --out_dir ./out_crnn_fix --plots_dir ./plots/CRNN --seq_len 168 --batch_size 64 --epochs 40
# # For light tuning...
# python crnn_reduce_error.py --... --tune --tune_trials 8 --tune_epochs 5
#!/usr/bin/env python3
"""
crnn_unified_fixed.py

Unified CRNN (Conv1D -> GRU -> Dense) CPU-only forecasting script (fixed).
- Fits scaler per-rate using ONLY training rows (2022-2023).
- Tests on 2024 (or fallback last X days).
- Detects timestamp column automatically (handles "Date" with DD-MM-YYYY HH:MM:SS).
- Saves models, plots and per-rate CSVs.
"""
import os
import glob
import argparse
import math
import random
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

# Optional: optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")

# -----------------------
# DEFAULTS
# -----------------------
DEFAULTS = dict(
    data_path="./2025-Data/Data/cleaned_rate_class_data",
    out_dir="./out_crnn1",
    plots_dir="./plots/CRNN1",
    seq_len=168,
    horizon=1,
    batch_size=64,
    epochs=40,
    lr=1e-3,
    weight_decay=1e-6,
    early_stopping=8,
    min_train_days_per_month=25,
    train_years=(2022, 2023),
    test_year=2024,
    fallback_test_days=30,
    num_workers=0,
    seed=42,
    embed_dim=8,
    cnn_channels=48,
    cnn_kernel=3,
    rnn_hidden=128,
    rnn_layers=2,
    rnn_bidir=False,
    dropout=0.1,
    tune_trials=20,
    tune_epochs=8,
)

DEVICE = torch.device("cpu")  # CPU-only

# -----------------------
# Utilities
# -----------------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-6
    return np.mean(100.0 * np.abs(y_pred - y_true) / denom)

def evaluate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MAE": None, "RMSE": None, "MAPE%": None, "SMAPE%": None}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    # mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)
    mape = float(
    100.0 * np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ))
    sm = float(smape(y_true, y_pred))
    return {"MAE": float(mae), "RMSE": rmse, "MAPE%": mape, "SMAPE%": sm}

# -----------------------
# Feature engineering
# -----------------------
def make_features(df_in):
    """
    Input: dataframe with 'timestamp' column (datetime) and 'load'.
    Returns: dataframe with calendar + lag + rolling features.
    """
    df = df_in.copy()
    # ensure timestamp column exists
    if "timestamp" not in df.columns:
        raise ValueError("make_features expects a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df = df.set_index("timestamp").asfreq("H")
    # fill load conservatively
    df["load"] = df["load"].interpolate(method="time", limit=6).ffill().bfill()
    # calendar features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # lags
    for lag in (1, 24, 168):
        df[f"lag_{lag}"] = df["load"].shift(lag)
    # rolling (shifted)
    for w in (24, 168):
        df[f"roll_mean_{w}"] = df["load"].shift(1).rolling(window=w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["load"].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    df = df.reset_index().dropna().reset_index(drop=True)
    return df

def create_windows_from_df(df, feature_cols, seq_len=168, horizon=1):
    vals = df[feature_cols + ["load"]].values
    n = len(df)
    Xs, ys = [], []
    for end_idx in range(seq_len, n - horizon + 1):
        start_idx = end_idx - seq_len
        Xs.append(vals[start_idx:end_idx, :-1])
        ys.append(vals[end_idx + horizon - 1, -1])
    if not Xs:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.stack(Xs), np.array(ys)

# -----------------------
# Data completeness check
# -----------------------
def rate_has_full_years(df, years, min_days_per_month=25):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for y in years:
        sub = df[df["timestamp"].dt.year == y]
        months_present = set(sub["timestamp"].dt.month.unique())
        if len(months_present) < 12:
            return False
        for m in range(1, 13):
            s2 = sub[sub["timestamp"].dt.month == m]
            if s2.empty:
                return False
            days = s2["timestamp"].dt.floor("D").nunique()
            if days < min_days_per_month:
                return False
    return True

# -----------------------
# Dataset & Model (GRU)
# -----------------------
class UnifiedDataset(Dataset):
    def __init__(self, X, y, rate_ids):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.rate_ids = rate_ids.astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.rate_ids[idx], self.y[idx]

class UnifiedCRNN_GRU(nn.Module):
    def __init__(self, input_dim, n_rates, embed_dim=8, cnn_channels=48, kernel_size=3,
                 rnn_hidden=128, rnn_layers=2, bidir=False, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.embed = nn.Embedding(n_rates, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim + embed_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=rnn_hidden,
                          num_layers=rnn_layers, batch_first=True, bidirectional=bidir, dropout=dropout if rnn_layers>1 else 0)
        rnn_out_dim = rnn_hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x, rate_ids):
        emb = self.embed(rate_ids)
        emb_seq = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        xcat = torch.cat([x, emb_seq], dim=2)
        xperm = xcat.permute(0,2,1)
        c = self.conv(xperm)
        c = c.permute(0,2,1)
        out, _ = self.rnn(c)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)

# -----------------------
# Training helpers
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    count = 0
    for Xb, rateb, yb in loader:
        Xb = Xb.to(device); rateb = rateb.to(device); yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb, rateb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running += loss.item() * Xb.size(0)
        count += Xb.size(0)
    return running / max(1, count)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    count = 0
    with torch.no_grad():
        for Xb, rateb, yb in loader:
            Xb = Xb.to(device); rateb = rateb.to(device); yb = yb.to(device)
            preds = model(Xb, rateb)
            loss = criterion(preds, yb)
            running += loss.item() * Xb.size(0)
            count += Xb.size(0)
    return running / max(1, count)

def eval_preds(model, loader, device):
    model.eval()
    all_preds = []; all_trues = []
    with torch.no_grad():
        for Xb, rateb, yb in loader:
            Xb = Xb.to(device); rateb = rateb.to(device)
            preds = model(Xb, rateb).cpu().numpy()
            all_preds.append(preds); all_trues.append(yb.numpy())
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds); all_trues = np.concatenate(all_trues)
    else:
        all_preds = np.array([]); all_trues = np.array([])
    return all_trues, all_preds

# -----------------------
# Main pipeline
# -----------------------
def run_unified(
    data_path,
    out_dir,
    plots_dir,
    seq_len=168,
    horizon=1,
    batch_size=64,
    epochs=40,
    lr=1e-3,
    weight_decay=1e-6,
    early_stopping=8,
    min_train_days_per_month=25,
    train_years=(2022,2023),
    test_year=2024,
    fallback_test_days=30,
    num_workers=0,
    seed=42,
    embed_dim=8,
    cnn_channels=48,
    cnn_kernel=3,
    rnn_hidden=128,
    rnn_layers=2,
    rnn_bidir=False,
    dropout=0.1,
    tune=False,
    tune_trials=20,
    tune_epochs=8
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ---------- find files ----------
    xlsx_files = sorted(glob.glob(os.path.join(data_path, "*_cleaned.xlsx")) + glob.glob(os.path.join(data_path, "*.xlsx")))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found under {data_path}")

    # load into dict keyed by rate name (deduplicate duplicates)
    data_dict = {}
    problems = []
    for fp in xlsx_files:
        try:
            base = os.path.basename(fp)
            rate = base.replace("_cleaned.xlsx", "").replace(".xlsx", "")
            if rate in data_dict:
                continue
            df = pd.read_excel(fp)
            # lowercase headers to make detection easier
            df.columns = [c if isinstance(c, str) else c for c in df.columns]
            # detect timestamp column (robust)
            possible_ts = [c for c in df.columns if str(c).lower() in {"timestamp","datetime","date","time"}]
            # fallback to common names including "Date"
            if not possible_ts:
                for c in df.columns:
                    if str(c).strip().lower() in ("date","datetime","timestamp","time","date_time"):
                        possible_ts.append(c)
            if not possible_ts:
                # try find column with 'date' substring
                for c in df.columns:
                    if "date" in str(c).lower() or "time" in str(c).lower():
                        possible_ts.append(c)
            if not possible_ts:
                raise ValueError(f"No timestamp-like column found in {fp}. Columns: {list(df.columns)}")
            ts_col = possible_ts[0]

            # detect load column
            possible_load = [c for c in df.columns if str(c).lower() in {"load","demand","value","kwh"}]
            if not possible_load:
                # fallback: second column if first is timestamp-like
                cand = [c for c in df.columns if c != ts_col]
                if cand:
                    possible_load = [cand[0]]
            if not possible_load:
                raise ValueError(f"No load-like column found in {fp}. Columns: {list(df.columns)}")
            load_col = possible_load[0]

            # normalize column names
            df = df.rename(columns={ts_col: "timestamp", load_col: "load"})
            # parse timestamp: dayfirst True because your data is DD-MM-YYYY
            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
            if df["timestamp"].isna().any():
                # try infer if parsing failed; raise for visibility
                raise ValueError(f"Timestamp parsing failed for some rows in {fp}. Check format (expected DD-MM-YYYY HH:MM:SS or similar).")
            df = df[["timestamp","load"]].copy()
            df["load"] = pd.to_numeric(df["load"], errors="coerce")
            df = df.dropna(subset=["load","timestamp"]).reset_index(drop=True)
            data_dict[rate] = df
        except Exception as e:
            problems.append({"file": fp, "error": str(e)})

    if len(data_dict) == 0:
        raise RuntimeError("No valid rate-class files loaded.")

    print(f"Found {len(data_dict)} unique rate-class files")

    # ---------- completeness filtering ----------
    years_to_check = tuple(sorted(set(list(train_years) + [test_year])))
    good_rates = []
    for rate, df in data_dict.items():
        try:
            ok = rate_has_full_years(df, years=years_to_check, min_days_per_month=min_train_days_per_month)
            if ok:
                good_rates.append(rate)
            else:
                problems.append({"rate": rate, "error": "insufficient full-year coverage (12 months × >= min days/month)"})
        except Exception as e:
            problems.append({"rate": rate, "error": f"completeness check failed: {e}"})

    if len(good_rates) == 0:
        raise RuntimeError("No rates passed completeness checks. See problems.")

    # reduce to good rates
    data_dict = {r: data_dict[r] for r in good_rates}
    print("Using rates:", list(data_dict.keys()))

    # label encoding
    le = LabelEncoder()
    le.fit(list(data_dict.keys()))
    rate_to_id = {r: int(le.transform([r])[0]) for r in data_dict.keys()}

    # per-rate processing: create features, split, fit scaler on train only, scale both, build windows
    feature_columns_order = None
    all_train_X = []; all_train_y = []; all_train_rid = []
    all_test_X = []; all_test_y = []; all_test_rid = []
    per_rate_scalers = {}
    per_rate_info = {}

    for rate, df in tqdm(data_dict.items(), desc="per-rate processing"):
        try:
            # build features
            df_feat = make_features(df.rename(columns={"timestamp":"timestamp", "load":"load"}))
            df_feat["year"] = df_feat["timestamp"].dt.year

            # train/test split by year
            train_df = df_feat[df_feat["year"].isin(train_years)].copy()
            test_df = df_feat[df_feat["year"] == test_year].copy()

            # fallback test period if no 2024
            if len(test_df) == 0:
                fallback_hours = 24 * fallback_test_days
                if len(df_feat) > fallback_hours + seq_len:
                    cutoff_time = df_feat["timestamp"].iloc[-fallback_hours]
                    test_df = df_feat[df_feat["timestamp"] >= cutoff_time].copy()
                    train_df = df_feat[df_feat["timestamp"] < cutoff_time].copy()
                else:
                    problems.append({"rate": rate, "error": "no 2024 data and not enough fallback data"})
                    continue

            if len(train_df) < 24 * 25:  # require at least ~25 days
                problems.append({"rate": rate, "error": f"insufficient train rows ({len(train_df)})"})
                continue

            # define feature columns (exclude timestamp/load/year)
            cols = [c for c in train_df.columns if c not in ("timestamp","load","year")]
            # ensure consistent order across rates
            if feature_columns_order is None:
                feature_columns_order = cols
            else:
                for c in feature_columns_order:
                    if c not in cols:
                        train_df[c] = 0.0
                        test_df[c] = 0.0
                # preserve feature_columns_order columns only (plus timestamp/load/year)
                train_df = train_df[["timestamp","load","year"] + feature_columns_order]
                test_df = test_df[["timestamp","load","year"] + feature_columns_order]

            # fit scaler on train features only
            scaler = StandardScaler()
            scaler.fit(train_df[feature_columns_order].values)
            per_rate_scalers[rate] = scaler

            # transform and replace
            Xtr_feats = scaler.transform(train_df[feature_columns_order].values)
            Xte_feats = scaler.transform(test_df[feature_columns_order].values)
            train_scaled = train_df[["timestamp","load","year"]].copy()
            test_scaled = test_df[["timestamp","load","year"]].copy()
            for i, c in enumerate(feature_columns_order):
                train_scaled[c] = Xtr_feats[:, i]
                test_scaled[c] = Xte_feats[:, i]

            # create windows per split
            Xtr, ytr = create_windows_from_df(train_scaled, feature_columns_order, seq_len=seq_len, horizon=horizon)
            Xte, yte = create_windows_from_df(test_scaled, feature_columns_order, seq_len=seq_len, horizon=horizon)

            if Xtr.size == 0:
                problems.append({"rate": rate, "error": "no train windows after windowing"})
                continue

            all_train_X.append(Xtr); all_train_y.append(ytr); all_train_rid.append(np.full((Xtr.shape[0],), rate_to_id[rate], dtype=np.int64))
            if Xte.size > 0:
                all_test_X.append(Xte); all_test_y.append(yte); all_test_rid.append(np.full((Xte.shape[0],), rate_to_id[rate], dtype=np.int64))

            per_rate_info[rate] = {"n_train_rows": len(train_df), "n_test_rows": len(test_df)}
        except Exception as e:
            problems.append({"rate": rate, "error": str(e)})
            continue

    if len(all_train_X) == 0:
        raise RuntimeError("No training windows created. Abort.")

    X_train = np.vstack(all_train_X)
    y_train = np.concatenate(all_train_y)
    rate_train = np.concatenate(all_train_rid)

    if len(all_test_X) > 0:
        X_test = np.vstack(all_test_X)
        y_test = np.concatenate(all_test_y)
        rate_test = np.concatenate(all_test_rid)
    else:
        X_test = np.empty((0, seq_len, len(feature_columns_order)))
        y_test = np.empty((0,))
        rate_test = np.empty((0,))

    print(f"Train samples: {X_train.shape[0]}  Test samples: {X_test.shape[0]}  Feature dim: {len(feature_columns_order)}")

    # persist scalers & label encoder
    with open(os.path.join(out_dir, "scalers_label_encoder.pkl"), "wb") as f:
        pickle.dump({"scalers": per_rate_scalers, "label_encoder": le, "feature_columns": feature_columns_order}, f)

    # -----------------------
    # Build dataloaders
    # -----------------------
    train_ds = UnifiedDataset(X_train, y_train, rate_train)
    test_ds = UnifiedDataset(X_test, y_test, rate_test)

    train_size = int(len(train_ds) * 0.9)
    val_size = len(train_ds) - train_size
    if train_size <= 0:
        raise RuntimeError("Not enough training windows to split into train/val")

    train_sub, val_sub = random_split(train_ds, [train_size, val_size])
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -----------------------
    # Model init
    # -----------------------
    n_rates = len(le.classes_)
    model = UnifiedCRNN_GRU(input_dim=len(feature_columns_order), n_rates=n_rates, embed_dim=embed_dim,
                            cnn_channels=cnn_channels, kernel_size=cnn_kernel,
                            rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                            bidir=rnn_bidir, dropout=dropout).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # -----------------------
    # (Optional) tuning (light)
    # -----------------------
    if tune and OPTUNA_AVAILABLE:
        def objective(trial):
            h = {
                "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
                "weight_decay": trial.suggest_loguniform("weight_decay", 1e-8, 1e-4),
                "cnn_channels": int(trial.suggest_categorical("cnn_channels", [32, 48])),
                "rnn_hidden": int(trial.suggest_categorical("rnn_hidden", [64, 128])),
                "embed_dim": int(trial.suggest_categorical("embed_dim", [4, 8])),
                "dropout": float(trial.suggest_uniform("dropout", 0.0, 0.3)),
            }
            # small train for tuning
            m = UnifiedCRNN_GRU(input_dim=len(feature_columns_order), n_rates=n_rates,
                                embed_dim=h["embed_dim"], cnn_channels=h["cnn_channels"],
                                kernel_size=cnn_kernel, rnn_hidden=h["rnn_hidden"],
                                rnn_layers=rnn_layers, bidir=rnn_bidir, dropout=h["dropout"]).to(DEVICE)
            opt = torch.optim.Adam(m.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
            crit = nn.MSELoss()
            best_val = float("inf"); patience_cnt = 0
            for e in range(1, tune_epochs + 1):
                train_one_epoch(m, train_loader, opt, crit, DEVICE)
                v = eval_one_epoch(m, val_loader, crit, DEVICE)
                if v < best_val - 1e-9:
                    best_val = v; patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= 3:
                        break
            return best_val

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_trials)
        best_trial = study.best_trial.params
        # update hyperparams from best_trial
        lr = best_trial.get("lr", lr)
        weight_decay = best_trial.get("weight_decay", weight_decay)
        cnn_channels = int(best_trial.get("cnn_channels", cnn_channels))
        rnn_hidden = int(best_trial.get("rnn_hidden", rnn_hidden))
        embed_dim = int(best_trial.get("embed_dim", embed_dim))
        dropout = float(best_trial.get("dropout", dropout))
        # rebuild model with tuned params
        model = UnifiedCRNN_GRU(input_dim=len(feature_columns_order), n_rates=n_rates, embed_dim=embed_dim,
                                cnn_channels=cnn_channels, kernel_size=cnn_kernel,
                                rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                                bidir=rnn_bidir, dropout=dropout).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -----------------------
    # Training loop
    # -----------------------
    best_val = float("inf"); patience = 0
    train_hist, val_hist = [], []
    start_time = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = eval_one_epoch(model, val_loader, criterion, DEVICE)
        train_hist.append(train_loss); val_hist.append(val_loss)
        print(f"Epoch {ep}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={(time.time()-t0):.1f}s")
        if val_loss < best_val - 1e-9:
            best_val = val_loss; patience = 0; best_state = model.state_dict()
            torch.save({"model_state": best_state, "feature_columns": feature_columns_order, "label_encoder": le}, os.path.join(out_dir, "unified_crnn_best.pth"))
        else:
            patience += 1
            if patience >= early_stopping:
                print("Early stopping triggered.")
                break

    total_minutes = (time.time() - start_time) / 60.0
    print(f"Training finished in {total_minutes:.1f} minutes. Best val MSE: {best_val:.6f}")

    # save training curve
    plt.figure(figsize=(8,4)); plt.plot(train_hist, label="train"); plt.plot(val_hist, label="val"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("CRNN Unified Training Loss")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "crnn_training_loss.png")); plt.close()

    # ensure best checkpoint present
    ckpt_fp = os.path.join(out_dir, "unified_crnn_best.pth")
    if not os.path.exists(ckpt_fp):
        torch.save({"model_state": model.state_dict(), "feature_columns": feature_columns_order, "label_encoder": le}, ckpt_fp)

    # -----------------------
    # Evaluate on test set and save per-rate outputs
    # -----------------------
    trues, preds = eval_preds(model, test_loader, DEVICE)
    metrics = evaluate_metrics(trues, preds)
    print("Overall test metrics:", metrics)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "crnn_unified_metrics_summary.csv"), index=False)

    # Reconstruct per-rate timestamps to save CSVs and plots
    idx = 0
    for rate in sorted(data_dict.keys()):
        scaler = per_rate_scalers.get(rate, None)
        if scaler is None:
            continue
        df_raw = data_dict[rate].copy()
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        feat = make_features(df_raw.rename(columns={"timestamp":"timestamp","load":"load"}))
        feat["year"] = feat["timestamp"].dt.year
        test_df = feat[feat["year"] == test_year].copy()
        if test_df.empty:
            # fallback
            fallback_hours = 24 * fallback_test_days
            if len(feat) > fallback_hours + seq_len:
                cutoff_time = feat["timestamp"].iloc[-fallback_hours]
                test_df = feat[feat["timestamp"] >= cutoff_time].copy()
            else:
                continue
        # ensure feature columns exist
        for c in feature_columns_order:
            if c not in test_df.columns:
                test_df[c] = 0.0
        Xte_feats = scaler.transform(test_df[feature_columns_order].values)
        test_scaled = test_df[["timestamp","load","year"]].copy()
        for i,c in enumerate(feature_columns_order):
            test_scaled[c] = Xte_feats[:, i]
        Xr, yr = create_windows_from_df(test_scaled, feature_columns_order, seq_len=seq_len, horizon=horizon)
        if Xr.size == 0:
            continue
        target_idx_start = seq_len + horizon - 1
        target_ts = test_scaled["timestamp"].iloc[target_idx_start: target_idx_start + len(yr)].reset_index(drop=True)
        n_this = len(yr)
        if idx + n_this > len(preds):
            n_this = max(0, len(preds) - idx)
        if n_this == 0:
            continue
        preds_slice = preds[idx: idx + n_this]
        trues_slice = trues[idx: idx + n_this]
        ts_slice = target_ts.iloc[:n_this].reset_index(drop=True)
        idx += n_this
        out_df = pd.DataFrame({"timestamp": ts_slice, "true": trues_slice, "pred": preds_slice})
        out_df.to_csv(os.path.join(out_dir, f"{rate}_predictions_CRNN.csv"), index=False)
        # plots
        plt.figure(figsize=(12,3)); plt.plot(out_df["timestamp"], out_df["true"], label="true"); plt.plot(out_df["timestamp"], out_df["pred"], label="pred"); plt.title(f"{rate} — Unified CRNN Forecast (Test)"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{rate}_forecast_CRNN.png")); plt.close()
        resid = out_df["true"] - out_df["pred"]; plt.figure(figsize=(10,2)); plt.plot(out_df["timestamp"], resid); plt.title(f"{rate} residuals"); plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{rate}_residuals_CRNN.png")); plt.close()

    # save problems log
    if problems:
        pd.DataFrame(problems).to_csv(os.path.join(out_dir, "crnn_unified_problems.csv"), index=False)

    print("Saved outputs to:", out_dir)
    print("Saved plots to:", plots_dir)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULTS["data_path"])
    parser.add_argument("--out_dir", default=DEFAULTS["out_dir"])
    parser.add_argument("--plots_dir", default=DEFAULTS["plots_dir"])
    parser.add_argument("--seq_len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--horizon", type=int, default=DEFAULTS["horizon"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--early_stopping", type=int, default=DEFAULTS["early_stopping"])
    parser.add_argument("--min_train_days_per_month", type=int, default=DEFAULTS["min_train_days_per_month"])
    parser.add_argument("--fallback_test_days", type=int, default=DEFAULTS["fallback_test_days"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--embed_dim", type=int, default=DEFAULTS["embed_dim"])
    parser.add_argument("--cnn_channels", type=int, default=DEFAULTS["cnn_channels"])
    parser.add_argument("--cnn_kernel", type=int, default=DEFAULTS["cnn_kernel"])
    parser.add_argument("--rnn_hidden", type=int, default=DEFAULTS["rnn_hidden"])
    parser.add_argument("--rnn_layers", type=int, default=DEFAULTS["rnn_layers"])
    parser.add_argument("--rnn_bidir", action="store_true")
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune_trials", type=int, default=DEFAULTS["tune_trials"])
    parser.add_argument("--tune_epochs", type=int, default=DEFAULTS["tune_epochs"])
    args = parser.parse_args()

    run_unified(
        data_path=args.data_path,
        out_dir=args.out_dir,
        plots_dir=args.plots_dir,
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        min_train_days_per_month=args.min_train_days_per_month,
        train_years=DEFAULTS["train_years"],
        test_year=DEFAULTS["test_year"],
        fallback_test_days=args.fallback_test_days,
        num_workers=args.num_workers,
        seed=args.seed,
        embed_dim=args.embed_dim,
        cnn_channels=args.cnn_channels,
        cnn_kernel=args.cnn_kernel,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        rnn_bidir=args.rnn_bidir,
        dropout=args.dropout,
        tune=args.tune,
        tune_trials=args.tune_trials,
        tune_epochs=args.tune_epochs,
    )


'''
>>>python crnn_reduce_error.py --data_path ./2025-Data/Data/cleaned_rate_class_data --out_dir ./out_crnn_fix1 --plots_dir ./plots/CRNN1 --seq_len 168 --batch_size 64 --epochs 40
Found 11 unique rate-class files
Using rates: ['CIEP', 'RSCP']
per-rate processing: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.69it/s]
Train samples: 34704  Test samples: 17232  Feature dim: 11
Epoch 1/40  train_loss=1328217.194010  val_loss=453422.460566  time=163.7s
Epoch 2/40  train_loss=258379.749493  val_loss=29655.114444  time=184.8s
Epoch 3/40  train_loss=23169.006728  val_loss=11423.833565  time=182.7s
Epoch 4/40  train_loss=17285.268636  val_loss=10921.382391  time=181.2s
Epoch 5/40  train_loss=15543.142141  val_loss=7498.654308  time=183.4s
Epoch 6/40  train_loss=14450.792146  val_loss=6841.705734  time=184.2s
Epoch 7/40  train_loss=14619.848431  val_loss=7913.364487  time=178.2s
Epoch 8/40  train_loss=13601.421313  val_loss=6729.992248  time=176.9s
Epoch 9/40  train_loss=12639.032452  val_loss=6166.394125  time=179.2s
Epoch 10/40  train_loss=12403.808282  val_loss=8656.020396  time=179.1s
Epoch 11/40  train_loss=12309.698907  val_loss=6297.750314  time=181.4s
Epoch 12/40  train_loss=11997.071307  val_loss=5498.098169  time=188.7s
Epoch 13/40  train_loss=11359.594149  val_loss=6575.196906  time=273.8s
Epoch 14/40  train_loss=11408.204659  val_loss=4991.479056  time=367.9s
Epoch 15/40  train_loss=11374.066930  val_loss=4878.342147  time=363.1s
Epoch 16/40  train_loss=11261.829140  val_loss=4141.644111  time=365.9s
Epoch 17/40  train_loss=11433.694462  val_loss=6779.205895  time=368.8s
Epoch 18/40  train_loss=11654.056076  val_loss=5816.273565  time=369.1s
Epoch 19/40  train_loss=12713.260507  val_loss=5884.213234  time=349.1s
Epoch 20/40  train_loss=12340.877468  val_loss=4052.543291  time=370.2s
Epoch 21/40  train_loss=12339.911378  val_loss=7309.124085  time=373.1s
Epoch 22/40  train_loss=12226.779078  val_loss=4683.390917  time=381.9s
Epoch 23/40  train_loss=12217.124293  val_loss=4202.284893  time=385.9s
Epoch 24/40  train_loss=11733.610126  val_loss=4044.665073  time=400.6s
Epoch 25/40  train_loss=12087.458932  val_loss=3994.374109  time=402.6s
Epoch 26/40  train_loss=11830.522155  val_loss=4450.877178  time=407.1s
Epoch 27/40  train_loss=12001.601883  val_loss=3820.504619  time=396.9s
Epoch 28/40  train_loss=11766.617778  val_loss=4266.712684  time=398.9s
Epoch 29/40  train_loss=11592.284798  val_loss=4576.123295  time=177.2s
Epoch 30/40  train_loss=11789.854410  val_loss=4036.830844  time=169.1s
Epoch 31/40  train_loss=11753.227513  val_loss=3680.649756  time=164.0s
Epoch 32/40  train_loss=11179.401158  val_loss=4382.643215  time=163.8s
Epoch 33/40  train_loss=11692.386141  val_loss=4427.735193  time=163.0s
Epoch 34/40  train_loss=11262.872309  val_loss=3483.247407  time=165.6s
Epoch 35/40  train_loss=11348.399221  val_loss=5761.715449  time=162.5s
Epoch 36/40  train_loss=11247.798619  val_loss=4460.921320  time=163.5s
Epoch 37/40  train_loss=10891.668728  val_loss=3776.555327  time=164.0s
Epoch 38/40  train_loss=11475.413548  val_loss=3701.773320  time=166.5s
Epoch 39/40  train_loss=11117.225341  val_loss=4388.746106  time=163.5s
Epoch 40/40  train_loss=10971.292561  val_loss=3491.598351  time=168.7s
Training finished in 168.8 minutes. Best val MSE: 3483.247407
Overall test metrics: {'MAE': 37.82216262817383, 'RMSE': 59.72547149658203, 'MAPE%': 1.6749869585037231, 'SMAPE%': 3.3499743938446045}
Saved outputs to: ./out_crnn_fix1
Saved plots to: ./plots/CRNN1
'''
