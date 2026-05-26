import os
import glob
import json
import argparse
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error


STATE_COLS = [
    "Vy", "AVz", "Yaw", "Beta",
    "Ax", "Ay", "AVx", "Roll"
]

CONTROL_COLS = [
    "Steer", "Vx"
]

FEATURE_COLS = STATE_COLS + CONTROL_COLS


class DynamicsDataset(Dataset):
    def __init__(self, X, y, meta=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.meta = meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx


class MCDropoutDynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim=8, hidden_dim=256, dropout_p=0.10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def read_episode(file_path):
    df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_episode_samples(df, file_name, history_len=20, stride=10):
    features = df[FEATURE_COLS].values.astype(np.float32)
    states = df[STATE_COLS].values.astype(np.float32)

    X_list = []
    y_list = []
    meta_list = []

    for i in range(history_len, len(df) - 1, stride):
        hist = features[i - history_len:i + 1]
        target = states[i + 1]

        X_list.append(hist.reshape(-1))
        y_list.append(target)

        meta_list.append({
            "file": file_name,
            "row_target": int(i + 1)
        })

    if len(X_list) == 0:
        return None, None, None

    return np.asarray(X_list), np.asarray(y_list), meta_list


def load_dataset_from_folder(data_folder, history_len=20, stride=10, max_files=None):
    files = sorted(glob.glob(os.path.join(data_folder, "*.txt")))

    if max_files is not None:
        files = files[:max_files]

    all_X = []
    all_y = []
    all_meta = []

    for file_path in files:
        file_name = os.path.basename(file_path)

        df = read_episode(file_path)
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        X_ep, y_ep, meta_ep = build_episode_samples(
            df,
            file_name=file_name,
            history_len=history_len,
            stride=stride
        )

        if X_ep is None:
            continue

        all_X.append(X_ep)
        all_y.append(y_ep)
        all_meta.extend(meta_ep)

        print(f"[OK] {file_name}: samples={len(X_ep)}")

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    meta = pd.DataFrame(all_meta)

    return X, y, meta


def chronological_test_split(X, y, meta, train_ratio=0.70, val_ratio=0.15):
    n = len(X)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    start_test = n_train + n_val

    return (
        X[start_test:],
        y[start_test:],
        meta.iloc[start_test:].reset_index(drop=True)
    )


def mc_dropout_predict(model, loader, device, n_mc=50):
    """
    Inferencia Monte Carlo Dropout.

    Importante:
    model.train() activa dropout.
    torch.no_grad() evita entrenamiento.
    """

    model.train()

    all_means = []
    all_stds = []
    all_true = []
    all_indices = []

    with torch.no_grad():
        for X_batch, y_batch, idx_batch in loader:
            X_batch = X_batch.to(device)

            preds_mc = []

            for _ in range(n_mc):
                pred = model(X_batch)
                preds_mc.append(pred.cpu().numpy())

            preds_mc = np.stack(preds_mc, axis=0)

            mean = preds_mc.mean(axis=0)
            std = preds_mc.std(axis=0)

            all_means.append(mean)
            all_stds.append(std)
            all_true.append(y_batch.numpy())
            all_indices.append(idx_batch.numpy())

    y_mean = np.vstack(all_means)
    y_std = np.vstack(all_stds)
    y_true = np.vstack(all_true)
    indices = np.concatenate(all_indices)

    return y_true, y_mean, y_std, indices


def compute_metrics(y_true, y_mean, y_std):
    rows = []

    for i, state in enumerate(STATE_COLS):
        error = np.abs(y_true[:, i] - y_mean[:, i])

        mae = mean_absolute_error(y_true[:, i], y_mean[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_mean[:, i]))

        rows.append({
            "state": state,
            "mae": float(mae),
            "rmse": float(rmse),
            "error_mean": float(np.mean(error)),
            "error_median": float(np.median(error)),
            "error_p90": float(np.percentile(error, 90)),
            "error_p95": float(np.percentile(error, 95)),
            "error_max": float(np.max(error)),
            "std_mean": float(np.mean(y_std[:, i])),
            "std_median": float(np.median(y_std[:, i])),
            "std_p90": float(np.percentile(y_std[:, i], 90)),
            "std_p95": float(np.percentile(y_std[:, i], 95)),
            "std_max": float(np.max(y_std[:, i])),
        })

    return pd.DataFrame(rows)


def build_results_df(meta_test, y_true, y_mean, y_std):
    results = meta_test.copy()

    for i, state in enumerate(STATE_COLS):
        results[f"{state}_true"] = y_true[:, i]
        results[f"{state}_pred_mean"] = y_mean[:, i]
        results[f"{state}_error"] = y_true[:, i] - y_mean[:, i]
        results[f"{state}_abs_error"] = np.abs(y_true[:, i] - y_mean[:, i])
        results[f"{state}_std_mc"] = y_std[:, i]

    return results


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    with open(os.path.join(args.model_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    history_len = metadata["history_len"]
    stride = metadata["stride"]
    input_dim = metadata["input_dim"]
    output_dim = metadata["output_dim"]
    hidden_dim = metadata["hidden_dim"]
    dropout_p = metadata["dropout_p"]
    train_ratio = metadata["train_ratio"]
    val_ratio = metadata["val_ratio"]

    X, y, meta = load_dataset_from_folder(
        args.data_folder,
        history_len=history_len,
        stride=stride,
        max_files=args.max_files
    )

    X_test, y_test, meta_test = chronological_test_split(
        X,
        y,
        meta,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    x_scaler = joblib.load(os.path.join(args.model_dir, "x_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(args.model_dir, "y_scaler.pkl"))

    X_test_s = x_scaler.transform(X_test)
    y_test_s = y_scaler.transform(y_test)

    test_ds = DynamicsDataset(X_test_s, y_test_s)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = MCDropoutDynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(args.model_dir, "model.pt"), map_location=device)
    )

    y_true_s, y_mean_s, y_std_s, indices = mc_dropout_predict(
        model=model,
        loader=test_loader,
        device=device,
        n_mc=args.n_mc
    )

    y_true = y_scaler.inverse_transform(y_true_s)
    y_mean = y_scaler.inverse_transform(y_mean_s)

    # Para std, se multiplica por la escala de cada variable.
    y_std = y_std_s * y_scaler.scale_

    metrics_df = compute_metrics(
        y_true=y_true,
        y_mean=y_mean,
        y_std=y_std
    )

    results_df = build_results_df(
        meta_test=meta_test,
        y_true=y_true,
        y_mean=y_mean,
        y_std=y_std
    )

    metrics_path = os.path.join(args.output_dir, "mc_dropout_metrics.csv")
    results_path = os.path.join(args.output_dir, "mc_dropout_predictions.csv")

    metrics_df.to_csv(metrics_path, index=False)
    results_df.to_csv(results_path, index=False)

    print("\nMétricas MC Dropout:")
    print(metrics_df)

    print("\nArchivos guardados:")
    print(metrics_path)
    print(results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mc_dropout_inference_results")

    parser.add_argument("--n_mc", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    main(args)