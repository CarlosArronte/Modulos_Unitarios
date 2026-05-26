import os
import glob
import json
import argparse
import joblib
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


STATE_COLS = [
    "Vy", "AVz", "Yaw", "Beta",
    "Ax", "Ay", "AVx", "Roll"
]

CONTROL_COLS = [
    "Steer", "Vx"
]

FEATURE_COLS = STATE_COLS + CONTROL_COLS


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_episode(file_path):
    df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]
        # Recalcular Beta para evitar artefactos de ±180° cuando Vx ≈ 0 o Vx < 0 y Vy ≈ 0.
    if "Vx" in df.columns and "Vy" in df.columns:
        vx = df["Vx"].values.astype(float)
        vy = df["Vy"].values.astype(float)

        eps = 1e-6
        beta_rad = np.arctan2(vy, np.maximum(np.abs(vx), eps))

        # Mantener el mismo nombre de columna para no modificar el resto del pipeline.
        df["Beta"] = np.rad2deg(beta_rad)

    return df


def validate_columns(df, file_path):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"El archivo {file_path} no contiene columnas requeridas: {missing}"
        )


def build_episode_samples(df, history_len=20, stride=10):
    """
    Construye muestras dentro de un único episodio.

    X_i = [x(t-H), u(t-H), ..., x(t), u(t)]
    y_i = x(t+1)
    """

    features = df[FEATURE_COLS].values.astype(np.float32)
    states = df[STATE_COLS].values.astype(np.float32)

    X_list = []
    y_list = []

    for i in range(history_len, len(df) - 1, stride):
        hist = features[i - history_len:i + 1]
        target = states[i + 1]

        X_list.append(hist.reshape(-1))
        y_list.append(target)

    if len(X_list) == 0:
        return None, None

    return np.asarray(X_list), np.asarray(y_list)


def load_dataset_from_folder(data_folder, history_len=20, stride=10, max_files=None):
    files = sorted(glob.glob(os.path.join(data_folder, "*.txt")))

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .txt en {data_folder}")

    all_X = []
    all_y = []
    episode_info = []

    print(f"Archivos encontrados: {len(files)}")

    for file_path in files:
        file_name = os.path.basename(file_path)

        try:
            df = read_episode(file_path)
            validate_columns(df, file_path)

            df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

            X_ep, y_ep = build_episode_samples(
                df,
                history_len=history_len,
                stride=stride
            )

            if X_ep is None:
                print(f"[SKIP] {file_name}: episodio muy corto")
                continue

            all_X.append(X_ep)
            all_y.append(y_ep)

            episode_info.append({
                "file": file_name,
                "rows": int(len(df)),
                "samples": int(len(X_ep))
            })

            print(f"[OK] {file_name}: rows={len(df)} samples={len(X_ep)}")

        except Exception as e:
            print(f"[ERROR] {file_name}: {e}")

    if len(all_X) == 0:
        raise RuntimeError("No se pudo construir ninguna muestra válida.")

    X = np.vstack(all_X)
    y = np.vstack(all_y)

    return X, y, episode_info


def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    n = len(X)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


class DynamicsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MCDropoutDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=8,
        hidden_dim=256,
        dropout_p=0.10
    ):
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


def evaluate_deterministic(model, loader, device, y_scaler=None):
    """
    Evaluación normal con dropout desactivado.
    """
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)

    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred)
        y_true = y_scaler.inverse_transform(y_true)

    metrics = compute_metrics(y_true, y_pred)

    return metrics, y_true, y_pred


def compute_metrics(y_true, y_pred):
    rows = []

    for i, state in enumerate(STATE_COLS):
        error = np.abs(y_true[:, i] - y_pred[:, i])

        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))

        rows.append({
            "state": state,
            "mae": float(mae),
            "rmse": float(rmse),
            "error_mean": float(np.mean(error)),
            "error_median": float(np.median(error)),
            "error_p90": float(np.percentile(error, 90)),
            "error_p95": float(np.percentile(error, 95)),
            "error_max": float(np.max(error)),
        })

    return pd.DataFrame(rows)


def train(args):
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    X_train, y_train, train_info = load_dataset_from_folder(
    data_folder=args.train_folder,
    history_len=args.history_len,
    stride=args.stride,
    max_files=None
)

    X_val, y_val, val_info = load_dataset_from_folder(
        data_folder=args.val_folder,
        history_len=args.history_len,
        stride=args.stride,
        max_files=None
    )

    X_test, y_test, test_info = load_dataset_from_folder(
        data_folder=args.test_folder,
        history_len=args.history_len,
        stride=args.stride,
        max_files=None
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    joblib.dump(x_scaler, os.path.join(args.save_dir, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(args.save_dir, "y_scaler.pkl"))

    train_ds = DynamicsDataset(X_train_s, y_train_s)
    val_ds = DynamicsDataset(X_val_s, y_val_s)
    test_ds = DynamicsDataset(X_test_s, y_test_s)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    input_dim = X_train_s.shape[1]
    output_dim = y_train_s.shape[1]

    model = MCDropoutDynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_epoch = -1
    patience_counter = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss:.6e} | "
                f"val_loss={val_loss:.6e}"
            )

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "model.pt")
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping en epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "model.pt"), map_location=device))

    test_metrics, y_true, y_pred = evaluate_deterministic(
        model=model,
        loader=test_loader,
        device=device,
        y_scaler=y_scaler
    )

    test_metrics.to_csv(
        os.path.join(args.save_dir, "test_metrics_deterministic.csv"),
        index=False
    )

    pd.DataFrame(history).to_csv(
        os.path.join(args.save_dir, "train_history.csv"),
        index=False
    )

    metadata = {
        "state_cols": STATE_COLS,
        "control_cols": CONTROL_COLS,
        "feature_cols": FEATURE_COLS,
        "history_len": args.history_len,
        "stride": args.stride,
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden_dim": args.hidden_dim,
        "dropout_p": args.dropout_p,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "train_folder": args.train_folder,
        "val_folder": args.val_folder,
        "test_folder": args.test_folder,
        "train_episodes": train_info,
        "val_episodes": val_info,
        "test_episodes": test_info,
        "num_train_samples": int(len(X_train)),
        "num_val_samples": int(len(X_val)),
        "num_test_samples": int(len(X_test))
    }

    with open(os.path.join(args.save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nMétricas determinísticas de test:")
    print(test_metrics)

    print("\nModelo guardado en:")
    print(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_folder", type=str, required=True)
    parser.add_argument("--val_folder", type=str, required=True)
    parser.add_argument("--test_folder", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="saved_mc_dropout_dyn")

    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_p", type=float, default=0.10)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-6)
    parser.add_argument("--print_every", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    train(args)