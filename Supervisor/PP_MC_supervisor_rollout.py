import os
import json
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from argparse import Namespace

from controllers.base_controller import BaseController
from controllers.pp_controller import PurePursuitController


# ============================================================
# COLUMNAS DEL MODELO DINÁMICO
# ============================================================

STATE_COLS = [
    "Vy", "AVz", "Yaw", "Beta",
    "Ax", "Ay", "AVx", "Roll"
]

CONTROL_COLS = [
    "Steer", "Vx"
]

FEATURE_COLS = STATE_COLS + CONTROL_COLS


# ============================================================
# MODELO MC DROPOUT
# Debe coincidir con el usado en entrenamiento/inferencia.
# ============================================================

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


# ============================================================
# CONTROLLER FACTORY
# ============================================================

def create_controller(conf) -> BaseController:
    return PurePursuitController(
        conf=conf,
        wheelbase=0.17145 + 0.15875,
    )


# ============================================================
# PREPROCESAMIENTO DE BETA
# ============================================================

def clean_beta_from_vx_vy(df, vx_min=1.0, speed_min=1.0):
    """
    Recalcula Beta para evitar artefactos de ±180° cuando Vx ≈ 0
    o cuando el vehículo está prácticamente parado.

    Esta versión es más robusta que usar eps=1e-6.
    """
    vx = df["Vx"].values.astype(float)
    vy = df["Vy"].values.astype(float)

    denom = np.maximum(np.abs(vx), vx_min)
    beta_rad = np.arctan2(vy, denom)
    beta_deg = np.rad2deg(beta_rad)

    speed = np.sqrt(vx**2 + vy**2)
    beta_deg[speed < speed_min] = 0.0

    df["Beta"] = beta_deg
    return df


def read_history_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"El archivo de historia no contiene las columnas requeridas: {missing}"
        )

    if "Vx" in df.columns and "Vy" in df.columns:
        df = clean_beta_from_vx_vy(df)

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


def build_history_window_from_file(history_file, history_row, history_len):
    """
    Construye la ventana histórica usada como estado inicial del rollout dinámico.

    history_window shape:
        (history_len + 1, len(FEATURE_COLS))

    Debe tener exactamente la misma estructura usada en el entrenamiento:

        [Vy, AVz, Yaw, Beta, Ax, Ay, AVx, Roll, Steer, Vx]
    """

    df = read_history_file(history_file)

    if history_row is None:
        history_row = len(df) - 1

    start = history_row - history_len
    end = history_row + 1

    if start < 0:
        raise ValueError(
            f"history_row={history_row} es demasiado pequeño para "
            f"history_len={history_len}. Debe ser >= {history_len}."
        )

    if end > len(df):
        raise ValueError(
            f"history_row={history_row} excede el tamaño del archivo "
            f"con {len(df)} filas."
        )

    window = df.iloc[start:end][FEATURE_COLS].values.astype(np.float64)

    if window.shape[0] != history_len + 1:
        raise RuntimeError(
            f"Ventana inválida: {window.shape}. "
            f"Esperado: ({history_len + 1}, {len(FEATURE_COLS)})"
        )

    return window


# ============================================================
# CARGA DEL MODELO Y SCALERS
# ============================================================

def load_mc_dropout_model(model_dir, device):
    metadata_path = os.path.join(model_dir, "metadata.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    input_dim = metadata["input_dim"]
    output_dim = metadata["output_dim"]
    hidden_dim = metadata["hidden_dim"]
    dropout_p = metadata["dropout_p"]

    model = MCDropoutDynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
    ).to(device)

    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    x_scaler = joblib.load(os.path.join(model_dir, "x_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(model_dir, "y_scaler.pkl"))

    return model, x_scaler, y_scaler, metadata


# ============================================================
# ROLLOUT DINÁMICO CON MC DROPOUT
# ============================================================

def rollout_dynamics_mc_dropout(
    model,
    history_window,
    U_future,
    x_scaler,
    y_scaler,
    device,
    n_mc=50,
):
    """
    Realiza rollout del modelo dinámico usando MC Dropout.

    Parameters
    ----------
    model:
        Red entrenada.

    history_window:
        np.ndarray shape (history_len + 1, n_features)
        Ventana histórica inicial.

    U_future:
        np.ndarray shape (H, 2)
        Vector de controles futuros generado por PP.
        Cada fila debe ser:
            [Steer_k, Vx_k]

    Returns
    -------
    means:
        np.ndarray shape (H, n_states)
        Media predicha por estado.

    stds:
        np.ndarray shape (H, n_states)
        Desviación estándar MC Dropout por estado.
    """

    model.train()  # importante: mantiene Dropout activo

    window = history_window.copy()

    means = []
    stds = []

    H = U_future.shape[0]

    with torch.no_grad():
        for k in range(H):
            # Entrada al modelo: ventana histórica aplanada
            x_input = window.reshape(1, -1)
            x_input_s = x_scaler.transform(x_input)

            x_tensor = torch.tensor(
                x_input_s,
                dtype=torch.float32,
                device=device
            )

            preds_s = []

            for _ in range(n_mc):
                y_pred_s = model(x_tensor).cpu().numpy()[0]
                preds_s.append(y_pred_s)

            preds_s = np.asarray(preds_s)

            mean_s = preds_s.mean(axis=0)
            std_s = preds_s.std(axis=0)

            # Volver a unidades originales
            mean = y_scaler.inverse_transform(
                mean_s.reshape(1, -1)
            )[0]

            # std se desescala multiplicando por la escala del y_scaler
            std = std_s * y_scaler.scale_

            means.append(mean)
            stds.append(std)

            # Control futuro generado por PP
            steer_k = U_future[k, 0]
            vx_k = U_future[k, 1]

            # Nueva fila de features para actualizar la ventana
            # Debe respetar FEATURE_COLS = STATE_COLS + CONTROL_COLS
            new_feature = np.concatenate([
                mean,
                np.array([steer_k, vx_k], dtype=np.float64)
            ])

            # Actualizar ventana: quitar fila más antigua y agregar nueva predicción
            window = np.vstack([
                window[1:],
                new_feature
            ])

    return np.asarray(means), np.asarray(stds)


# ============================================================
# RESULTADOS EN DATAFRAME
# ============================================================

def build_rollout_results_df(U_future, means, stds):
    rows = []

    for k in range(U_future.shape[0]):
        row = {
            "k": k,
            "Steer_cmd": U_future[k, 0],
            "Vx_cmd": U_future[k, 1],
        }

        for i, state in enumerate(STATE_COLS):
            row[f"{state}_mean"] = means[k, i]
            row[f"{state}_std"] = stds[k, i]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print("Device:", device)

    # ----------------------------------------
    # Cargar config y controlador PP
    # ----------------------------------------
    with open(args.config, "r") as f:
        conf = Namespace(**yaml.safe_load(f))

    controller = create_controller(conf)

    if not hasattr(controller, "get_future_pp_controls"):
        raise AttributeError(
            "Tu PurePursuitController no tiene el método "
            "'get_future_pp_controls'. Debes agregarlo primero en "
            "controllers/pp_controller.py."
        )

    # ----------------------------------------
    # Cargar modelo MC Dropout
    # ----------------------------------------
    model, x_scaler, y_scaler, metadata = load_mc_dropout_model(
        args.model_dir,
        device
    )

    history_len = metadata["history_len"]

    # ----------------------------------------
    # Construir ventana histórica inicial
    # ----------------------------------------
    history_window = build_history_window_from_file(
        history_file=args.history_file,
        history_row=args.history_row,
        history_len=history_len,
    )

    print("History window:", history_window.shape)

    # ----------------------------------------
    # Pose inicial para PP
    # ----------------------------------------
    pose_x = args.pose_x
    pose_y = args.pose_y
    pose_theta = args.pose_theta

    if pose_x is None:
        pose_x = conf.sx

    if pose_y is None:
        pose_y = conf.sy

    if pose_theta is None:
        pose_theta = conf.stheta

    print(
        f"Initial PP pose: x={pose_x:.3f}, "
        f"y={pose_y:.3f}, theta={pose_theta:.3f}"
    )

    # ----------------------------------------
    # Rollout nominal del PP
    # ----------------------------------------
    U_future, pp_info = controller.get_future_pp_controls(
        pose_x,
        pose_y,
        pose_theta,
        tlad=args.tlad,
        vgain=args.vgain,
        horizon=args.horizon,
        waypoint_step=args.waypoint_step,
        include_current=True,
    )

    print("U_future:", U_future.shape)
    print("First controls:")
    print(U_future[:min(5, len(U_future))])

    # ----------------------------------------
    # Rollout del modelo dinámico con MC Dropout
    # ----------------------------------------
    means, stds = rollout_dynamics_mc_dropout(
        model=model,
        history_window=history_window,
        U_future=U_future,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        n_mc=args.n_mc,
    )

    print("Predicted means:", means.shape)
    print("Predicted stds:", stds.shape)

    # ----------------------------------------
    # Guardar resultados
    # ----------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    results_df = build_rollout_results_df(
        U_future=U_future,
        means=means,
        stds=stds,
    )

    output_path = os.path.join(
        args.output_dir,
        "pp_mc_supervisor_rollout.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nRollout guardado en:")
    print(output_path)

    print("\nPrimeras filas:")
    print(results_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config_map.yaml")
    parser.add_argument("--model_dir", type=str, required=True)

    parser.add_argument("--history_file", type=str, required=True)
    parser.add_argument("--history_row", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="supervisor_rollout_results")

    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--waypoint_step", type=int, default=5)

    parser.add_argument("--tlad", type=float, default=0.82461887897713965)
    parser.add_argument("--vgain", type=float, default=1.375)

    parser.add_argument("--n_mc", type=int, default=50)

    parser.add_argument("--pose_x", type=float, default=None)
    parser.add_argument("--pose_y", type=float, default=None)
    parser.add_argument("--pose_theta", type=float, default=None)

    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)