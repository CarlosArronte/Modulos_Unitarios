import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error


STATE_COLS = [
    "Vy", "AVz", "Yaw", "Beta",
    "Ax", "Ay", "AVx", "Roll"
]

CONTROL_COLS = [
    "Steer", "Vx"
]


def build_gp_dataset(df, history_len=20, stride=1):
    """
    Construye el dataset para entrenar un GP que predice x(t+1)
    usando una ventana histórica de x y u.

    Entrada:
        X_i = [x(t-history), u(t-history), ..., x(t), u(t)]

    Salida:
        y_i = x(t+1)
    """

    required_cols = STATE_COLS + CONTROL_COLS

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    features = df[STATE_COLS + CONTROL_COLS].values
    states = df[STATE_COLS].values

    X_list = []
    y_list = []

    for i in range(history_len, len(df) - 1, stride):
        hist = features[i - history_len:i + 1]
        X_list.append(hist.flatten())

        x_next = states[i + 1]
        y_list.append(x_next)

    X = np.asarray(X_list)
    y = np.asarray(y_list)

    return X, y


def chronological_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split temporal: no mezcla aleatoriamente los datos.
    Esto es más correcto para series temporales.
    """

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


def create_gp():
    """
    Kernel inicial para el GP.

    ConstantKernel:
        escala global de la función.

    RBF:
        suavidad de la función aprendida.

    WhiteKernel:
        ruido de observación / error no explicado.
    """

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=42
    )

    return gp


def evaluate_prediction(y_true, y_mean, y_std, state_name):
    """
    Evalúa error e incertidumbre para una variable de estado.
    """

    abs_error = np.abs(y_true - y_mean)

    mae = mean_absolute_error(y_true, y_mean)
    rmse = np.sqrt(mean_squared_error(y_true, y_mean))

    metrics = {
        "state": state_name,
        "mae": mae,
        "rmse": rmse,
        "error_mean": float(np.mean(abs_error)),
        "error_median": float(np.median(abs_error)),
        "error_p90": float(np.percentile(abs_error, 90)),
        "error_p95": float(np.percentile(abs_error, 95)),
        "error_max": float(np.max(abs_error)),
        "std_mean": float(np.mean(y_std)),
        "std_median": float(np.median(y_std)),
        "std_p90": float(np.percentile(y_std, 90)),
        "std_p95": float(np.percentile(y_std, 95)),
        "std_max": float(np.max(y_std)),
    }

    return metrics


def main():
    data_file = "dataset.txt"
    save_dir = "saved_gp_onestep"

    history_len = 20
    stride = 5

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(data_file)

    X, y = build_gp_dataset(
        df,
        history_len=history_len,
        stride=stride
    )

    print("Dataset GP")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X,
        y,
        train_ratio=0.7,
        val_ratio=0.15
    )

    x_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    joblib.dump(x_scaler, os.path.join(save_dir, "x_scaler.pkl"))

    gps = {}
    all_metrics = []

    for idx, state_name in enumerate(STATE_COLS):
        print("\n" + "=" * 60)
        print(f"Entrenando GP para {state_name}(t+1)")
        print("=" * 60)

        gp = create_gp()

        gp.fit(X_train_scaled, y_train[:, idx])

        y_mean, y_std = gp.predict(
            X_test_scaled,
            return_std=True
        )

        metrics = evaluate_prediction(
            y_true=y_test[:, idx],
            y_mean=y_mean,
            y_std=y_std,
            state_name=state_name
        )

        all_metrics.append(metrics)
        gps[state_name] = gp

        model_path = os.path.join(save_dir, f"gp_{state_name}.pkl")
        joblib.dump(gp, model_path)

        print(f"Modelo guardado en: {model_path}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"Error p95: {metrics['error_p95']:.6f}")
        print(f"STD p95: {metrics['std_p95']:.6f}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(save_dir, "gp_test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 60)
    print("Resumen final")
    print("=" * 60)
    print(metrics_df)
    print(f"\nMétricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()