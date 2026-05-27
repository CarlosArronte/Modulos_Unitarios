import os
import glob
import json
import joblib
import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error


# STATE_COLS = [
#     "Vy", "AVz", "Yaw", "Beta",
#     "Ax", "Ay", "AVx", "Roll"
# ]
STATE_COLS = [
    "Vy",
    "AVz",
    "Beta",
    "Ax",
    "Ay"
] #No roll dynamics in the gym and no good Yaw behavior

CONTROL_COLS = [
    "Steer", "Vx"
]

FEATURE_COLS = STATE_COLS + CONTROL_COLS


def read_episode(file_path):
    """
    Lee un episodio .txt generado por CarSim/MATLAB.

    Intenta detectar automáticamente separador:
    - coma
    - tabulación
    - espacios múltiples
    - punto y coma
    """

    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path, delim_whitespace=True)

    # Limpieza básica de nombres
    df.columns = [str(c).strip() for c in df.columns]

    return df


def validate_columns(df, file_path):
    missing = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        raise ValueError(
            f"El archivo {file_path} no contiene las columnas requeridas: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )


def build_episode_samples(df, history_len=20, stride=1):
    """
    Construye muestras dentro de un único episodio.

    Entrada:
        X_i = [x(t-history), u(t-history), ..., x(t), u(t)]

    Salida:
        y_i = x(t+1)

    Importante:
        Como esta función se aplica por archivo, ninguna ventana cruza entre episodios.
    """

    features = df[FEATURE_COLS].values.astype(np.float64)
    states = df[STATE_COLS].values.astype(np.float64)

    X_list = []
    y_list = []

    for i in range(history_len, len(df) - 1, stride):
        hist = features[i - history_len:i + 1]
        x_next = states[i + 1]

        X_list.append(hist.flatten())
        y_list.append(x_next)

    if len(X_list) == 0:
        return None, None

    X = np.asarray(X_list)
    y = np.asarray(y_list)

    return X, y


def load_dataset_from_folder(data_folder, history_len=20, stride=5, max_files=None):
    """
    Lee todos los .txt de una carpeta y construye el dataset global.
    """

    pattern = os.path.join(data_folder, "*.txt")
    files = sorted(glob.glob(pattern))

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .txt en: {data_folder}")

    all_X = []
    all_y = []
    episode_info = []

    print(f"Archivos encontrados: {len(files)}")

    for file_path in files:
        file_name = os.path.basename(file_path)

        try:
            df = read_episode(file_path)
            validate_columns(df, file_path)

            # Eliminar filas con NaN en columnas usadas
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
                "rows": len(df),
                "samples": len(X_ep)
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
    """
    Split cronológico sobre las muestras ya concatenadas.

    Para una versión más estricta, se puede hacer split por archivo/maniobra.
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


def subsample_train_set(X_train, y_train, max_train_samples, random_state=42):
    """
    El GP exacto escala mal con muchos datos.
    Esta función permite entrenar con un subconjunto.
    """

    if max_train_samples is None:
        return X_train, y_train

    if len(X_train) <= max_train_samples:
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
    idx = np.sort(idx)

    return X_train[idx], y_train[idx]


def create_gp(input_dim):
    """
    GP exacto para una variable de estado.

    RBF con ARD:
        un length_scale por dimensión de entrada.

    WhiteKernel:
        modela ruido / variabilidad no explicada.
    """

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(
            length_scale=np.ones(input_dim),
            length_scale_bounds=(1e-3, 1e3)
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-8, 1e1)
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=42
    )

    return gp


def evaluate_output(y_true, y_mean, y_std, state_name):
    error = np.abs(y_true - y_mean)

    mae = mean_absolute_error(y_true, y_mean)
    rmse = np.sqrt(mean_squared_error(y_true, y_mean))

    metrics = {
        "state": state_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "error_mean": float(np.mean(error)),
        "error_median": float(np.median(error)),
        "error_p90": float(np.percentile(error, 90)),
        "error_p95": float(np.percentile(error, 95)),
        "error_max": float(np.max(error)),
        "std_mean": float(np.mean(y_std)),
        "std_median": float(np.median(y_std)),
        "std_p90": float(np.percentile(y_std, 90)),
        "std_p95": float(np.percentile(y_std, 95)),
        "std_max": float(np.max(y_std)),
    }

    return metrics


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    print("\nCargando dataset desde carpeta:")
    print(args.data_folder)

    X, y, episode_info = load_dataset_from_folder(
        data_folder=args.data_folder,
        history_len=args.history_len,
        stride=args.stride,
        max_files=args.max_files
    )

    print("\nDataset construido:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    input_dim = X.shape[1]

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X,
        y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    print("\nSplit:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    X_train_gp, y_train_gp = subsample_train_set(
        X_train,
        y_train,
        max_train_samples=args.max_train_samples,
        random_state=args.random_state
    )

    print("\nTrain usado por el GP:")
    print("X_train_gp:", X_train_gp.shape)
    print("y_train_gp:", y_train_gp.shape)

    x_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train_gp)
    X_test_scaled = x_scaler.transform(X_test)

    joblib.dump(x_scaler, os.path.join(args.save_dir, "x_scaler.pkl"))

    all_metrics = []

    for idx, state_name in enumerate(STATE_COLS):
        print("\n" + "=" * 70)
        print(f"Entrenando GP para {state_name}(t+1)")
        print("=" * 70)

        gp = create_gp(input_dim=input_dim)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_train_scaled, y_train_gp[:, idx])

        y_mean, y_std = gp.predict(
            X_test_scaled,
            return_std=True
        )

        metrics = evaluate_output(
            y_true=y_test[:, idx],
            y_mean=y_mean,
            y_std=y_std,
            state_name=state_name
        )

        all_metrics.append(metrics)

        model_path = os.path.join(args.save_dir, f"gp_{state_name}.pkl")
        joblib.dump(gp, model_path)

        print(f"Kernel aprendido: {gp.kernel_}")
        print(f"MAE:       {metrics['mae']:.6f}")
        print(f"RMSE:      {metrics['rmse']:.6f}")
        print(f"Error p95: {metrics['error_p95']:.6f}")
        print(f"STD p95:   {metrics['std_p95']:.6f}")
        print(f"Guardado:  {model_path}")

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = os.path.join(args.save_dir, "gp_test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    metadata = {
        "state_cols": STATE_COLS,
        "control_cols": CONTROL_COLS,
        "feature_cols": FEATURE_COLS,
        "history_len": args.history_len,
        "stride": args.stride,
        "input_dim": int(input_dim),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "max_train_samples": args.max_train_samples,
        "num_total_samples": int(len(X)),
        "num_train_samples": int(len(X_train)),
        "num_train_samples_gp": int(len(X_train_gp)),
        "num_val_samples": int(len(X_val)),
        "num_test_samples": int(len(X_test)),
        "episodes": episode_info
    }

    metadata_path = os.path.join(args.save_dir, "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 70)
    print("Resumen final")
    print("=" * 70)
    print(metrics_df)

    print("\nArchivos guardados en:")
    print(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Carpeta con archivos .txt del dataset"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_gp_pure",
        help="Carpeta de salida"
    )

    parser.add_argument(
        "--history_len",
        type=int,
        default=20,
        help="Longitud de la ventana histórica"
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Salto temporal al construir muestras"
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=3000,
        help="Máximo número de muestras para entrenar cada GP"
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Número máximo de archivos a usar. Útil para pruebas rápidas"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    main(args)