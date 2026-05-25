import os
import glob
import json
import argparse
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


STATE_COLS = [
    "Vy", "AVz", "Yaw", "Beta",
    "Ax", "Ay", "AVx", "Roll"
]

CONTROL_COLS = [
    "Steer", "Vx"
]

FEATURE_COLS = STATE_COLS + CONTROL_COLS


def read_episode(file_path):
    """
    Lee un archivo .txt de CarSim/MATLAB.
    Intenta detectar automáticamente el separador.
    """

    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path, delim_whitespace=True)

    df.columns = [str(c).strip() for c in df.columns]

    return df


def validate_columns(df, file_path):
    missing = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        raise ValueError(
            f"El archivo {file_path} no contiene columnas requeridas: {missing}"
        )


def build_episode_samples(df, history_len=20, stride=1):
    """
    Construye muestras one-step dentro de un episodio.

    X_i = [x(t-H), u(t-H), ..., x(t), u(t)]
    y_i = x(t+1)
    """

    features = df[FEATURE_COLS].values.astype(np.float64)
    states = df[STATE_COLS].values.astype(np.float64)

    X_list = []
    y_list = []
    index_list = []

    for i in range(history_len, len(df) - 1, stride):
        hist = features[i - history_len:i + 1]
        x_next = states[i + 1]

        X_list.append(hist.flatten())
        y_list.append(x_next)
        index_list.append(i + 1)

    if len(X_list) == 0:
        return None, None, None

    X = np.asarray(X_list)
    y = np.asarray(y_list)
    idx = np.asarray(index_list)

    return X, y, idx


def load_dataset_from_folder(data_folder, history_len=20, stride=10, max_files=None):
    files = sorted(glob.glob(os.path.join(data_folder, "*.txt")))

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .txt en {data_folder}")

    all_X = []
    all_y = []
    all_meta = []

    for file_path in files:
        file_name = os.path.basename(file_path)

        try:
            df = read_episode(file_path)
            validate_columns(df, file_path)

            df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

            X_ep, y_ep, idx_ep = build_episode_samples(
                df,
                history_len=history_len,
                stride=stride
            )

            if X_ep is None:
                print(f"[SKIP] {file_name}: episodio muy corto")
                continue

            all_X.append(X_ep)
            all_y.append(y_ep)

            for k in range(len(X_ep)):
                all_meta.append({
                    "file": file_name,
                    "row_target": int(idx_ep[k])
                })

            print(f"[OK] {file_name}: samples={len(X_ep)}")

        except Exception as e:
            print(f"[ERROR] {file_name}: {e}")

    X = np.vstack(all_X)
    y = np.vstack(all_y)
    meta = pd.DataFrame(all_meta)

    return X, y, meta


def chronological_split(X, y, meta, train_ratio=0.70, val_ratio=0.15):
    """
    Replica el split usado en entrenamiento:
    train / val / test cronológico sobre las muestras concatenadas.
    """

    n = len(X)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    meta_test = meta.iloc[n_train + n_val:].reset_index(drop=True)

    return X_test, y_test, meta_test


def load_gp_models(model_dir):
    models = {}

    for state in STATE_COLS:
        model_path = os.path.join(model_dir, f"gp_{state}.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

        models[state] = joblib.load(model_path)

    return models


def predict_all_states(models, X_scaled):
    """
    Ejecuta inferencia para los 8 GP.

    Devuelve:
        y_mean: predicción media, shape [N, 8]
        y_std: incertidumbre, shape [N, 8]
    """

    means = []
    stds = []

    for state in STATE_COLS:
        gp = models[state]

        mean, std = gp.predict(
            X_scaled,
            return_std=True
        )

        means.append(mean)
        stds.append(std)

    y_mean = np.column_stack(means)
    y_std = np.column_stack(stds)

    return y_mean, y_std


def compute_metrics(y_true, y_pred, y_std):
    rows = []

    for i, state in enumerate(STATE_COLS):
        error = np.abs(y_true[:, i] - y_pred[:, i])

        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))

        rows.append({
            "state": state,
            "mae": mae,
            "rmse": rmse,
            "error_mean": np.mean(error),
            "error_median": np.median(error),
            "error_p90": np.percentile(error, 90),
            "error_p95": np.percentile(error, 95),
            "error_max": np.max(error),
            "std_mean": np.mean(y_std[:, i]),
            "std_median": np.median(y_std[:, i]),
            "std_p90": np.percentile(y_std[:, i], 90),
            "std_p95": np.percentile(y_std[:, i], 95),
            "std_max": np.max(y_std[:, i]),
        })

    return pd.DataFrame(rows)


def build_predictions_dataframe(meta_test, y_true, y_pred, y_std):
    """
    Construye un CSV con valores reales, predichos, error e incertidumbre.
    """

    results = meta_test.copy()

    for i, state in enumerate(STATE_COLS):
        results[f"{state}_true"] = y_true[:, i]
        results[f"{state}_pred"] = y_pred[:, i]
        results[f"{state}_error"] = y_true[:, i] - y_pred[:, i]
        results[f"{state}_abs_error"] = np.abs(y_true[:, i] - y_pred[:, i])
        results[f"{state}_std"] = y_std[:, i]

    return results


def main(args):
    metadata_path = os.path.join(args.model_dir, "metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        history_len = metadata.get("history_len", args.history_len)
        stride = metadata.get("stride", args.stride)
        train_ratio = metadata.get("train_ratio", args.train_ratio)
        val_ratio = metadata.get("val_ratio", args.val_ratio)

        print("Usando parámetros desde metadata.json:")
        print(f"history_len = {history_len}")
        print(f"stride = {stride}")
        print(f"train_ratio = {train_ratio}")
        print(f"val_ratio = {val_ratio}")

    else:
        history_len = args.history_len
        stride = args.stride
        train_ratio = args.train_ratio
        val_ratio = args.val_ratio

        print("metadata.json no encontrado. Usando argumentos manuales.")

    print("\nCargando dataset...")
    X, y, meta = load_dataset_from_folder(
        data_folder=args.data_folder,
        history_len=history_len,
        stride=stride,
        max_files=args.max_files
    )

    X_test, y_test, meta_test = chronological_split(
        X,
        y,
        meta,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    print("\nTest set:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    scaler_path = os.path.join(args.model_dir, "x_scaler.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No se encontró el scaler: {scaler_path}")

    x_scaler = joblib.load(scaler_path)

    X_test_scaled = x_scaler.transform(X_test)

    print("\nCargando modelos GP...")
    models = load_gp_models(args.model_dir)

    print("Ejecutando inferencia...")
    y_pred, y_std = predict_all_states(
        models=models,
        X_scaled=X_test_scaled
    )

    metrics_df = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_std=y_std
    )

    results_df = build_predictions_dataframe(
        meta_test=meta_test,
        y_true=y_test,
        y_pred=y_pred,
        y_std=y_std
    )

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_path = os.path.join(args.output_dir, "gp_inference_metrics.csv")
    results_path = os.path.join(args.output_dir, "gp_test_predictions.csv")

    metrics_df.to_csv(metrics_path, index=False)
    results_df.to_csv(results_path, index=False)

    print("\nMétricas de inferencia:")
    print(metrics_df)

    print("\nArchivos guardados:")
    print(metrics_path)
    print(results_path)

    if args.preview > 0:
        print("\nPrimeras predicciones:")
        print(results_df.head(args.preview))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Carpeta con los .txt del dataset"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Carpeta donde están los modelos GP entrenados"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="gp_inference_results",
        help="Carpeta donde guardar resultados de inferencia"
    )

    parser.add_argument(
        "--history_len",
        type=int,
        default=20
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=10
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
        "--max_files",
        type=int,
        default=None
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Número de filas a mostrar en consola"
    )

    args = parser.parse_args()

    main(args)