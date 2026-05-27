import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_STATES = ["Vy", "AVz", "Beta", "Ax", "Ay"]


def load_results(results_dir):
    real_path = os.path.join(results_dir, "real_states.csv")
    rollout_path = os.path.join(results_dir, "rollout_predictions.csv")

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"No encontré: {real_path}")

    if not os.path.exists(rollout_path):
        raise FileNotFoundError(f"No encontré: {rollout_path}")

    real_df = pd.read_csv(real_path)
    rollout_df = pd.read_csv(rollout_path)

    return real_df, rollout_df


def merge_predictions_with_real(real_df, rollout_df, states):
    """
    Une cada predicción futura con el estado real medido cuando el simulador
    alcanza ese target_step.
    """

    merged = rollout_df.merge(
        real_df,
        left_on="target_step",
        right_on="global_step",
        how="left",
        suffixes=("", "_real_obs"),
    )

    for state in states:
        pred_col = f"{state}_mean"
        std_col = f"{state}_std"
        real_col = f"{state}_real"

        if real_col not in merged.columns:
            print(f"[WARN] No existe columna real para {state}: {real_col}")
            continue

        merged[f"{state}_error"] = merged[real_col] - merged[pred_col]
        merged[f"{state}_abs_error"] = np.abs(merged[f"{state}_error"])
        merged[f"{state}_inside_1std"] = (
            merged[f"{state}_abs_error"] <= merged[std_col]
        )
        merged[f"{state}_inside_2std"] = (
            merged[f"{state}_abs_error"] <= 2.0 * merged[std_col]
        )

    return merged


def compute_summary(merged_df, states):
    rows = []

    for state in states:
        real_col = f"{state}_real"
        mean_col = f"{state}_mean"
        std_col = f"{state}_std"
        abs_err_col = f"{state}_abs_error"

        if real_col not in merged_df.columns:
            continue

        df = merged_df.dropna(subset=[real_col, mean_col, std_col])

        if len(df) == 0:
            continue

        rows.append({
            "state": state,
            "n": len(df),
            "mae": df[abs_err_col].mean(),
            "rmse": np.sqrt(np.mean((df[real_col] - df[mean_col]) ** 2)),
            "error_p50": df[abs_err_col].quantile(0.50),
            "error_p90": df[abs_err_col].quantile(0.90),
            "error_p95": df[abs_err_col].quantile(0.95),
            "std_mean": df[std_col].mean(),
            "std_p90": df[std_col].quantile(0.90),
            "std_p95": df[std_col].quantile(0.95),
            "coverage_1std": df[f"{state}_inside_1std"].mean(),
            "coverage_2std": df[f"{state}_inside_2std"].mean(),
        })

    return pd.DataFrame(rows)


def plot_single_rollout(merged_df, rollout_start_step, states, output_dir, k_sigma=2.0):
    """
    Grafica un rollout específico:
        media ± k_sigma*std vs real.
    """

    os.makedirs(output_dir, exist_ok=True)

    df = merged_df[merged_df["rollout_start_step"] == rollout_start_step].copy()

    if df.empty:
        raise ValueError(f"No hay rollout_start_step={rollout_start_step}")

    for state in states:
        real_col = f"{state}_real"
        mean_col = f"{state}_mean"
        std_col = f"{state}_std"

        if real_col not in df.columns:
            continue

        k = df["k"].values
        mean = df[mean_col].values
        std = df[std_col].values
        real = df[real_col].values

        upper = mean + k_sigma * std
        lower = mean - k_sigma * std

        plt.figure(figsize=(9, 5))
        plt.plot(k, mean, label=f"{state} mean")
        plt.fill_between(k, lower, upper, alpha=0.25, label=f"±{k_sigma} std")
        plt.plot(k, real, marker="o", linestyle="--", label=f"{state} real")

        plt.xlabel("Prediction step k")
        plt.ylabel(state)
        plt.title(f"{state}: rollout start step {rollout_start_step}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            output_dir,
            f"rollout_{rollout_start_step}_{state}.png"
        )
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Figura guardada: {out_path}")


def plot_error_by_horizon(merged_df, states, output_dir):
    """
    Grafica cómo crece el error absoluto medio con el horizonte k.
    """

    os.makedirs(output_dir, exist_ok=True)

    for state in states:
        abs_err_col = f"{state}_abs_error"

        if abs_err_col not in merged_df.columns:
            continue

        df = (
            merged_df
            .dropna(subset=[abs_err_col])
            .groupby("k")[abs_err_col]
            .agg(["mean", "median", "count"])
            .reset_index()
        )

        plt.figure(figsize=(9, 5))
        plt.plot(df["k"], df["mean"], marker="o", label="mean abs error")
        plt.plot(df["k"], df["median"], marker="s", label="median abs error")

        plt.xlabel("Prediction step k")
        plt.ylabel(f"{state} absolute error")
        plt.title(f"{state}: error vs prediction horizon")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"error_by_horizon_{state}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Figura guardada: {out_path}")


def plot_uncertainty_by_horizon(merged_df, states, output_dir):
    """
    Grafica la incertidumbre media con el horizonte k.
    """

    os.makedirs(output_dir, exist_ok=True)

    for state in states:
        std_col = f"{state}_std"

        if std_col not in merged_df.columns:
            continue

        df = (
            merged_df
            .dropna(subset=[std_col])
            .groupby("k")[std_col]
            .agg(["mean", "median"])
            .reset_index()
        )

        plt.figure(figsize=(9, 5))
        plt.plot(df["k"], df["mean"], marker="o", label="mean std")
        plt.plot(df["k"], df["median"], marker="s", label="median std")

        plt.xlabel("Prediction step k")
        plt.ylabel(f"{state} MC std")
        plt.title(f"{state}: uncertainty vs prediction horizon")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"uncertainty_by_horizon_{state}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Figura guardada: {out_path}")


def plot_error_vs_uncertainty(merged_df, states, output_dir):
    """
    Dispersión error absoluto vs std MC.
    Sirve para ver si la incertidumbre crece cuando crece el error.
    """

    os.makedirs(output_dir, exist_ok=True)

    for state in states:
        abs_err_col = f"{state}_abs_error"
        std_col = f"{state}_std"

        if abs_err_col not in merged_df.columns:
            continue

        df = merged_df.dropna(subset=[abs_err_col, std_col])

        if len(df) == 0:
            continue

        corr = df[abs_err_col].corr(df[std_col])

        plt.figure(figsize=(6, 6))
        plt.scatter(df[std_col], df[abs_err_col], alpha=0.25, s=10)

        plt.xlabel(f"{state} MC std")
        plt.ylabel(f"{state} absolute error")
        plt.title(f"{state}: error vs uncertainty | corr={corr:.3f}")
        plt.grid(True)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"error_vs_uncertainty_{state}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Figura guardada: {out_path}")


def select_rollout_start(merged_df, mode="first", state="Beta"):
    """
    Selecciona un rollout para graficar.

    mode:
        first      -> primer rollout disponible
        worst      -> rollout con mayor error medio en una variable
        uncertain  -> rollout con mayor incertidumbre media en una variable
    """

    starts = sorted(merged_df["rollout_start_step"].dropna().unique())

    if not starts:
        raise ValueError("No hay rollout_start_step disponibles.")

    if mode == "first":
        return int(starts[0])

    if mode == "worst":
        col = f"{state}_abs_error"
        if col not in merged_df.columns:
            raise ValueError(f"No existe columna {col}")

        grouped = (
            merged_df
            .dropna(subset=[col])
            .groupby("rollout_start_step")[col]
            .mean()
            .sort_values(ascending=False)
        )
        return int(grouped.index[0])

    if mode == "uncertain":
        col = f"{state}_std"
        if col not in merged_df.columns:
            raise ValueError(f"No existe columna {col}")

        grouped = (
            merged_df
            .dropna(subset=[col])
            .groupby("rollout_start_step")[col]
            .mean()
            .sort_values(ascending=False)
        )
        return int(grouped.index[0])

    raise ValueError(f"Modo no reconocido: {mode}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    states = args.states.split(",")

    real_df, rollout_df = load_results(args.results_dir)

    merged_df = merge_predictions_with_real(
        real_df=real_df,
        rollout_df=rollout_df,
        states=states,
    )

    merged_path = os.path.join(args.output_dir, "merged_predictions_vs_real.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"[OK] Merge guardado: {merged_path}")

    summary_df = compute_summary(merged_df, states)
    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Summary guardado: {summary_path}")

    print("\nResumen:")
    print(summary_df)

    rollout_start = args.rollout_start

    if rollout_start is None:
        rollout_start = select_rollout_start(
            merged_df,
            mode=args.rollout_mode,
            state=args.selection_state,
        )

    print(f"\nRollout seleccionado para gráficos: {rollout_start}")

    plot_single_rollout(
        merged_df=merged_df,
        rollout_start_step=rollout_start,
        states=states,
        output_dir=args.output_dir,
        k_sigma=args.k_sigma,
    )

    plot_error_by_horizon(
        merged_df=merged_df,
        states=states,
        output_dir=args.output_dir,
    )

    plot_uncertainty_by_horizon(
        merged_df=merged_df,
        states=states,
        output_dir=args.output_dir,
    )

    plot_error_vs_uncertainty(
        merged_df=merged_df,
        states=states,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Carpeta con real_states.csv y rollout_predictions.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="presentation_results",
        help="Carpeta donde se guardan figuras y tablas",
    )

    parser.add_argument(
        "--states",
        type=str,
        default="Vy,AVz,Beta,Ax,Ay",
        help="Estados a graficar separados por coma",
    )

    parser.add_argument(
        "--k_sigma",
        type=float,
        default=2.0,
        help="Factor para envelope: media ± k_sigma*std",
    )

    parser.add_argument(
        "--rollout_start",
        type=int,
        default=None,
        help="rollout_start_step específico. Si no se indica, se selecciona automáticamente.",
    )

    parser.add_argument(
        "--rollout_mode",
        type=str,
        default="worst",
        choices=["first", "worst", "uncertain"],
        help="Modo de selección automática del rollout.",
    )

    parser.add_argument(
        "--selection_state",
        type=str,
        default="Beta",
        help="Variable usada para seleccionar rollout worst/uncertain.",
    )

    args = parser.parse_args()
    main(args)