import os
import argparse
import joblib
import numpy as np


def export_scalers(model_dir):
    x_scaler = joblib.load(os.path.join(model_dir, "x_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(model_dir, "y_scaler.pkl"))

    output_path = os.path.join(model_dir, "scaler_params.npz")

    np.savez(
        output_path,
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_,
        y_scale=y_scaler.scale_,
    )

    print("Scaler params exported to:")
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    export_scalers(args.model_dir)