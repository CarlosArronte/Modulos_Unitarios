import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
from math import ceil
import sys
from pathlib import Path
import os
import argparse
import torch
from collections import deque
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pp_rollover.controllers.base_controller import BaseController
from pp_rollover.controllers.pp_controller import PurePursuitController

from dynamic_supervisor.PP_MC_supervisor_rollout import (
    load_mc_dropout_model,
    rollout_dynamics_mc_dropout,
    build_rollout_results_df,
)


# ============================================================
# CONTROLLER FACTORY (Open–Closed Principle)
# ============================================================

def create_controller(conf) -> BaseController:
    return PurePursuitController(
        conf=conf,
        wheelbase=0.17145 + 0.15875,
    )

# ============================================================
# RENDER CALLBACK   
# ============================================================

def build_render_callback(controller: BaseController):
    def render_callback(env_renderer):
        e = env_renderer

        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]

        e.left = min(x) - 800
        e.right = max(x) + 800
        e.bottom = min(y) - 800
        e.top = max(y) + 800

        controller.render_waypoints(e)

    return render_callback

# ==============================
# STATE EXTRACTOR
# ==============================

def compute_realtime_state(obs, prev_obs=None, dt=0.01, vx_min=1.0):
    vx = float(obs["linear_vels_x"][0])
    vy = float(obs["linear_vels_y"][0])
    avz = float(obs["ang_vels_z"][0])

    # Beta robusta
    beta = np.rad2deg(np.arctan2(vy, max(abs(vx), vx_min)))

    if prev_obs is None:
        ax = 0.0
        ay = 0.0
    else:
        vx_prev = float(prev_obs["linear_vels_x"][0])
        vy_prev = float(prev_obs["linear_vels_y"][0])

        ax = (vx - vx_prev) / dt
        ay_raw = (vy - vy_prev) / dt

        # Aproximación útil de aceleración lateral en frame vehículo:
        # ay ≈ dVy/dt + Vx * yaw_rate
        ay = ay_raw + vx * avz

    return {
        "Vx": vx,
        "Vy": vy,
        "AVz": np.rad2deg(avz),
        "Beta": beta,
        "Ax": ax,
        "Ay": ay,
    }

STATE_COLS = ["Vy", "AVz", "Beta", "Ax", "Ay"]
CONTROL_COLS = ["Steer", "Vx"]

def main(args):

    #device by args
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    #PP params 
    work = {
        'tlad': 0.82461887897713965,
        'vgain': 1.375,
    }

    with open('config_map.yaml') as f:
        conf = Namespace(**yaml.safe_load(f))

    controller = create_controller(conf)

 

    env = gym.make(
        'f110_gym:f110-v0',
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )

    env.add_render_callback(build_render_callback(controller))

    obs, _, _, _ = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )

    env.render()

    lap = 0
    last_lap_count = obs['lap_counts'][0]
    step_in_lap = 0
    total_steps = 0

    samples_per_lap_estimate = None
    max_laps_estimated = None

    start = time.time()

   #Cargar modelo de MC Dropout
    model, x_scaler, y_scaler, metadata = load_mc_dropout_model(
    args.model_dir,
    device
    )

    history_len = metadata["history_len"]
    history_window = deque(maxlen=history_len + 1)

    prev_obs = None

    real_state_records = []
    rollout_records = []

    os.makedirs(args.output_dir, exist_ok=True)

    while True:
        if total_steps >= args.max_steps:
            break
        else:
            print(f"Step {total_steps}")

        #Calculo u(t) con el PP y la raceline
        speed, steer = controller.plan(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            tlad=work['tlad'],
            vgain=work['vgain'],
        )

        #Aplicar acción en el entorno
        obs, _, done, info = env.step(
            np.array([[steer, speed]])
        )

        #Actualizar estado del entorno
        current_state = compute_realtime_state(
            obs,
            prev_obs=prev_obs,
            dt=0.01,)  

        real_state_records.append({
            "global_step": total_steps,
            "lap": lap,
            "step_in_lap": step_in_lap,
            "pose_x": float(obs["poses_x"][0]),
            "pose_y": float(obs["poses_y"][0]),
            "pose_theta": float(obs["poses_theta"][0]),
            "Vx_real": current_state["Vx"],
            "Vy_real": current_state["Vy"],
            "AVz_real": current_state["AVz"],
            "Beta_real": current_state["Beta"],
            "Ax_real": current_state["Ax"],
            "Ay_real": current_state["Ay"],
            "Steer_cmd": steer,
            "Vx_cmd": speed,
        })

        prev_obs = obs.copy()

        feature_row = np.array([
            current_state["Vy"],
            current_state["AVz"],
            current_state["Beta"],
            current_state["Ax"],
            current_state["Ay"],            
            steer,
            speed,
        ], dtype=np.float64) 
        
        history_window.append(feature_row)

        if len(history_window) < history_len + 1:
            if total_steps % 20 == 0:
                print(f"History window: {len(history_window)}/{history_len + 1}")
            step_in_lap += 1
            total_steps += 1
            continue

        run_supervisor = (step_in_lap % args.supervisor_every == 0)

        if run_supervisor:
            history_array = np.asarray(history_window, dtype=np.float64)

            #PP rollout 
            U_future, pp_future_info = controller.get_future_pp_controls(
                obs['poses_x'][0],
                obs['poses_y'][0],
                obs['poses_theta'][0],
                tlad=work['tlad'],
                vgain=work['vgain'],
                horizon=args.horizon,
                waypoint_step=args.waypoint_step,
                include_current=True,
            )

            # ----------------------------------------
                # Rollout del modelo dinámico con MC Dropout
                # ----------------------------------------
            means, stds = rollout_dynamics_mc_dropout(
                model=model,
                history_window=history_array,
                U_future=U_future,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                device=device,
                n_mc=args.n_mc,
            )

            for k in range(args.horizon):
                row = {
                    "rollout_start_step": total_steps,
                    "target_step": total_steps + k + 1,
                    "k": k,
                    "pose_x_start": float(obs["poses_x"][0]),
                    "pose_y_start": float(obs["poses_y"][0]),
                    "pose_theta_start": float(obs["poses_theta"][0]),
                    "Steer_cmd_future": float(U_future[k, 0]),
                    "Vx_cmd_future": float(U_future[k, 1]),
                }

                for i, state in enumerate(STATE_COLS):
                    row[f"{state}_mean"] = float(means[k, i])
                    row[f"{state}_std"] = float(stds[k, i])

                rollout_records.append(row)

            # ----------------------------------------
        # Guardar resultados
        # ----------------------------------------
        

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
    
        step_in_lap += 1
        total_steps += 1
        env.render(mode='human')

        # ==============================
        # COLISIÓN → reset (dejar en el punto anterior al choque)
        # ==============================
        if obs['collisions'][0] == 1:
            print(
                f"COLLISION | x={obs['poses_x'][0]:.2f} "
                f"y={obs['poses_y'][0]:.2f} "
                f"v={obs['linear_vels_x'][0]:.2f}"
            )

            obs, _, _, _ = env.reset(
                np.array([[conf.sx, conf.sy, conf.stheta]])
            )
            step_in_lap = 0
            continue

        # ==============================
        # VUELTA COMPLETADA
        # ==============================
        current_lap = obs['lap_counts'][0]
        if current_lap > last_lap_count:
            lap += 1

            effective_steps = max(0, step_in_lap - WARMUP_STEPS)
            print(f"Lap {lap} completed | steps: {effective_steps}")

            # Estimar muestras por vuelta tras la primera vuelta limpia
            if samples_per_lap_estimate is None and effective_steps > 0:
                samples_per_lap_estimate = effective_steps
                max_laps_estimated = ceil(
                    TARGET_SAMPLES / samples_per_lap_estimate
                )
                print(
                    f"[INFO] Estimated samples/lap: {samples_per_lap_estimate} | "
                    f"Estimated MAX_LAPS: {max_laps_estimated}"
                )

            last_lap_count = current_lap
            step_in_lap = 0

            # Corte principal por muestras reales
            if total_steps >= TARGET_SAMPLES:
                print("[INFO] Target samples reached.")
                break

            # Corte secundario de seguridad (por vueltas)
            if (
                max_laps_estimated is not None
                and lap >= max_laps_estimated
            ):
                print("[INFO] Estimated max laps reached.")
                break

    real_path = os.path.join(args.output_dir, "real_states.csv")
    rollout_path = os.path.join(args.output_dir, "rollout_predictions.csv")

    pd.DataFrame(real_state_records).to_csv(real_path, index=False)
    pd.DataFrame(rollout_records).to_csv(rollout_path, index=False)

    print("\nArchivos guardados:")
    print(real_path)
    print(rollout_path)
    print(
        "Total laps:", lap,
        "Total samples:", total_steps,
        "Real elapsed time:", time.time() - start,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="supervisor_rollout_results")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--waypoint_step", type=int, default=5)

    parser.add_argument("--tlad", type=float, default=0.82461887897713965)
    parser.add_argument("--vgain", type=float, default=1.375)
    parser.add_argument("--n_mc", type=int, default=50)

    parser.add_argument("--supervisor_every", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_every", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=3000)

    parser.add_argument("--cpu", action="store_true")


    args = parser.parse_args()
    main(args)