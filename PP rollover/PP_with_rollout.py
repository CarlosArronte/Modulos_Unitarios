import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
from math import ceil

from controllers.base_controller import BaseController
from controllers.pp_controller import PurePursuitController

from loggers.telemetry_logger import TelemetryLogger

logger = TelemetryLogger("run_pp.csv",0.33, dt=0.01)
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
# CONFIGURACIÓN DS
# ==============================
TARGET_SAMPLES = 50000     # líneas totales deseadas
WARMUP_STEPS = 7           # pasos ignorados tras reset


def main():
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

    while True:
        speed, steer = controller.plan(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            tlad=work['tlad'],
            vgain=work['vgain'],
        )

        obs, _, done, info = env.step(
            np.array([[steer, speed]])
        )

        #PP rollout 
        U_future, pp_future_info = controller.get_future_pp_controls(
            obs['poses_x'][0],
            obs['poses_y'][0],
            obs['poses_theta'][0],
            tlad=work['tlad'],
            vgain=work['vgain'],
            horizon=30,
            waypoint_step=5,
            include_current=True,
        )

        print(U_future.shape)
        print(U_future[:5])

        # ==============================
        # LOGGING (ignorar warmup)
        # ==============================
        if step_in_lap >= WARMUP_STEPS:
            logger.step(
                obs=obs,
                steer_cmd=steer,
                speed_cmd=speed,
            )
            total_steps += 1

        step_in_lap += 1
        env.render(mode='human')

        # ==============================
        # COLISIÓN → reset
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

    logger.close()

    print(
        "Total laps:", lap,
        "Total samples:", total_steps,
        "Real elapsed time:", time.time() - start,
    )


if __name__ == '__main__':
    main()
