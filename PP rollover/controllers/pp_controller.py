import numpy as np
from numba import njit
from pyglet.gl import GL_POINTS

from controllers.base_controller import BaseController


# ============================================================
# NUMBA HELPERS
# ============================================================

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[1:] - trajectory[:-1]
    l2s = diffs[:, 0]**2 + diffs[:, 1]**2

    dots = np.empty(len(l2s))
    for i in range(len(dots)):
        dots[i] = np.dot(point - trajectory[i], diffs[i])

    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0

    #projections = trajectory[:-1] + (t[:, None] * diffs)
    projections = trajectory[:-1] + (t.reshape((t.shape[0], 1)) * diffs) #by numba version

    dists = np.empty(len(projections))
    for i in range(len(dists)):
        d = point - projections[i]
        dists[i] = np.sqrt(np.dot(d, d))

    idx = np.argmin(dists)
    return projections[idx], dists[idx], t[idx], idx


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    start_i = int(t)
    start_t = t % 1.0

    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i]
        end = trajectory[i + 1] + 1e-6
        V = end - start

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - 2.0 * np.dot(start, point)
            - radius**2
        )

        disc = b * b - 4 * a * c
        if disc < 0:
            continue

        disc = np.sqrt(disc)
        t1 = (-b - disc) / (2 * a)
        t2 = (-b + disc) / (2 * a)

        if i == start_i:
            if 0 <= t1 <= 1 and t1 >= start_t:
                return start + t1 * V, i, t1
            if 0 <= t2 <= 1 and t2 >= start_t:
                return start + t2 * V, i, t2
        else:
            if 0 <= t1 <= 1:
                return start + t1 * V, i, t1
            if 0 <= t2 <= 1:
                return start + t2 * V, i, t2

    if wrap:
        for i in range(start_i):
            start = trajectory[i]
            end = trajectory[i + 1] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - 2.0 * np.dot(start, point)
                - radius**2
            )

            disc = b * b - 4 * a * c
            if disc < 0:
                continue

            disc = np.sqrt(disc)
            t1 = (-b - disc) / (2 * a)
            t2 = (-b + disc) / (2 * a)

            if 0 <= t1 <= 1:
                return start + t1 * V, i, t1
            if 0 <= t2 <= 1:
                return start + t2 * V, i, t2

    return None, None, None


@njit(fastmath=False, cache=True)
def get_actuation_curvature(
    theta,
    lookahead_point,
    position,
    Ld,
    wheelbase,
    v_max,
    v_min,
    kappa_gain,
):
    waypoint_y = np.dot(
        np.array([np.sin(-theta), np.cos(-theta)]),
        lookahead_point[:2] - position,
    )

    if np.abs(waypoint_y) < 1e-6:
        return v_max, 0.0, 0.0

    curvature = 2.0 * waypoint_y / (Ld * Ld)

    # 1) velocidad base por curvatura
    speed = v_max / (1.0 + kappa_gain * np.abs(curvature))

    # 2) límite físico por aceleración lateral
    a_lat_max = 5.0
    speed_lat = np.sqrt(a_lat_max / (np.abs(curvature) + 1e-6))
    speed = min(speed, speed_lat)

    # 3) límite por waypoint
    speed = min(speed, lookahead_point[2])
    speed = min(max(speed, v_min), v_max)

    steering = np.arctan(wheelbase * curvature)

    return speed, steering, curvature


# ============================================================
# PURE PURSUIT CONTROLLER
# ============================================================

class PurePursuitController(BaseController):
    def __init__(self, conf, wheelbase):
        self.conf = conf
        self.wheelbase = wheelbase
        self.max_reacquire = 20.0
        self.drawn_waypoints = []

        # ----------------------------------------
        # Load raw CSV
        # ----------------------------------------
        raw_wpts = np.loadtxt(
            conf.wpt_path,
            delimiter=conf.wpt_delim,
            skiprows=conf.wpt_rowskip,
        )

        # ----------------------------------------
        # Geometry (x, y)
        # ----------------------------------------
        self.wpts_xy = raw_wpts[:, [conf.wpt_xind, conf.wpt_yind]]

        # ----------------------------------------
        # Waypoint type handling
        # ----------------------------------------
        if conf.wpt_path_type == 2:
            # Raceline: velocity provided
            if not hasattr(conf, "wpt_vind"):
                raise ValueError(
                    "wpt_vind must be defined when wpt_path_type == 2 (raceline)"
                )
            v_profile = raw_wpts[:, conf.wpt_vind]

        elif conf.wpt_path_type == 1:
            # Centerline: constant max velocity
            v_profile = np.full(
                (self.wpts_xy.shape[0],),
                conf.v_max,
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Invalid wpt_path_type={conf.wpt_path_type} "
                "(expected 1=centerline, 2=raceline)"
            )

        # ----------------------------------------
        # Unified waypoint format: [x, y, v]
        # ----------------------------------------
        self.waypoints = np.column_stack((
            self.wpts_xy,
            v_profile,
        ))

    def render_waypoints(self, env_renderer):
        scaled = 50.0 * self.wpts_xy

        for i in range(self.wpts_xy.shape[0]):
            if len(self.drawn_waypoints) < self.wpts_xy.shape[0]:
                b = env_renderer.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ('v3f/stream', [scaled[i, 0], scaled[i, 1], 0.0]),
                    ('c3B/stream', [183, 193, 222]),
                )
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled[i, 0], scaled[i, 1], 0.0
                ]

    def plan(self, pose_x, pose_y, pose_theta, *, tlad, vgain):
        position = np.array([pose_x, pose_y])

        nearest, dist, t, idx = nearest_point_on_trajectory(
            position, self.wpts_xy
        )

        if dist < tlad:
            p, i2, _ = first_point_on_trajectory_intersecting_circle(
                position, tlad, self.wpts_xy, idx + t, wrap=True
            )
            if p is None:
                return 0.0, 0.0

            wp = np.array([
                p[0],
                p[1],
                self.waypoints[i2, 2],
            ])

        elif dist < self.max_reacquire:
            wp = np.array([
                self.wpts_xy[idx, 0],
                self.wpts_xy[idx, 1],
                self.waypoints[idx, 2],
            ])
        else:
            return 0.0, 0.0

        # ----------------------------------------
        # Adaptive lookahead
        # ----------------------------------------
        speed0, _, curvature0 = get_actuation_curvature(
            pose_theta,
            wp,
            position,
            tlad,
            self.wheelbase,
            self.conf.v_max,
            self.conf.v_min,
            self.conf.kappa_gain,
        )

        tlad_eff = np.clip(
            tlad / (1.0 + 1.5 * abs(curvature0)),
            0.6,
            1.8,
        )

        speed, steer, _ = get_actuation_curvature(
            pose_theta,
            wp,
            position,
            tlad_eff,
            self.wheelbase,
            self.conf.v_max,
            self.conf.v_min,
            self.conf.kappa_gain,
        )

        return vgain * speed, steer

        #Rollout code
    def _wrap_index(self, idx):
        """
        Hace wrap del índice para pistas cerradas.
        """
        return int(idx % self.wpts_xy.shape[0])


    def _path_heading_at_index(self, idx):
        """
        Calcula la orientación local de la raceline/centerline en un índice dado.
        """
        idx0 = self._wrap_index(idx)
        idx1 = self._wrap_index(idx + 1)

        p0 = self.wpts_xy[idx0]
        p1 = self.wpts_xy[idx1]

        d = p1 - p0
        return np.arctan2(d[1], d[0])


    def _get_virtual_pose_from_index(self, idx):
        """
        Devuelve una pose virtual sobre la raceline/centerline:
            x, y, theta
        """
        idx = self._wrap_index(idx)

        x = self.wpts_xy[idx, 0]
        y = self.wpts_xy[idx, 1]
        theta = self._path_heading_at_index(idx)

        return x, y, theta


    def plan_with_info(self, pose_x, pose_y, pose_theta, *, tlad, vgain):
        """
        Igual que plan(), pero devuelve información extra útil para debugging
        y para construir rollouts.

        Returns
        -------
        speed : float
        steer : float
        info : dict
        """
        position = np.array([pose_x, pose_y])

        nearest, dist, t, idx = nearest_point_on_trajectory(
            position, self.wpts_xy
        )

        if dist < tlad:
            p, i2, _ = first_point_on_trajectory_intersecting_circle(
                position, tlad, self.wpts_xy, idx + t, wrap=True
            )
            if p is None:
                return 0.0, 0.0, {
                    "valid": False,
                    "nearest_idx": int(idx),
                    "target_idx": None,
                    "dist_to_path": float(dist),
                    "reason": "no_lookahead_intersection",
                }

            wp = np.array([
                p[0],
                p[1],
                self.waypoints[i2, 2],
            ])
            target_idx = int(i2)

        elif dist < self.max_reacquire:
            wp = np.array([
                self.wpts_xy[idx, 0],
                self.wpts_xy[idx, 1],
                self.waypoints[idx, 2],
            ])
            target_idx = int(idx)

        else:
            return 0.0, 0.0, {
                "valid": False,
                "nearest_idx": int(idx),
                "target_idx": None,
                "dist_to_path": float(dist),
                "reason": "too_far_from_path",
            }

        speed0, _, curvature0 = get_actuation_curvature(
            pose_theta,
            wp,
            position,
            tlad,
            self.wheelbase,
            self.conf.v_max,
            self.conf.v_min,
            self.conf.kappa_gain,
        )

        tlad_eff = np.clip(
            tlad / (1.0 + 1.5 * abs(curvature0)),
            0.6,
            1.8,
        )

        speed, steer, curvature = get_actuation_curvature(
            pose_theta,
            wp,
            position,
            tlad_eff,
            self.wheelbase,
            self.conf.v_max,
            self.conf.v_min,
            self.conf.kappa_gain,
        )

        info = {
            "valid": True,
            "nearest_idx": int(idx),
            "target_idx": int(target_idx),
            "dist_to_path": float(dist),
            "t": float(t),
            "tlad": float(tlad),
            "tlad_eff": float(tlad_eff),
            "curvature": float(curvature),
            "waypoint_x": float(wp[0]),
            "waypoint_y": float(wp[1]),
            "waypoint_v": float(wp[2]),
        }

        return vgain * speed, steer, info


    def plan(self, pose_x, pose_y, pose_theta, *, tlad, vgain):
        """
        Mantiene compatibilidad con tu código actual.
        """
        speed, steer, _ = self.plan_with_info(
            pose_x,
            pose_y,
            pose_theta,
            tlad=tlad,
            vgain=vgain,
        )
        return speed, steer


    def get_future_pp_controls(
        self,
        pose_x,
        pose_y,
        pose_theta,
        *,
        tlad,
        vgain,
        horizon=20,
        waypoint_step=5,
        include_current=True,
    ):
        """
        Genera una secuencia futura nominal de comandos PP usando la raceline/centerline.

        La función NO avanza el simulador.
        La función NO usa el modelo dinámico.
        Solo genera los comandos futuros nominales:

            U_future[k] = [Steer_k, Vx_k]

        Parameters
        ----------
        pose_x, pose_y, pose_theta : float
            Pose actual del vehículo.

        tlad : float
            Lookahead base del Pure Pursuit.

        vgain : float
            Ganancia de velocidad.

        horizon : int
            Número de pasos futuros a generar.

        waypoint_step : int
            Salto en índices de waypoint entre pasos futuros.
            Ejemplo: waypoint_step=5 toma poses virtuales cada 5 waypoints.

        include_current : bool
            Si True, el primer comando se calcula desde la pose actual real.
            Si False, todos los comandos se calculan desde poses virtuales futuras.

        Returns
        -------
        U_future : np.ndarray, shape (horizon, 2)
            Cada fila es [steer, speed].

        info_list : list[dict]
            Información auxiliar de cada paso.
        """

        position = np.array([pose_x, pose_y])

        nearest, dist, t, idx0 = nearest_point_on_trajectory(
            position, self.wpts_xy
        )

        U_future = np.zeros((horizon, 2), dtype=np.float64)
        info_list = []

        for k in range(horizon):
            if k == 0 and include_current:
                px = pose_x
                py = pose_y
                ptheta = pose_theta
                virtual_idx = int(idx0)
            else:
                virtual_idx = self._wrap_index(idx0 + k * waypoint_step)
                px, py, ptheta = self._get_virtual_pose_from_index(virtual_idx)

            speed, steer, info = self.plan_with_info(
                px,
                py,
                ptheta,
                tlad=tlad,
                vgain=vgain,
            )

            # Ojo: F1TENTH env.step espera [steer, speed]
            U_future[k, 0] = steer
            U_future[k, 1] = speed

            info["rollout_step"] = int(k)
            info["virtual_idx"] = int(virtual_idx)
            info["virtual_x"] = float(px)
            info["virtual_y"] = float(py)
            info["virtual_theta"] = float(ptheta)

            info_list.append(info)

        return U_future, info_list