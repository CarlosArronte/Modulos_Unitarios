import numpy as np
from pyglet.gl import GL_POINTS

from controllers.base_controller import BaseController
from controllers.pp_controller import nearest_point_on_trajectory


class MPCController(BaseController):
    """
    Lightweight shooting-based MPC for F1TENTH.
    - Kinematic bicycle model
    - Tracks centerline/raceline waypoints
    - Optimizes over (steer, speed) sequences
    """

    def __init__(
        self,
        conf,
        wheelbase,
        dt=0.01,
        horizon=12,
        num_samples=256,
        steer_limit=0.4189,
        v_min=None,
        v_max=None,
        w_pos=6.0,
        w_heading=1.5,
        w_speed=1.0,
        w_steer=0.2,
        w_smooth=0.5,
        seed=123,
    ):
        self.conf = conf
        self.wheelbase = wheelbase
        self.dt = dt
        self.horizon = horizon
        self.num_samples = num_samples
        self.steer_limit = steer_limit
        self.v_min = conf.v_min if v_min is None else v_min
        self.v_max = conf.v_max if v_max is None else v_max
        self.w_pos = w_pos
        self.w_heading = w_heading
        self.w_speed = w_speed
        self.w_steer = w_steer
        self.w_smooth = w_smooth
        self.rng = np.random.default_rng(seed)
        self.drawn_waypoints = []

        # ----------------------------------------
        # Load raw CSV
        # ----------------------------------------
        raw_wpts = np.loadtxt(
            conf.wpt_path,
            delimiter=conf.wpt_delim,
            skiprows=conf.wpt_rowskip,
        )

        self.wpts_xy = raw_wpts[:, [conf.wpt_xind, conf.wpt_yind]]

        if conf.wpt_path_type == 2:
            if not hasattr(conf, "wpt_vind"):
                raise ValueError(
                    "wpt_vind must be defined when wpt_path_type == 2 (raceline)"
                )
            v_profile = raw_wpts[:, conf.wpt_vind]
        elif conf.wpt_path_type == 1:
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

        self.waypoints = np.column_stack((self.wpts_xy, v_profile))
        self.n_wpts = self.wpts_xy.shape[0]

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

    def _simulate(self, x0, y0, th0, controls):
        xs = np.empty(self.horizon)
        ys = np.empty(self.horizon)
        ths = np.empty(self.horizon)

        x, y, th = x0, y0, th0
        for t in range(self.horizon):
            steer, v = controls[t]
            steer = np.clip(steer, -self.steer_limit, self.steer_limit)
            v = np.clip(v, self.v_min, self.v_max)

            th = th + (v / self.wheelbase) * np.tan(steer) * self.dt
            x = x + v * np.cos(th) * self.dt
            y = y + v * np.sin(th) * self.dt

            xs[t] = x
            ys[t] = y
            ths[t] = th

        return xs, ys, ths

    def _sample_controls(self, v_ref):
        # Sample smooth steer/speed sequences around a reference
        steer_seq = self.rng.normal(0.0, 0.25, size=(self.num_samples, self.horizon))
        steer_seq = np.clip(steer_seq, -self.steer_limit, self.steer_limit)

        speed_seq = self.rng.normal(v_ref, 0.8, size=(self.num_samples, self.horizon))
        speed_seq = np.clip(speed_seq, self.v_min, self.v_max)

        controls = np.stack([steer_seq, speed_seq], axis=2)
        return controls

    def plan(self, pose_x, pose_y, pose_theta, **kwargs):
        position = np.array([pose_x, pose_y])
        _, _, _, idx = nearest_point_on_trajectory(position, self.wpts_xy)

        # Reference points ahead along the waypoint list
        ref_indices = (idx + np.arange(1, self.horizon + 1)) % self.n_wpts
        ref_xy = self.wpts_xy[ref_indices]
        ref_v = self.waypoints[ref_indices, 2]
        v_ref = float(np.clip(ref_v[0], self.v_min, self.v_max))

        controls_batch = self._sample_controls(v_ref)
        best_cost = np.inf
        best_u0 = np.array([0.0, v_ref])

        for k in range(self.num_samples):
            controls = controls_batch[k]
            xs, ys, ths = self._simulate(pose_x, pose_y, pose_theta, controls)

            dx = xs - ref_xy[:, 0]
            dy = ys - ref_xy[:, 1]
            pos_cost = self.w_pos * (dx * dx + dy * dy)

            # Heading error relative to tangent between consecutive refs
            ref_dir = np.arctan2(
                np.roll(ref_xy[:, 1], -1) - ref_xy[:, 1],
                np.roll(ref_xy[:, 0], -1) - ref_xy[:, 0],
            )
            heading_err = np.unwrap(ths - ref_dir)
            heading_cost = self.w_heading * (heading_err * heading_err)

            speed_cost = self.w_speed * (controls[:, 1] - ref_v) ** 2
            steer_cost = self.w_steer * (controls[:, 0] ** 2)

            dsteer = np.diff(controls[:, 0], prepend=controls[0, 0])
            dspeed = np.diff(controls[:, 1], prepend=controls[0, 1])
            smooth_cost = self.w_smooth * (dsteer * dsteer + 0.1 * dspeed * dspeed)

            cost = (
                np.sum(pos_cost)
                + np.sum(heading_cost)
                + np.sum(speed_cost)
                + np.sum(steer_cost)
                + np.sum(smooth_cost)
            )

            if cost < best_cost:
                best_cost = cost
                best_u0 = controls[0]

        steer_cmd = float(np.clip(best_u0[0], -self.steer_limit, self.steer_limit))
        speed_cmd = float(np.clip(best_u0[1], self.v_min, self.v_max))
        return speed_cmd, steer_cmd
