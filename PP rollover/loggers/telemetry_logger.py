import csv
import time
import numpy as np

EPS = 1e-6


class TelemetryLogger:
    def __init__(self, csv_path, wheelbase, dt=0.01):
        self.dt = dt
        self.wheelbase = wheelbase
        self.csv_path = csv_path

        # --- LiDAR layout fijo ---
        self.n_left = 10
        self.n_front = 30
        self.n_right = 10
        self.n_sectors = self.n_left + self.n_front + self.n_right  # 50

        self.left_sector_size = 36
        self.front_sector_size = 12
        self.right_sector_size = 36

        self.prev_mean_ranges = np.zeros(self.n_sectors)

        self.prev_steer = 0.0
        self.prev2_steer = 0.0
        self.prev_speed = 0.0
        self.prev2_speed = 0.0
        self.last_cmd_time = time.time()

        self.file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self._build_header())

    # --------------------------------------------------
    def _build_header(self):
        h = []

        # LiDAR sector stats (50 sectores)
        for s in range(self.n_sectors):
            h += [
                f"lidar_min_s{s}",
                f"lidar_mean_s{s}",
                f"lidar_std_s{s}",
                f"lidar_p10_s{s}",
                f"lidar_p25_s{s}",
                f"lidar_p50_s{s}",
                f"lidar_p75_s{s}",
                f"lidar_p90_s{s}",
                f"lidar_valid_ratio_s{s}",
                f"lidar_mean_inv_s{s}",
                f"lidar_range_grad_s{s}",
            ]

        # Global LiDAR
        h += [
            "left_free_space",
            "right_free_space",
            "front_free_space",
            "free_space_lr",
            "track_width_est",
            "center_offset_lidar",
        ]

        # Vehicle dynamics
        h += [
            "speed",
            "side_slip",
            "yaw_rate",
            "yaw_rate_over_speed",
        ]
                # Curvature & lateral dynamics (local, PP-compatible)
        h += [
            "kappa_pp_est",
            "radius_pp_est",
            "lat_acc_est",
            "lat_acc_norm",
            "steer_rate",
            "steer_acc",
            "speed_rate",
        ]


        # Temporal
        h += [
            "dt",
            "time_since_last_cmd",
            "prev_steer",
            "prev2_steer",
            "prev_delta_steer",
            "prev_speed",
            "prev2_speed",
            "prev_delta_speed",
        ]

        # Control targets
        h += [
            "steering_cmd",
            "speed_cmd",
            "timestamp",
        ]

        return h

    # --------------------------------------------------
    def _sectorize(self, ranges):
        """
        Layout fijo:
        [ 10 x 36 ] [ 30 x 12 ] [ 10 x 36 ] = 1080
        """
        sectors = []
        idx = 0

        # Left
        for _ in range(self.n_left):
            sectors.append(ranges[idx:idx + self.left_sector_size])
            idx += self.left_sector_size

        # Front
        for _ in range(self.n_front):
            sectors.append(ranges[idx:idx + self.front_sector_size])
            idx += self.front_sector_size

        # Right
        for _ in range(self.n_right):
            sectors.append(ranges[idx:idx + self.right_sector_size])
            idx += self.right_sector_size

        # Sanity check (solo en debug mental; no print)
        # assert idx == 1080

        return sectors

    # --------------------------------------------------
    def step(self, obs, steer_cmd, speed_cmd):
        ranges = np.array(obs["scans"][0])
        ranges = np.clip(ranges, 0.0, 30.0)

        sectors = self._sectorize(ranges)

        row = []
        mean_ranges = []

        for i, s in enumerate(sectors):
            valid = np.isfinite(s)
            sv = s[valid]

            if len(sv) == 0:
                sv = np.array([30.0])

            mn = np.min(sv)
            mu = np.mean(sv)
            sd = np.std(sv)
            p10, p25, p50, p75, p90 = np.percentile(sv, [10, 25, 50, 75, 90])
            vr = len(sv) / len(s)
            inv = np.mean(1.0 / (sv + EPS))
            grad = self.prev_mean_ranges[i] - mu

            self.prev_mean_ranges[i] = mu
            mean_ranges.append(mu)

            row += [mn, mu, sd, p10, p25, p50, p75, p90, vr, inv, grad]

        # --------------------------------------------------
        # Global LiDAR (indices exactos)
        left = mean_ranges[0:self.n_left]
        front = mean_ranges[self.n_left:self.n_left + self.n_front]
        right = mean_ranges[self.n_left + self.n_front:]

        left_fs = np.mean(left)
        right_fs = np.mean(right)
        front_fs = np.percentile(front, 90)

        track_width = np.median(left) + np.median(right)
        center_offset = (np.median(right) - np.median(left)) / 2.0

        row += [
            left_fs,
            right_fs,
            front_fs,
            left_fs - right_fs,
            track_width,
            center_offset,
        ]

        # --------------------------------------------------
        # Vehicle dynamics
        vx = obs["linear_vels_x"][0]
        vy = obs["linear_vels_y"][0]
        speed = np.sqrt(vx**2 + vy**2)
        slip = np.arctan2(vy, vx + EPS)
        yaw_rate = obs["ang_vels_z"][0]
        yaw_rate_os = yaw_rate / max(speed, EPS)

        row += [speed, slip, yaw_rate, yaw_rate_os]

                # --------------------------------------------------
        # Curvature & lateral dynamics (local, no map)
        steer = steer_cmd

        kappa = np.tan(steer) / (self.wheelbase + EPS)
        radius = 1.0 / (kappa + EPS)

        lat_acc = speed**2 * kappa
        lat_acc_norm = lat_acc / (speed**2 + EPS)

        steer_rate = (steer - self.prev_steer) / self.dt
        steer_acc = (steer - 2*self.prev_steer + self.prev2_steer) / (self.dt**2)

        speed_rate = (speed - self.prev_speed) / self.dt

        row += [
            kappa,
            radius,
            lat_acc,
            lat_acc_norm,
            steer_rate,
            steer_acc,
            speed_rate,
        ]


        # --------------------------------------------------
        # Temporal
        now = time.time()
        dt_cmd = now - self.last_cmd_time
        self.last_cmd_time = now

        row += [
            self.dt,
            dt_cmd,
            self.prev_steer,
            self.prev2_steer,
            self.prev_steer - self.prev2_steer,
            self.prev_speed,
            self.prev2_speed,
            self.prev_speed - self.prev2_speed,
        ]

        # --------------------------------------------------
        # Control
        row += [
            steer_cmd,
            speed_cmd,
            now,
        ]

        self.writer.writerow(row)
        self.file.flush()

        # update buffers
        self.prev2_steer = self.prev_steer
        self.prev_steer = steer_cmd
        self.prev2_speed = self.prev_speed
        self.prev_speed = speed

    # --------------------------------------------------
    def close(self):
        self.file.close()
