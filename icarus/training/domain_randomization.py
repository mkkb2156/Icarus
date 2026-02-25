"""Domain randomization configuration for facade drone training.

Aligned with PLAN.md DR specifications:
  - Mass: ±15% (simulates water tank fill level)
  - Wind: 0-12 m/s (Ornstein-Uhlenbeck process)
  - Spray recoil: 0-15 N (backward cone)
  - Sensor noise: IMU σ_accel=0.1 m/s², σ_gyro=0.01 rad/s

Based on SimpleFlight (IEEE RA-L 2024) best practices:
  - 10% DR = optimal speed/robustness balance
  - SysID + Selective DR >> Pure DR
  - Motor time constant DR has low impact — skip it
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainRandomizationConfig:
    """Domain randomization ranges for facade drone training."""

    # Mass randomization (per episode reset)
    mass_scale_min: float = 0.85  # -15%
    mass_scale_max: float = 1.15  # +15%

    # Inertia randomization (coupled with mass via recompute_inertia)
    # Handled automatically when mass is randomized with recompute_inertia=True

    # Wind (Ornstein-Uhlenbeck process, curriculum-controlled)
    wind_speed_min: float = 0.0
    wind_speed_max: float = 12.0  # m/s — M350 max wind resistance
    wind_ou_theta: float = 0.15  # mean reversion rate
    wind_direction_change_interval: tuple[float, float] = (3.0, 8.0)  # seconds

    # Spray recoil (when spray_status=1, curriculum-controlled)
    spray_force_min: float = 0.0
    spray_force_max: float = 15.0  # N
    spray_cone_half_angle: float = 0.3  # rad (~17°)

    # Motor response delay (NOT randomized — low impact per SimpleFlight)
    # motor_delay_range: tuple[float, float] = (0.01, 0.05)  # skipped

    # Motor thrust coefficient (optional, for Phase 3 refinement)
    thrust_coeff_scale_min: float = 0.90
    thrust_coeff_scale_max: float = 1.10

    # Sensor noise
    imu_accel_noise_std: float = 0.1  # m/s²
    imu_gyro_noise_std: float = 0.01  # rad/s
    depth_noise_std: float = 0.02  # m (2cm quantization)

    # Initial position variation
    init_pos_noise_min: float = 0.1
    init_pos_noise_max: float = 1.0  # curriculum-controlled


# Default configuration instance
DEFAULT_DR_CONFIG = DomainRandomizationConfig()
