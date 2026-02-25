"""Configuration for the FacadeDrone environment.

Aligned with PLAN.md and ROADMAP.md Phase 1 specifications:
- 23-dim observation space (rotation matrix, not quaternion — per SimpleFlight)
- 4-dim CTBR action space [thrust, omega_roll, omega_pitch, omega_yaw]
- 200Hz physics / 50Hz policy (decimation=4)
- Target wall distance: 1.5m
- 4096 parallel environments
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# M350 RTK platform constants (from PLAN.md — [EST] values for initial sim)
# ---------------------------------------------------------------------------
M350_MASS = 6.47  # kg — aircraft + 2x TB65, no payload
M350_ARM_LENGTH = 0.4475  # m — center to motor
M350_MAX_TILT_DEG = 30.0  # degrees
M350_MAX_ANGULAR_VEL_PITCH = 5.24  # rad/s (~300 deg/s)
M350_MAX_ANGULAR_VEL_YAW = 1.75  # rad/s (~100 deg/s)
M350_MAX_THRUST_PER_MOTOR = 45.0  # N [EST]

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


@configclass
class FacadeDroneEnvCfg(DirectRLEnvCfg):
    """Configuration for the facade drone training environment."""

    # --- MDP dimensions ---
    episode_length_s: float = 30.0
    decimation: int = 4  # 200Hz physics / 50Hz policy
    action_space: int = 4  # [thrust, omega_roll, omega_pitch, omega_yaw]
    observation_space: int = 23  # see observation breakdown below
    state_space: int = 0

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,  # 200Hz physics
        render_interval=decimation,
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        replicate_physics=True,
    )

    # --- Robot asset ---
    # Using Crazyflie as initial proxy; replace with M350 URDF/USD in Phase 2
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="${ISAACLAB_ASSETS_DIR}/Robots/Crazyflie/cf2x.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 0.0, 1.5),  # 1.5m from wall (x), 1.5m altitude (z)
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        ),
    )

    # --- Wall model ---
    wall_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 50.0, 100.0),  # thin in x, wide in y, tall in z
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # static wall
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.7),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 50.0),  # wall center
        ),
    )

    # --- Task parameters ---
    target_wall_distance: float = 1.5  # m — desired distance from wall
    wall_position_x: float = 0.0  # x-coordinate of wall surface

    # --- Action scaling ---
    # CTBR action bounds (policy outputs normalized [-1, 1], we scale)
    thrust_to_weight: float = 1.9  # max thrust as multiple of hover thrust
    moment_scale: float = 0.02  # Nm per unit action for body rates

    # --- Reward weights ---
    rew_distance_scale: float = 1.0
    rew_oscillation_scale: float = 1.0
    rew_smoothness_scale: float = 1.0
    rew_energy_scale: float = 1.0
    rew_alive: float = 0.5
    rew_spray_scale: float = 1.0

    # --- Domain randomization ---
    # Wind parameters (curriculum-controlled)
    wind_max_speed: float = 0.0  # updated by curriculum
    wind_ou_theta: float = 0.15  # Ornstein-Uhlenbeck mean reversion rate

    # Spray parameters (curriculum-controlled)
    spray_enabled: bool = False
    spray_force_max: float = 0.0  # N
    spray_cone_half_angle: float = 0.3  # rad (~17 degrees)

    # Mass randomization
    mass_scale_range: tuple[float, float] = (0.85, 1.15)  # ±15%

    # Initial position noise (curriculum-controlled)
    init_pos_noise: float = 0.1  # m

    # --- Termination ---
    min_wall_distance: float = 0.1  # crash threshold
    max_wall_distance: float = 5.0  # drift threshold
    min_altitude: float = 0.5
    max_altitude: float = 100.0
