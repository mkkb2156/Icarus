"""FacadeDroneEnv — Isaac Lab DirectRLEnv for drone facade operations training.

This environment trains a quadrotor to maintain d=1.5m distance from a vertical
wall surface while handling wind disturbances and spray recoil forces.

Aligned with PLAN.md Phase 1 / ROADMAP.md Phase 1A specifications.

Action Space (4-dim angular rate + thrust):
    [thrust, omega_roll, omega_pitch, omega_yaw]
    Policy outputs [-1, 1], scaled to physical units.
    In simulation, these map directly to body-frame forces/torques.
    At deployment, these become PSDK angular-rate + thrust setpoints
    (DJI safety arbiter handles motor mixing and failsafe).

Observation Space (23-dim):
    State (18):  lin_vel_body(3) + ang_vel_body(3) + rotation_matrix(9) + position_error(3)
    Environment (4): distance_to_wall(1) + wind_estimate(3)
    Task (1): spray_status(1)

References:
    - Isaac Lab Quadcopter example (DirectRLEnv pattern)
    - SimpleFlight (IEEE RA-L 2024): rotation matrix > quaternion for sim-to-real
    - Swift (Nature 2023): angular rate + thrust action space
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .facade_drone_env_cfg import FacadeDroneEnvCfg
from .reward import compute_facade_reward


class FacadeDroneEnv(DirectRLEnv):
    """Facade drone environment for RL training in Isaac Lab.

    The drone must maintain target_wall_distance (1.5m) from a vertical wall
    while handling wind, spray recoil, and mass variations.
    """

    cfg: FacadeDroneEnvCfg

    def __init__(self, cfg: FacadeDroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache robot properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Action buffers
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Previous action for smoothness penalty
        self._prev_action = torch.zeros_like(self._actions)

        # Wind state (Ornstein-Uhlenbeck process)
        self._wind_velocity = torch.zeros(self.num_envs, 3, device=self.device)

        # Spray state (binary per env)
        self._spray_status = torch.zeros(self.num_envs, device=self.device)
        self._spray_force = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Episode reward tracking (per component, for logging)
        self._reward_components = [
            "distance", "oscillation", "smoothness", "energy", "alive", "spray"
        ]
        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in self._reward_components}

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Spawn robot, wall, and ground plane into the scene."""
        # Spawn robot (Crazyflie proxy — will be replaced with M350 USD)
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Spawn wall as static rigid body
        self._wall = RigidObject(self.cfg.wall_cfg)  # wall_cfg stays as-is (not a registered scene entity)
        self.scene.rigid_objects["wall"] = self._wall

        # Ground plane (using TerrainImporter pattern from quadcopter example)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        # Clone environments and set up collision filtering
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Pre-physics step: convert actions to forces/torques
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Convert 4D angular-rate + thrust actions to body-frame thrust and moments.

        Actions are normalized to [-1, 1] from the policy:
          actions[:, 0] → thrust (mapped to [0, max_thrust] via (a+1)/2)
          actions[:, 1] → roll moment
          actions[:, 2] → pitch moment
          actions[:, 3] → yaw moment
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Thrust: map [-1, 1] → [0, thrust_to_weight * weight]
        # At action=0, thrust = 0.5 * t2w * weight ≈ hover thrust
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )

        # Body moments: scaled linearly
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    # ------------------------------------------------------------------
    # Apply action: external forces (thrust + wind + spray)
    # ------------------------------------------------------------------

    def _apply_action(self):
        """Apply thrust, wind, and spray forces to the robot body."""
        # Update wind (Ornstein-Uhlenbeck step)
        self._step_wind()

        # Update spray force
        self._step_spray()

        # Compute wind force in world frame: F = 0.5 * rho * Cd * A * v^2
        # Simplified: proportional to wind velocity for the training range
        drag_coeff = 0.5  # combined drag factor
        wind_force = drag_coeff * self._wind_velocity  # (N, 3) in world frame

        # Apply forces via wrench composer (same pattern as quadcopter example)
        # Thrust + spray are in body frame; wind is added as world-frame perturbation
        total_forces = self._thrust + self._spray_force + wind_force.unsqueeze(1)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=total_forces,
            torques=self._moment,
        )

    # ------------------------------------------------------------------
    # Observations (23-dim)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Build 23-dim observation tensor.

        State (18): lin_vel_body(3) + ang_vel_body(3) + rotation_matrix(9) + position_error(3)
        Environment (4): distance_to_wall(1) + wind_estimate(3)
        Task (1): spray_status(1)
        """
        # Root state in world frame
        root_pos_w = self._robot.data.root_pos_w  # (N, 3)
        root_quat_w = self._robot.data.root_quat_w  # (N, 4) [w, x, y, z]
        root_lin_vel_w = self._robot.data.root_lin_vel_w  # (N, 3)
        root_ang_vel_w = self._robot.data.root_ang_vel_w  # (N, 3)

        # Rotation matrix from quaternion (SimpleFlight: rotation matrix > quaternion)
        rot_matrix = _quat_to_rot_matrix(root_quat_w)  # (N, 3, 3)
        rot_matrix_flat = rot_matrix.reshape(self.num_envs, 9)  # (N, 9)

        # Transform velocities to body frame
        rot_matrix_t = rot_matrix.transpose(1, 2)  # (N, 3, 3)
        lin_vel_b = torch.bmm(rot_matrix_t, root_lin_vel_w.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        ang_vel_b = torch.bmm(rot_matrix_t, root_ang_vel_w.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # Position error: target position is (target_wall_distance, 0, initial_altitude)
        target_pos = torch.zeros_like(root_pos_w)
        target_pos[:, 0] = self.cfg.wall_position_x + self.cfg.target_wall_distance
        target_pos[:, 1] = 0.0
        target_pos[:, 2] = 1.5  # target altitude
        pos_error = root_pos_w - target_pos  # (N, 3)

        # Distance to wall
        distance_to_wall = (root_pos_w[:, 0] - self.cfg.wall_position_x).unsqueeze(-1)  # (N, 1)

        # Wind estimate (add noise to simulate imperfect estimation)
        wind_noise = 0.1 * torch.randn_like(self._wind_velocity)
        wind_estimate = self._wind_velocity + wind_noise  # (N, 3)

        # Spray status
        spray_status = self._spray_status.unsqueeze(-1)  # (N, 1)

        # Concatenate: 3 + 3 + 9 + 3 + 1 + 3 + 1 = 23
        obs = torch.cat([
            lin_vel_b,           # 3
            ang_vel_b,           # 3
            rot_matrix_flat,     # 9
            pos_error,           # 3
            distance_to_wall,    # 1
            wind_estimate,       # 3
            spray_status,        # 1
        ], dim=-1)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        """Compute multi-objective reward."""
        root_pos_w = self._robot.data.root_pos_w
        root_ang_vel_w = self._robot.data.root_ang_vel_w

        distance_to_wall = root_pos_w[:, 0] - self.cfg.wall_position_x

        reward = compute_facade_reward(
            distance_to_wall=distance_to_wall,
            target_distance=self.cfg.target_wall_distance,
            angular_velocity=root_ang_vel_w,
            action=self._actions,
            prev_action=self._prev_action,
            spray_status=self._spray_status,
        )

        # Track individual reward components for logging
        distance_error = distance_to_wall - self.cfg.target_wall_distance
        self._episode_sums["distance"] += torch.exp(-2.0 * distance_error**2)
        self._episode_sums["oscillation"] += -0.1 * torch.sum(root_ang_vel_w**2, dim=-1)
        self._episode_sums["smoothness"] += -0.05 * torch.sum(
            (self._actions - self._prev_action) ** 2, dim=-1
        )
        self._episode_sums["energy"] += -0.01 * torch.sum(self._actions**2, dim=-1)
        self._episode_sums["alive"] += self.cfg.rew_alive
        self._episode_sums["spray"] += self._spray_status * 1.5 * torch.exp(-2.0 * distance_error**2)

        # Update previous action
        self._prev_action = self._actions.clone()

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        root_pos_w = self._robot.data.root_pos_w
        distance_to_wall = root_pos_w[:, 0] - self.cfg.wall_position_x
        altitude = root_pos_w[:, 2]

        # Died conditions
        crashed_wall = distance_to_wall < self.cfg.min_wall_distance
        drifted_away = distance_to_wall > self.cfg.max_wall_distance
        too_low = altitude < self.cfg.min_altitude
        too_high = altitude > self.cfg.max_altitude
        died = crashed_wall | drifted_away | too_low | too_high

        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        num_resets = len(env_ids)

        # Log episode metrics before reset
        self.extras["log"] = {}

        for key in self._reward_components:
            avg = torch.mean(self._episode_sums[key][env_ids])
            self.extras["log"][f"Episode_Reward/{key}"] = avg.item() / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Log termination stats
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()

        # Log distance metrics
        root_pos_w = self._robot.data.root_pos_w[env_ids]
        distance_to_wall = root_pos_w[:, 0] - self.cfg.wall_position_x
        self.extras["log"]["Metrics/distance_to_wall_mean"] = distance_to_wall.mean().item()
        self.extras["log"]["Metrics/distance_to_wall_std"] = distance_to_wall.std().item()

        # Reset robot via Isaac Lab API
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Spread out initial episode lengths to avoid reset spikes
        if num_resets == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset to default state with noise
        default_root_state = self._robot.data.default_root_state[env_ids]

        # Add position noise (curriculum-controlled)
        pos_noise = self.cfg.init_pos_noise * (2 * torch.rand(num_resets, 3, device=self.device) - 1)
        default_root_state[:, :3] += pos_noise

        # Ensure minimum wall distance
        default_root_state[:, 0] = default_root_state[:, 0].clamp(
            min=self.cfg.wall_position_x + 0.5
        )

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._wind_velocity[env_ids] = 0.0

        # Randomize spray status
        if self.cfg.spray_enabled:
            self._spray_status[env_ids] = (
                torch.rand(num_resets, device=self.device) > 0.5
            ).float()
        else:
            self._spray_status[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Wind simulation (Ornstein-Uhlenbeck process)
    # ------------------------------------------------------------------

    def _step_wind(self):
        """Update wind velocity using Ornstein-Uhlenbeck process.

        dv = theta * (mu - v) * dt + sigma * sqrt(dt) * N(0,1)
        where mu = 0 (wind direction changes randomly), theta = mean reversion rate
        """
        if self.cfg.wind_max_speed <= 0:
            return

        dt = self.cfg.sim.dt * self.cfg.decimation  # policy timestep
        theta = self.cfg.wind_ou_theta
        sigma = self.cfg.wind_max_speed * 0.3  # noise scale

        noise = sigma * math.sqrt(dt) * torch.randn_like(self._wind_velocity)
        self._wind_velocity += theta * (0.0 - self._wind_velocity) * dt + noise

        # Clamp to max wind speed
        wind_speed = torch.norm(self._wind_velocity, dim=-1, keepdim=True)
        scale = torch.clamp(self.cfg.wind_max_speed / (wind_speed + 1e-8), max=1.0)
        self._wind_velocity *= scale

    # ------------------------------------------------------------------
    # Spray simulation
    # ------------------------------------------------------------------

    def _step_spray(self):
        """Apply spray recoil force when spray is active.

        Spray creates a backward force (negative x in body frame)
        within a cone defined by spray_cone_half_angle.
        """
        if not self.cfg.spray_enabled:
            self._spray_force.zero_()
            return

        # Random force magnitude within [0, spray_force_max]
        force_mag = self.cfg.spray_force_max * torch.rand(self.num_envs, device=self.device)
        force_mag *= self._spray_status  # zero if spray off

        # Random direction within backward cone
        cone_angle = self.cfg.spray_cone_half_angle
        theta = cone_angle * torch.rand(self.num_envs, device=self.device)
        phi = 2 * math.pi * torch.rand(self.num_envs, device=self.device)

        # Spray force in body frame (pointing backward = negative x)
        self._spray_force[:, 0, 0] = -force_mag * torch.cos(theta)
        self._spray_force[:, 0, 1] = force_mag * torch.sin(theta) * torch.cos(phi)
        self._spray_force[:, 0, 2] = force_mag * torch.sin(theta) * torch.sin(phi)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _quat_to_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        quat: Quaternion tensor of shape (N, 4) in [w, x, y, z] format.

    Returns:
        Rotation matrix of shape (N, 3, 3).
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Rotation matrix elements
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    rot = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=1)

    return rot
