"""Multi-objective reward function for facade drone training.

Aligned with PLAN.md reward design:
  r = w1*r_distance + w2*r_oscillation + w3*r_smoothness + w4*r_energy + r_alive + w5*r_spray

References:
  - SimpleFlight (IEEE RA-L 2024): action smoothness penalty is critical for sim-to-real
  - Swift (Nature 2023): Gaussian distance reward for precision hovering
"""

from __future__ import annotations

import torch


def compute_facade_reward(
    distance_to_wall: torch.Tensor,
    target_distance: float,
    angular_velocity: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    spray_status: torch.Tensor,
) -> torch.Tensor:
    """Compute multi-objective reward for facade drone task.

    Args:
        distance_to_wall: Distance from drone to wall surface. Shape: (N,)
        target_distance: Desired wall distance in meters (1.5m).
        angular_velocity: Body-frame angular velocity [wx, wy, wz]. Shape: (N, 3)
        action: Current CTBR action [thrust, wr, wp, wy]. Shape: (N, 4)
        prev_action: Previous step CTBR action. Shape: (N, 4)
        spray_status: Binary spray on/off. Shape: (N,)

    Returns:
        Total reward per environment. Shape: (N,)
    """
    # 1. Distance keeping — Gaussian centered at target distance
    #    r=1.0 at d=target, r≈0.14 at d=target±1.0m
    distance_error = distance_to_wall - target_distance
    r_distance = torch.exp(-2.0 * distance_error**2)

    # 2. Oscillation penalty — minimize body angular rates for stability
    r_oscillation = -0.1 * torch.sum(angular_velocity**2, dim=-1)

    # 3. Action smoothness — penalize jerky commands (SimpleFlight key factor #3)
    r_smoothness = -0.05 * torch.sum((action - prev_action) ** 2, dim=-1)

    # 4. Energy efficiency — penalize excessive control effort
    r_energy = -0.01 * torch.sum(action**2, dim=-1)

    # 5. Alive bonus — reward for not crashing
    r_alive = 0.5

    # 6. Spray compensation — extra reward for stability while spraying
    r_spray = spray_status * 1.5 * r_distance

    total = (
        1.0 * r_distance
        + 1.0 * r_oscillation
        + 1.0 * r_smoothness
        + 1.0 * r_energy
        + r_alive
        + 1.0 * r_spray
    )

    return total
