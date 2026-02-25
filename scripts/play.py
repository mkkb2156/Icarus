"""Play (inference) script for trained FacadeDrone policy.

Usage:
    # Visualize trained policy
    python scripts/play.py --num_envs 32 --checkpoint logs/latest/nn/best.pth

    # Record video
    python scripts/play.py --num_envs 16 --checkpoint logs/latest/nn/best.pth --record
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained Icarus FacadeDrone policy.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to visualize.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
parser.add_argument("--record", action="store_true", help="Record video frames.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False  # Force rendering for play mode

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import icarus.training  # noqa: F401


def main():
    from icarus.training.facade_drone_env_cfg import FacadeDroneEnvCfg

    env_cfg = FacadeDroneEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Enable full domain randomization for demo
    env_cfg.wind_max_speed = 12.0
    env_cfg.spray_enabled = True
    env_cfg.spray_force_max = 15.0

    env = gym.make("Icarus-FacadeDrone-Direct-v0", cfg=env_cfg)

    # Load trained policy
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    # Extract the actor network from rl_games checkpoint
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    # Build a simple MLP for inference
    policy = _build_policy(env_cfg.observation_space, env_cfg.action_space).to("cuda:0")
    policy.load_state_dict(model_state, strict=False)
    policy.eval()

    # Run inference loop
    obs, _ = env.reset()
    for step in range(args.num_steps):
        with torch.no_grad():
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(obs_tensor)
            actions = actions.clamp(-1.0, 1.0)

        obs, rewards, terminated, truncated, info = env.step(actions)

        if step % 100 == 0:
            print(f"Step {step}: mean_reward={rewards.mean().item():.3f}")

    env.close()
    simulation_app.close()


def _build_policy(obs_dim: int, act_dim: int) -> torch.nn.Module:
    """Build the MLP policy matching the training architecture."""
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, act_dim),
        torch.nn.Tanh(),  # Output in [-1, 1]
    )


if __name__ == "__main__":
    main()
