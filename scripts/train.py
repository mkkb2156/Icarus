"""Training script for Icarus FacadeDrone environment.

Usage:
    # Quick test (verify environment works)
    python scripts/train.py --num_envs 64 --max_iterations 10

    # Full training (headless, ~2-6 hours on RTX 4090)
    python scripts/train.py --num_envs 4096 --headless --max_iterations 5000

    # Resume from checkpoint
    python scripts/train.py --num_envs 4096 --headless --checkpoint logs/latest/nn/last.pth
"""

from __future__ import annotations

import argparse
import os
import sys

# Isaac Lab must be imported before other modules
from isaaclab.app import AppLauncher

# Parse arguments before Isaac Lab initialization
parser = argparse.ArgumentParser(description="Train Icarus FacadeDrone policy with PPO.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--headless", action="store_true", help="Run without rendering.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import Isaac Lab and project modules ---
import gymnasium as gym
from isaaclab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.torch_runner import Runner

# Register our environment
import icarus.training  # noqa: F401 — triggers gym.register()


def main():
    # Create the environment
    env_cfg_entry_point = "icarus.training.facade_drone_env_cfg:FacadeDroneEnvCfg"

    # Import and instantiate config
    from icarus.training.facade_drone_env_cfg import FacadeDroneEnvCfg
    env_cfg = FacadeDroneEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Create the gym environment
    env = gym.make(
        "Icarus-FacadeDrone-Direct-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if not args.headless else None,
    )

    # Wrap for rl_games
    env = RlGamesVecEnvWrapper(env, rl_device="cuda:0", clip_obs=10.0, clip_actions=1.0)

    # Register the environment factory for rl_games
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: env,
        },
    )

    # Load rl_games config
    rl_games_cfg_path = os.path.join(
        os.path.dirname(__file__),
        "..", "icarus", "training", "agents", "rl_games_ppo_cfg.yaml",
    )

    # Create runner
    runner = Runner()
    runner.load({"params": _load_yaml(rl_games_cfg_path)})
    runner.reset()

    # Override settings from CLI
    runner.algo_observer = DefaultAlgoObserver()

    # Train
    runner.run({
        "train": True,
        "play": False,
        "checkpoint": args.checkpoint,
        "sigma": None,
    })

    # Cleanup
    env.close()
    simulation_app.close()


def _load_yaml(path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(os.path.abspath(path)) as f:
        return yaml.safe_load(f)["params"]


if __name__ == "__main__":
    main()
