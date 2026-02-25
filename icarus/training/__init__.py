"""Icarus training environments for Isaac Lab."""

import gymnasium as gym

# Register the FacadeDrone environment with gymnasium
gym.register(
    id="Icarus-FacadeDrone-Direct-v0",
    entry_point="icarus.training.facade_drone_env:FacadeDroneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "icarus.training.facade_drone_env_cfg:FacadeDroneEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)
