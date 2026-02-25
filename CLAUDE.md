# CLAUDE.md

Icarus is an Embodied AI drone system for autonomous facade operations (cleaning/inspection). RL policies are trained in NVIDIA Isaac Lab (GPU-parallel simulation), exported to ONNX/TensorRT, and deployed on DJI M350 RTK drones via PSDK CTBR control. Currently in Phase 1 (virtual environment training).

## Architecture

```
icarus/
  training/
    facade_drone_env.py       # DirectRLEnv subclass (core environment)
    facade_drone_env_cfg.py   # Environment configuration (@configclass)
    reward.py                 # compute_facade_reward() — 6-objective reward
    curriculum.py             # 4-stage curriculum (wind/spray/noise ramp)
    domain_randomization.py   # DR config (mass, wind, spray, sensors)
    __init__.py               # gym.register("Icarus-FacadeDrone-Direct-v0")
    agents/
      rl_games_ppo_cfg.yaml   # PPO hyperparameters for rl_games
scripts/
  train.py                   # Training entry point (Isaac Lab + rl_games)
  play.py                    # Inference/visualization of trained policy
  export_onnx.py             # PyTorch checkpoint -> ONNX export
  setup_env.sh               # One-shot environment setup
demo/                        # Streamlit investor demo (no GPU required)
  app.py                     # Main page
  pages/
    1_training_results.py    # Training curves + videos
    2_system_architecture.py # Technical architecture diagrams
    3_nl_mission_planner.py  # NL -> boustrophedon path -> Plotly 3D
models/                      # Exported .onnx / .engine (gitignored)
```

Key docs: `PLAN.md` (full technical spec, zh-TW), `ROADMAP.md` (5-phase product roadmap, zh-TW).

## Key Technical Decisions

- **Action space**: CTBR `[thrust, omega_roll, omega_pitch, omega_yaw]` — consensus best for sim-to-real (Swift, SimpleFlight, RAPTOR)
- **Observation**: 23-dim with **rotation matrix** (9 values, not quaternion) per SimpleFlight IEEE RA-L 2024
- **Physics 200Hz / Policy 50Hz** (decimation=4)
- **DirectRLEnv** pattern (not ManagerBasedRLEnv) — follows Isaac Lab quadcopter example
- **MLP** `[256, 128, 64]` with ELU activation, ~41K params
- **Crazyflie** as proxy robot asset (M350 USD replacement planned for Phase 2)
- **Target wall distance**: d=1.5m

## Isaac Lab DirectRLEnv Pattern

The environment follows the official Isaac Lab quadcopter example exactly:

- Env class inherits `DirectRLEnv`, config inherits `DirectRLEnvCfg`
- **7 required methods**: `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`
- `_get_observations()` returns `{"policy": tensor}` dict
- `_get_dones()` returns `(terminated, truncated)` tuple
- Forces applied via `self._robot.permanent_wrench_composer.set_forces_and_torques()`
- Robot config field name on cfg class: `robot` (not `robot_cfg`)
- Body ID cached via `self._robot.find_bodies("body")[0]`
- Reset pattern: call `self._robot.reset(env_ids)` then `super()._reset_idx(env_ids)`
- Gym registration lives in `icarus/training/__init__.py`

## Code Conventions

- Python >= 3.10; all modules start with `from __future__ import annotations`
- **Ruff** linter, line-length 120 (configured in pyproject.toml)
- `@configclass` for Isaac Lab configs; `@dataclass` for non-Isaac-Lab data structures
- All tensor ops vectorized for N parallel envs — no Python loops over envs
- Type annotations on function signatures
- Module-level docstring states purpose and academic references
- Code and docstrings in **English**; PLAN.md / ROADMAP.md in **zh-TW** (Traditional Chinese)
- Isaac Lab must be imported before project modules (see train.py import order)

## Build & Run

```bash
# Environment setup (Ubuntu 22.04, Python 3.10, NVIDIA RTX GPU required)
./scripts/setup_env.sh

# Quick smoke test
python scripts/train.py --num_envs 64 --max_iterations 10

# Full training (~2-6h on RTX 4090)
python scripts/train.py --num_envs 4096 --headless --max_iterations 5000

# Visualize trained policy
python scripts/play.py --num_envs 32 --checkpoint logs/latest/nn/best.pth

# Export to ONNX
python scripts/export_onnx.py --checkpoint logs/latest/nn/best.pth --output models/facade_policy.onnx

# Investor demo (no GPU needed)
cd demo && pip install -r requirements.txt && streamlit run app.py
```

Install extras: `pip install -e ".[training]"` (Isaac Lab), `pip install -e ".[dev]"` (pytest, ruff), `pip install -e ".[demo]"` (streamlit, plotly, anthropic).

## Domain Knowledge

| Term | Meaning |
|------|---------|
| CTBR | Collective Thrust + Body Rates — lowest-level DJI PSDK control mode, bypasses internal attitude controller |
| PSDK | Payload SDK — DJI's API for on-board computers (Manifold 3) |
| M350 RTK | DJI Matrice 350 RTK — target industrial drone platform (6.47kg, 4 rotors) |
| Manifold 3 | DJI's on-board computer (Jetson Orin NX 16GB), runs TensorRT FP16 inference |
| DR | Domain Randomization — mass ±15%, wind 0-12 m/s, spray 0-15 N |
| SimpleFlight | IEEE RA-L 2024 — rotation matrix input + action smoothness penalty are key for sim-to-real |
| Swift | Nature 2023 — champion drone racing with RL, validated CTBR action space |
