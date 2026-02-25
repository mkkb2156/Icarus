# Icarus - Embodied AI Drone Facade Operations System
## Implementation Plan

---

### Strategic Vision: Software-Defined Flight (軟體定義飛行)

**Core thesis**: Use Edge AI + Reinforcement Learning to transform drones from "human-operated tools" into "intelligent agents with muscle memory", bypassing 30 years of precision hardware development through computational intelligence.

**Why AI beats PID**:
- PID is reactive and linear — it can only passively correct. RL policy learns proactive compensation for nonlinear disturbances (spray recoil, building turbulence)
- Joystick abstraction forces 4-axis decomposition. Neural policy optimizes motor thrust in high-dimensional space simultaneously
- Traditional systems add 200ms of stability margin for human reaction time. AI operates at millisecond-level "machine instinct"

**Three differentiators**:
1. **Parametric Decoupling** — Don't model physics explicitly. Abstract the world into parameter vectors, let domain randomization teach the AI to resist disturbances autonomously
2. **Sim-to-Real Zero-Shot Transfer** — 99% virtual training (Isaac Lab, 4096 parallel drones) + 1% real-world System ID calibration. R&D cycle: months, not years
3. **Sell Intelligence, Not Hardware** — The trained model is portable to any AI chip. It gives commodity flight controllers professional-grade stability, breaking the high-end FC monopoly

**Three-layer execution architecture**:
```
[Virtual Evolution Layer] (Cloud/GPU)     → Isaac Lab: 4096 parallel drones, domain randomization, PPO training
[Data Alignment Layer]   (Sim-to-Real)    → 50hrs real flight data calibrates simulator constants
[Edge Execution Layer]   (Manifold 3)     → TensorRT FP16 inference < 10ms, PSDK Virtual Stick output
```

---

### Technical Summary

Build a Python-based control software for DJI M350 RTK + Manifold 3 (Jetson Orin) that uses Embodied AI (reinforcement learning via PPO) to replace traditional PID control for autonomous building facade inspection and cleaning at target distance d=1.5m. The system subscribes to PSDK telemetry, runs TensorRT-accelerated RL policy inference at 25-50Hz, and outputs Virtual Stick commands.

Runtime data pipeline:
```
[PSDK Telemetry] → [Multi-rate Alignment] → [Observation Tensor (20-dim)] → [TensorRT Policy] → [Safety Filter] → [Virtual Stick Command]
     200Hz              50Hz fusion                                              <10ms               every cycle          25-50Hz
```

---

## Project Structure

```
icarus/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py                 # System-wide configuration & constants
│   ├── types.py                  # Data classes: StateVector, EnvVector, TaskVector, ObservationTensor
│   └── logging_utils.py          # Structured logging for flight operations
├── psdk/
│   ├── __init__.py
│   ├── bridge.py                 # C→Python bridge (ctypes wrapper for libdji_psdk.so)
│   ├── telemetry.py              # Async telemetry subscriber (IMU/GPS/radar)
│   ├── perception.py             # Radar pointcloud & depth data subscription
│   ├── flight_controller.py      # Virtual Stick command interface (25-50Hz)
│   └── data_alignment.py         # Multi-rate sensor fusion & timestamp alignment
├── inference/
│   ├── __init__.py
│   ├── engine.py                 # TensorRT engine loader & inference runner
│   └── policy.py                 # RL policy wrapper (observation → action mapping)
├── safety/
│   ├── __init__.py
│   ├── failsafe.py               # Shadow system: boundary checks, emergency override
│   ├── geofence.py               # Wall-distance & operational bounding box
│   └── shadow_test.py            # Shadow testing: compare AI vs human commands
├── training/
│   ├── __init__.py
│   ├── env_config.py             # Isaac Lab environment: wall model, d=1.5m target
│   ├── reward.py                 # Reward function: distance + oscillation + energy
│   ├── domain_randomization.py   # Mass, inertia, motor delay, spray recoil, wind
│   ├── curriculum.py             # Progressive difficulty scheduling
│   └── export.py                 # PyTorch → ONNX → TensorRT export pipeline
├── planning/
│   ├── __init__.py
│   ├── path_planner.py           # Boustrophedon (弓字型) cleaning path generator
│   └── multi_drone.py            # Multi-drone coordination (UDP/ROS2)
├── reporting/
│   ├── __init__.py
│   └── report_generator.py       # Post-mission 3D visual inspection report
├── controller.py                 # Main control loop orchestrator
└── main.py                       # Entry point
tests/
├── __init__.py
├── test_types.py
├── test_data_alignment.py
├── test_reward.py
├── test_failsafe.py
├── test_policy.py
└── test_path_planner.py
pyproject.toml
```

---

## Phase 1: Core Infrastructure & Data Types

### Step 1.1 — Project scaffolding (`pyproject.toml`, `__init__.py` files)
- Python ≥ 3.10
- Dependencies: numpy, scipy, torch, onnx, tensorrt, asyncio/uvloop, pydantic, structlog, pytest

### Step 1.2 — Core data types (`core/types.py`)
Define the parametric control vectors as frozen dataclasses/pydantic models:
- **StateVector** `S`: `[linear_velocity(3), angular_velocity(3), rotation_matrix(9), position_error(3)]` → 18 dims
  - **Note**: SimpleFlight (IEEE RA-L 2024) 研究證實旋轉矩陣比四元數有更好的 sim-to-real 遷移效果
- **EnvironmentVector** `E`: `[distance_to_wall(1), wind_vector_estimate(3)]` → 4 dims
- **TaskVector** `T`: `[spray_status(1)]` → 1 dim
- **ObservationTensor** `O = concat(S, E, T)` → 23-dim float32 vector
- **ActionVector** — dual-mode design based on PSDK capability discovery:
  - **Velocity mode** (safe default): `[v_x, v_y, v_z, omega_yaw]` → 4-dim (maps to Virtual Stick velocity control)
  - **CTBR mode** (optimal sim-to-real): `[thrust, omega_roll, omega_pitch, omega_yaw]` → 4-dim (collective thrust + body rates)
  - Mode selection via `core/config.py` — runtime detection validates M350 hardware support
  - **Rationale**: PSDK `dji_flight_controller.h` defines `DJI_FLIGHT_CONTROLLER_HORIZONTAL_ANGULAR_RATE_CONTROL_MODE = 3` and `DJI_FLIGHT_CONTROLLER_VERTICAL_THRUST_CONTROL_MODE = 2`, indicating CTBR may be available. Phase 2 on-device testing will confirm M350 support; if unavailable, system falls back to velocity mode automatically.
- **FlightCommandInterface** — abstract protocol for platform abstraction (DJI PSDK / MAVLink future), supports both velocity and CTBR action spaces

### Step 1.3 — Configuration (`core/config.py`)
- Target wall distance: `d_target = 1.5m`
- Control frequency: 50Hz (configurable 25-50Hz)
- Safety thresholds: min wall distance, max tilt, max velocity, geofence box
- PSDK topic frequencies
- TensorRT engine path, model input/output dimensions

### Step 1.4 — Logging (`core/logging_utils.py`)
- Structured flight telemetry logging (structlog)
- Binary flight data recorder for post-analysis

---

## Phase 2: PSDK Data Layer

### Step 2.1 — C Bridge (`psdk/bridge.py`)
ctypes-based wrapper for `libdji_psdk.so`:
- **Subscription API**:
  - `DjiFcSubscription_Init()`
  - `DjiFcSubscription_SubscribeTopic(topic, frequency, callback)`
  - `DjiFcSubscription_GetLatestValueOfTopic(topic)`
- **Flight Controller API**:
  - `DjiFlightController_Init()`
  - `DjiFlightController_ObtainJoystickCtrlAuthority()`
  - `DjiFlightController_ReleaseJoystickCtrlAuthority()`
  - `DjiFlightController_SetJoystickMode(mode)`
  - `DjiFlightController_ExecuteJoystickAction(command)`
- **Perception API**:
  - `DjiPerception_Init()`
  - `DjiPerception_SubscribeRadarPointcloud(callback)` — mmWave radar data
- Python ctypes struct definitions matching PSDK C headers:
  - `T_DjiFcSubscriptionQuaternion` — attitude quaternion
  - `T_DjiFcSubscriptionVelocity` — NED velocity
  - `T_DjiFcSubscriptionPositionFused` — RTK fused lat/lon/alt
  - `T_DjiFcSubscriptionAccelerationBody` — body-frame acceleration
  - `T_DjiFlightControllerJoystickMode` — control mode config
  - `T_DjiFlightControllerJoystickCommand` — roll/pitch/yaw/throttle

### Step 2.2 — Telemetry Subscriber (`psdk/telemetry.py`)
- Async subscription manager running in dedicated thread
- Subscribed topics:
  - `TOPIC_QUATERNION` @ 200Hz (attitude)
  - `TOPIC_VELOCITY` @ 200Hz (NED velocity)
  - `TOPIC_GPS_FUSED` / `TOPIC_POSITION_FUSED` @ 50Hz (RTK position)
  - `TOPIC_ACCELERATION_BODY` @ 200Hz (body acceleration)
  - `TOPIC_ANGULAR_RATE_FUSIONED` @ 200Hz (angular velocity)
- Thread-safe ring buffers per topic with timestamps
- Provides `get_latest()` and `get_interpolated(timestamp)` accessors

### Step 2.3 — Perception Module (`psdk/perception.py`)
- Subscribe to M350 mmWave radar via `DjiPerception_SubscribeRadarPointcloud`
- Convert raw pointcloud → scalar `distance_to_wall` for the AI observation
- Obstacle map extraction for safety layer
- Fallback: use stereo vision depth if radar unavailable

### Step 2.4 — Data Alignment (`psdk/data_alignment.py`)
**Research Topic #1: Multi-rate sensor synchronization**
- Master clock: IMU @ 200Hz drives the alignment cycle
- For each control tick:
  1. Take latest IMU sample (quaternion, angular_vel, accel) — guaranteed fresh
  2. Look up nearest GPS/RTK sample (50Hz) — linear interpolation if Δt < 20ms, hold if Δt < 40ms, mark stale otherwise
  3. Look up nearest radar/depth sample (10Hz) — hold last valid, mark stale if > 150ms
  4. Look up nearest wind estimate — hold last valid
- Output: timestamped `ObservationTensor` at control frequency
- Implementation: lockless double-buffer pattern for real-time safety
- Staleness flags propagated to safety layer (degrade to hover if critical data stale)

### Step 2.5 — Flight Controller Interface (`psdk/flight_controller.py`)
- **Dual joystick mode configuration** (selected via `core/config.py`):

  **Mode A — Velocity (safe default):**
  - Horizontal: `DJI_FLIGHT_CONTROLLER_HORIZONTAL_VELOCITY_CONTROL_MODE` (body frame)
  - Vertical: `DJI_FLIGHT_CONTROLLER_VERTICAL_VELOCITY_CONTROL_MODE`
  - Yaw: `DJI_FLIGHT_CONTROLLER_YAW_ANGLE_RATE_CONTROL_MODE`
  - Coordinate: **body-fixed** frame
  - Stable mode: **enabled** (`DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_ENABLE`)
  - Action mapping: `send_command(v_x, v_y, v_z, omega_yaw)`

  **Mode B — CTBR (optimal sim-to-real, requires Phase 2 hardware validation):**
  - Horizontal: `DJI_FLIGHT_CONTROLLER_HORIZONTAL_ANGULAR_RATE_CONTROL_MODE` (enum=3)
  - Vertical: `DJI_FLIGHT_CONTROLLER_VERTICAL_THRUST_CONTROL_MODE` (enum=2)
  - Yaw: `DJI_FLIGHT_CONTROLLER_YAW_ANGLE_RATE_CONTROL_MODE` (enum=1)
  - Coordinate: **body-fixed** frame
  - Stable mode: **disabled** (`DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_DISABLE`, enum=0)
  - Action mapping: `send_command(thrust_pct, omega_roll, omega_pitch, omega_yaw)`
  - **Note**: Disabling stable mode removes DJI's internal attitude stabilization — the RL policy must provide full attitude control

- **Runtime mode detection**: On initialization, attempt to set CTBR mode and verify via readback. If M350 firmware rejects the mode, log warning and fall back to velocity mode.
- Converts `ActionVector` to `T_DjiFlightControllerJoystickCommand` → calls `ExecuteJoystickAction`
- Command rate: 25-50Hz (configurable)
- Command smoothing: exponential moving average filter (velocity mode); minimal smoothing for CTBR (latency-sensitive)
- Authority management: obtain on start, release on stop/failsafe

---

## Phase 3: AI Inference Pipeline

### Step 3.1 — TensorRT Engine (`inference/engine.py`)
- Load pre-built `.engine` file (serialized for Jetson Orin)
- Fallback: build from ONNX on first launch, cache to disk
- Pre-allocate CUDA device memory (input: `[1, 20]` float16, output: `[1, 4]` float16)
- Use CUDA streams + pinned host memory for async transfer
- Target latency: <10ms on Orin (FP16 mode)
- Engine rebuild detection: hash ONNX file, rebuild if changed

### Step 3.2 — Policy Wrapper (`inference/policy.py`)
- Observation normalization: running mean/std from training (loaded from checkpoint)
- Input: `ObservationTensor(23)` → normalize → TensorRT inference → `ActionVector(4)`
- **Action post-processing (mode-dependent)**:

  **Velocity mode limits:**
  - `v_x ∈ [-3, 3] m/s`, `v_y ∈ [-3, 3] m/s`, `v_z ∈ [-2, 2] m/s`, `ω_yaw ∈ [-60, 60] deg/s`
  - Apply exponential smoothing between consecutive actions

  **CTBR mode limits:**
  - `thrust ∈ [0, 100] %` (collective thrust percentage)
  - `ω_roll ∈ [-150, 150] deg/s`, `ω_pitch ∈ [-150, 150] deg/s`, `ω_yaw ∈ [-100, 100] deg/s`
  - Minimal smoothing only (CTBR is latency-sensitive; the policy must learn smooth outputs via reward shaping)

- Backend abstraction:
  - `TensorRTPolicy` — production (Jetson)
  - `TorchPolicy` — development/debugging
  - `DummyPolicy` — testing (returns zero commands)
- **Note**: Velocity and CTBR policies are separate trained models with different action semantics. The config specifies which model to load.

---

## Phase 4: Safety & Fail-safe Layer

### Step 4.1 — Fail-safe Shadow System (`safety/failsafe.py`)
**Research Topic #3: Safety interception layer**
- Executes EVERY control cycle, AFTER policy inference, BEFORE command dispatch
- State machine: `NORMAL → CAUTION → OVERRIDE → EMERGENCY`
- Check cascade:

| Check | Threshold | Action |
|-------|-----------|--------|
| Wall proximity | d < 0.5m | OVERRIDE: full reverse thrust |
| Wall proximity | d < 0.8m | CAUTION: clamp forward velocity to 0 |
| Attitude limit | roll/pitch > 25° | OVERRIDE: zero AI commands, hold attitude |
| Velocity limit | |v| > 5 m/s | Clamp to limit |
| Command jerk | |Δcmd| > threshold | Smooth with rate limiter |
| Data staleness | IMU > 50ms stale | EMERGENCY: hover + release authority |
| Geofence breach | Outside ops box | OVERRIDE: return to center |
| Altitude limit | h < 5m or h > 150m | Clamp vertical velocity |

- Override mechanism: release PSDK joystick authority → DJI native obstacle avoidance takes over
- Emergency: trigger `DjiFlightController_ExecuteEmergencyBrakeAction()` or RTK return-to-home
- All overrides logged with full telemetry context

### Step 4.2 — Shadow Testing Mode (`safety/shadow_test.py`)
**Phase 3 from spec: Sim-to-Real validation**
- Mode where human pilot flies manually
- AI computes commands in background (no execution)
- Records: `(timestamp, human_command, ai_command, observation)`
- Post-flight analysis: compute divergence metrics
- If divergence > threshold → flag for reward function tuning

### Step 4.3 — Geofence (`safety/geofence.py`)
- 3D operational bounding box defined relative to building face
- Configurable per-mission in config
- Soft boundary (warning) + hard boundary (override)

---

## Phase 5: RL Training Environment (Isaac Lab)

### Step 5.1 — Environment Configuration (`training/env_config.py`)
- Isaac Lab `DirectRLEnv` subclass: `FacadeDroneEnv`
- 4096 parallel environments (GPU-accelerated PhysX)
- Wall model: flat vertical surface, configurable dimensions
- Target following distance: `d = 1.5m` from wall
- Observation space (20-dim): `O = [dist_to_wall, velocity(3), acceleration(3), orientation(4), angular_vel(3), wind_est(3), spray_status(1), target_offset(2)]`
- Action space (4-dim continuous, mode-dependent):
  - **Velocity mode**: `a = [v_x, v_y, v_z, ω_yaw]` — maps to Virtual Stick velocity control
  - **CTBR mode**: `a = [thrust, ω_roll, ω_pitch, ω_yaw]` — maps to collective thrust + body angular rates (preferred for sim-to-real per Swift/SimpleFlight research; requires Phase 2 M350 hardware validation)
- Physics step: 200Hz, policy step: 50Hz (4:1 decimation ratio)
- Episode termination: crash (d < 0.1m), drift (d > 5m), timeout (60s)

### Step 5.2 — Reward Function (`training/reward.py`)
**Research Topic #2: RL reward design**
```python
def compute_reward(obs, action, prev_action):
    # Distance keeping: Gaussian around target d=1.5m
    r_distance = exp(-alpha * (d_actual - d_target)^2)   # alpha=2.0

    # Oscillation penalty: minimize body angular rates
    r_oscillation = -beta * (omega_x^2 + omega_y^2 + omega_z^2)  # beta=0.1

    # Energy efficiency: penalize excessive control effort
    r_energy = -gamma * sum(action^2)                     # gamma=0.01

    # Smoothness: penalize jerky commands
    r_smoothness = -delta * sum((action - prev_action)^2) # delta=0.05

    # Alive bonus: reward for not crashing
    r_alive = 0.5

    # Spray compensation bonus: extra reward for stability when spraying
    r_spray = spray_status * kappa * r_distance           # kappa=1.5

    return w1*r_distance + w2*r_oscillation + w3*r_energy + w4*r_smoothness + r_alive + w5*r_spray
```

### Step 5.3 — Domain Randomization (`training/domain_randomization.py`)
Randomized per-environment per-episode:
- **Mass**: base ± 15% (simulates water tank fill level changes)
- **Inertia tensor**: ± 10%
- **Motor response delay**: 10-50ms random
- **Motor thrust coefficient**: ± 10%
- **Spray recoil force**: when `spray_status=1`, apply random force vector 0-15N within rear cone, forcing AI to learn forward-lean compensation
- **Wind**: 0-12 m/s, Ornstein-Uhlenbeck process for temporal correlation (realistic gusts)
- **Sensor noise**: Gaussian on IMU (σ_accel=0.1, σ_gyro=0.01), quantization noise on depth (±2cm)
- **Wall texture/distance offset**: ± 0.2m starting position variation

### Step 5.4 — Curriculum Learning (`training/curriculum.py`)
Progressive difficulty schedule:
1. Epoch 0-500: No wind, no spray, tight initial position
2. Epoch 500-1500: Mild wind (0-3 m/s), no spray
3. Epoch 1500-3000: Medium wind (0-6 m/s), spray activated (low recoil)
4. Epoch 3000+: Full randomization (0-12 m/s wind, full spray recoil)

### Step 5.5 — Training Algorithm
- **PPO** (Proximal Policy Optimization) via `rl_games` or `skrl`
- Hyperparameters: lr=3e-4, clip=0.2, entropy_coef=0.01, GAE λ=0.95, γ=0.99
- MLP policy: `[20] → 256 → 128 → 64 → [4]` with ELU activations
- Training on GPU cluster, target: convergence within ~2000 epochs

### Step 5.6 — Model Export (`training/export.py`)
- `torch.onnx.export()` with fixed shapes `[1, 20]` → `[1, 4]`
- ONNX simplification via `onnxsim`
- TensorRT engine build: FP16, fixed batch=1, optimized for Orin
- Validation: PyTorch vs ONNX vs TensorRT output comparison (max_diff < 1e-3)

---

## Phase 6: Main Control Loop & Integration

### Step 6.1 — Controller (`controller.py`)
Main orchestrator running at 50Hz:
```
while running:
    1. telemetry.get_aligned_observation()    # Read sensor data
    2. obs_tensor = build_observation(data)    # Construct 20-dim vector
    3. action = policy.infer(obs_tensor)       # TensorRT inference
    4. safe_action = failsafe.check(action, data)  # Safety filter
    5. flight_controller.send_command(safe_action)  # Virtual Stick output
    6. logger.record(data, action, safe_action)     # Flight data recording
```
- Deadline monitoring: if inference exceeds 20ms, hold last safe command
- Graceful shutdown: release joystick authority → hover → land
- Mode switch: SHADOW_TEST / AUTONOMOUS / MANUAL_OVERRIDE

### Step 6.2 — Entry Point (`main.py`)
- CLI argument parsing (mode, config file, model path)
- Initialize PSDK bridge → telemetry → inference engine → safety → controller
- Signal handling (SIGINT/SIGTERM → graceful shutdown)
- Health monitoring thread

---

## Phase 7: Production Features (Future)

### Step 7.1 — Path Planner (`planning/path_planner.py`)
- Boustrophedon (弓字型) path generation from 3D wall model/pointcloud
- Configurable strip width, overlap percentage
- Integration with mission waypoints

### Step 7.2 — Multi-Drone Coordination (`planning/multi_drone.py`)
- UDP/ROS2 inter-drone position sharing
- Dynamic collision avoidance between multiple M350s
- Task partitioning: divide wall into zones

### Step 7.3 — Report Generator (`reporting/report_generator.py`)
- Post-mission 3D visualization of cleaned areas
- AI-detected residual contamination zones
- Coverage metrics and flight statistics

---

## Research Topic #4: Platform Portability (MAVLink/PX4)

**Goal**: Standardize the control interface so the trained AI model can be deployed on non-DJI platforms.

**Strategy**:
- `FlightCommandInterface` protocol in `core/types.py` defines the abstract contract
- DJI PSDK is the primary implementation (`psdk/flight_controller.py`)
- Future MAVLink implementation maps the same `ActionVector` to MAVLink `SET_ATTITUDE_TARGET` or `SET_POSITION_TARGET_LOCAL_NED`
- Observation tensor structure is platform-agnostic (SI units: m/s, rad/s, quaternion)
- The trained RL model never touches platform-specific APIs — it only sees `ObservationTensor` and outputs `ActionVector`

This ensures the "intelligence" is decoupled from the "airframe", enabling the vision of selling portable flight AI.

---

## Research Topic #5: PSDK CTBR Control Mode Discovery

**Discovery**: DJI PSDK header `dji_flight_controller.h` defines control mode enums that suggest CTBR (Collective Thrust + Body Rates) may be available on M350 RTK:

```c
// Horizontal control modes (T_DjiFlightControllerHorizontalControlMode)
DJI_FLIGHT_CONTROLLER_HORIZONTAL_VELOCITY_CONTROL_MODE       = 0,  // velocity (current implementation)
DJI_FLIGHT_CONTROLLER_HORIZONTAL_POSITION_CONTROL_MODE       = 1,  // position
DJI_FLIGHT_CONTROLLER_HORIZONTAL_ANGULAR_RATE_CONTROL_MODE   = 3,  // body angular rates ← CTBR

// Vertical control modes (T_DjiFlightControllerVerticalControlMode)
DJI_FLIGHT_CONTROLLER_VERTICAL_VELOCITY_CONTROL_MODE         = 0,  // velocity (current implementation)
DJI_FLIGHT_CONTROLLER_VERTICAL_POSITION_CONTROL_MODE         = 1,  // position/altitude
DJI_FLIGHT_CONTROLLER_VERTICAL_THRUST_CONTROL_MODE           = 2,  // thrust percentage ← CTBR

// Stable mode (T_DjiFlightControllerStableMode)
DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_ENABLE             = 0,  // DJI internal stabilization ON
DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_DISABLE            = 1,  // DJI internal stabilization OFF ← required for CTBR
```

**Significance**: If CTBR is supported on M350, this fundamentally changes our control architecture:
- **Velocity mode** = outer-loop controller on top of DJI black-box attitude controller (current design)
- **CTBR mode** = direct body-rate + thrust control, bypassing DJI's internal controller — this is the preferred action space for sim-to-real transfer (per Swift, SimpleFlight, RAPTOR research)

**Validation plan** (Phase 2 priority):
1. On Manifold 3, call `DjiFlightController_SetJoystickMode()` with CTBR enum values
2. Check return code — `DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS` confirms support
3. If supported: train both velocity and CTBR policies, compare sim-to-real transfer quality
4. If rejected by firmware: fall back to velocity mode (no wasted effort — velocity pipeline already works)

**Impact on training**: If CTBR confirmed, train two parallel policies in Isaac Lab:
- Velocity policy: `[v_x, v_y, v_z, ω_yaw]` — safe fallback, always available
- CTBR policy: `[thrust, ω_roll, ω_pitch, ω_yaw]` — expected to have better sim-to-real transfer, lower tracking error

**Source**: [DJI Payload-SDK dji_flight_controller.h](https://github.com/dji-sdk/Payload-SDK/blob/master/psdk_lib/include/dji_flight_controller.h)

---

## Implementation Priority (What to build now)

**Immediate (this session) — 14 files:**
1. `core/types.py` — Foundation data structures & FlightCommandInterface protocol
2. `core/config.py` — Configuration management (pydantic)
3. `core/logging_utils.py` — Structured flight data logging
4. `psdk/bridge.py` — PSDK ctypes bridge (C function wrappers + struct definitions)
5. `psdk/telemetry.py` — Async telemetry subscription with ring buffers
6. `psdk/perception.py` — Radar/depth data subscription
7. `psdk/data_alignment.py` — Multi-rate sensor fusion (Research Topic #1)
8. `psdk/flight_controller.py` — Virtual Stick command output
9. `inference/engine.py` — TensorRT engine wrapper
10. `inference/policy.py` — Policy interface with TensorRT/Torch/Dummy backends
11. `safety/failsafe.py` — Safety shadow system (Research Topic #3)
12. `safety/geofence.py` — 3D operational bounding box
13. `training/reward.py` — RL reward function (Research Topic #2)
14. `training/domain_randomization.py` — Domain randomization config
15. `controller.py` — Main 50Hz control loop
16. `main.py` — Entry point with CLI + graceful shutdown
17. Unit tests for types, data alignment, reward, failsafe, policy

**Deferred:**
- `training/env_config.py`, `training/curriculum.py`, `training/export.py` — Isaac Lab environment (requires Isaac Sim)
- `safety/shadow_test.py` — Shadow testing (requires real flight data)
- `planning/`, `reporting/` — Phase 7 production features
- MAVLink/PX4 `FlightCommandInterface` implementation
