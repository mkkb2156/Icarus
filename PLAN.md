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
[Edge Execution Layer]   (Manifold 3)     → TensorRT FP16 inference < 10ms, PSDK angular-rate + thrust output (DJI safety arbiter retained)
```

---

### Technical Summary

Build a Python-based control software for DJI M350 RTK + Manifold 3 (Jetson Orin) that uses Embodied AI (reinforcement learning via PPO) to replace traditional PID control for autonomous building facade inspection and cleaning at target distance d=1.5m. The system subscribes to PSDK telemetry, runs TensorRT-accelerated RL policy inference, and outputs angular-rate + thrust setpoints via PSDK Virtual Stick joystick interface. DJI's safety arbiter (authority management, motor mixing, failsafe) remains active at all times — the RL policy provides high-frequency outer-loop setpoints, not raw motor access. Angular rate + thrust mode supports up to 400Hz command rate; training uses 50Hz policy (4:1 decimation from 200Hz physics), deployment can scale to 200Hz given ~0.5ms inference latency.

Runtime data pipeline:
```
[PSDK Telemetry] → [Multi-rate Alignment] → [Observation Tensor (23-dim)] → [TensorRT Policy] → [Safety Filter] → [Rate+Thrust Setpoint] → [DJI Safety Arbiter]
     400Hz IMU           50-200Hz fusion                                          <1ms                every cycle       50-200Hz              (always active)
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
- **ActionVector**: `[thrust, omega_roll, omega_pitch, omega_yaw]` → 4-dim angular rate + thrust setpoints
  - PSDK `dji_flight_controller.h` mode combination: `HORIZONTAL_ANGULAR_RATE_CONTROL_MODE` + `VERTICAL_THRUST_CONTROL_MODE` + `YAW_ANGLE_RATE_CONTROL_MODE` in `HORIZONTAL_BODY_COORDINATE` (body/FRU frame)
  - Angular rate + thrust is the consensus best action space for sim-to-real transfer (Swift, SimpleFlight, RAPTOR)
  - **Important nuance**: This is high-frequency outer-loop setpoint control. DJI's safety arbiter (motor mixing, ESC, authority management, failsafe) remains active. The RL policy provides outer-loop + semi-inner-loop setpoints; DJI handles motor-level control and safety enforcement.
  - Phase 2 on-device validation required: (a) confirm mode acceptance, (b) measure actual command rate and latency, (c) characterize authority takeover boundaries
- **FlightCommandInterface** — abstract protocol for platform abstraction (DJI PSDK / MAVLink future)

### Step 1.3 — Configuration (`core/config.py`)
- Target wall distance: `d_target = 1.5m`
- Control frequency: 50Hz default, configurable up to 200Hz (CTBR supports 400Hz max)
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
- Subscribed topics (rates per DJI OSDK documentation):
  - `TOPIC_ANGULAR_RATE_FUSIONED` @ 400Hz (body angular velocity — highest rate, critical for CTBR)
  - `TOPIC_QUATERNION` @ 200Hz (attitude)
  - `TOPIC_VELOCITY` @ 200Hz (NED velocity)
  - `TOPIC_ACCELERATION_BODY` @ 200Hz (body acceleration)
  - `TOPIC_ALTITUDE_BAROMETER` @ 200Hz (barometric altitude)
  - `TOPIC_GPS_FUSED` / `TOPIC_POSITION_FUSED` @ 50Hz (RTK position)
- Thread-safe ring buffers per topic with timestamps
- Provides `get_latest()` and `get_interpolated(timestamp)` accessors

### Step 2.3 — Perception Module (`psdk/perception.py`)
- Subscribe to M350 mmWave radar via `DjiPerception_SubscribeRadarPointcloud`
- Convert raw pointcloud → scalar `distance_to_wall` for the AI observation
- Obstacle map extraction for safety layer
- Fallback: use stereo vision depth if radar unavailable

### Step 2.4 — Data Alignment (`psdk/data_alignment.py`)
**Research Topic #1: Multi-rate sensor synchronization**
- Master clock: angular rate @ 400Hz drives the alignment cycle
- For each control tick:
  1. Take latest IMU samples (angular_vel @ 400Hz, quaternion/accel @ 200Hz) — guaranteed fresh
  2. Look up nearest GPS/RTK sample (50Hz) — linear interpolation if Δt < 20ms, hold if Δt < 40ms, mark stale otherwise
  3. Look up nearest radar/depth sample (10Hz) — hold last valid, mark stale if > 150ms
  4. Look up nearest wind estimate — hold last valid
- Output: timestamped `ObservationTensor` at control frequency
- Implementation: lockless double-buffer pattern for real-time safety
- Staleness flags propagated to safety layer (degrade to hover if critical data stale)

### Step 2.5 — Flight Controller Interface (`psdk/flight_controller.py`)
- **Joystick mode configuration (angular rate + thrust)**:
  - Horizontal: `DJI_FLIGHT_CONTROLLER_HORIZONTAL_ANGULAR_RATE_CONTROL_MODE` (enum=3)
  - Vertical: `DJI_FLIGHT_CONTROLLER_VERTICAL_THRUST_CONTROL_MODE` (enum=2)
  - Yaw: `DJI_FLIGHT_CONTROLLER_YAW_ANGLE_RATE_CONTROL_MODE` (enum=1)
  - Coordinate: `HORIZONTAL_BODY_COORDINATE` (body/FRU frame)
  - Stable mode: **disabled** (`DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_DISABLE`, enum=0)
  - **Note**: Disabling stable mode removes DJI's attitude-level stabilization loop, allowing the RL policy to set angular-rate + thrust setpoints directly. However, DJI's safety arbiter (motor mixing, ESC protection, authority management, failsafe triggers) remains active at all times. This is outer-loop + semi-inner-loop setpoint control, not raw motor access. The retained safety layer is a commercial advantage for regulatory compliance and customer risk assurance.
- `send_command(thrust_pct, omega_roll, omega_pitch, omega_yaw)` → converts to `T_DjiFlightControllerJoystickCommand` → calls `ExecuteJoystickAction`
- **Authority management (critical for deployment)**:
  - Must first obtain joystick authority via `ObtainJoystickCtrlAuthority()`
  - Authority can be forcibly returned to RC under multiple conditions: RC not in P mode, RC pause button, low battery go-home/landing, PSDK disconnection, approaching flight boundaries
  - Design for graceful degradation: monitor authority state, handle takeover events, resume when authority returns
  - Release authority on stop/failsafe
- **Phase 2 validation** (3 concrete test objectives):
  1. **Command rate & latency**: Can we sustain 20-50Hz (or at least 10-20Hz) joystick commands? Measure end-to-end latency from policy output to motor response.
  2. **Mode availability constraints**: GPS/health flags, visual system conditions; in water mist/reflection scenarios, which modes degrade?
  3. **Authority takeover boundaries**: Map all conditions under which RC/low-battery/boundary events seize control. Design graceful degradation protocol.
- Command rate: 50Hz training / up to 200Hz deployment (angular rate + thrust supports 400Hz max per OSDK docs)
- Minimal command smoothing (latency-sensitive; policy learns smooth outputs via reward shaping)

---

## Phase 3: AI Inference Pipeline

### Step 3.1 — TensorRT Engine (`inference/engine.py`)
- Load pre-built `.engine` file (serialized for Jetson Orin)
- Fallback: build from ONNX on first launch, cache to disk
- Pre-allocate CUDA device memory (input: `[1, 23]` float16, output: `[1, 4]` float16)
- Use CUDA streams + pinned host memory for async transfer
- Target latency: <10ms on Orin (FP16 mode)
- Engine rebuild detection: hash ONNX file, rebuild if changed

### Step 3.2 — Policy Wrapper (`inference/policy.py`)
- Observation normalization: running mean/std from training (loaded from checkpoint)
- Input: `ObservationTensor(23)` → normalize → TensorRT inference → `ActionVector(4)`
- Action post-processing (CTBR):
  - `thrust ∈ [0, 100] %` (collective thrust percentage)
  - `ω_roll ∈ [-150, 150] deg/s`, `ω_pitch ∈ [-150, 150] deg/s`, `ω_yaw ∈ [-100, 100] deg/s`
  - Minimal smoothing (CTBR is latency-sensitive; the policy learns smooth outputs via reward shaping)
- Backend abstraction:
  - `TensorRTPolicy` — production (Jetson)
  - `TorchPolicy` — development/debugging
  - `DummyPolicy` — testing (returns hover thrust, zero rates)

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
- Observation space (23-dim): `O = [linear_vel(3), angular_vel(3), rotation_matrix(9), position_error(3), dist_to_wall(1), wind_est(3), spray_status(1)]`
- Action space (4-dim continuous): `a = [thrust, ω_roll, ω_pitch, ω_yaw]` — CTBR (Collective Thrust + Body Rates), the consensus best action space for sim-to-real (Swift, SimpleFlight, RAPTOR)
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
- MLP policy: `[23] → 256 → 128 → 64 → [4]` with ELU activations
- Training on GPU cluster, target: convergence within ~2000 epochs

### Step 5.6 — Model Export (`training/export.py`)
- `torch.onnx.export()` with fixed shapes `[1, 23]` → `[1, 4]`
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
    2. obs_tensor = build_observation(data)    # Construct 23-dim vector
    3. action = policy.infer(obs_tensor)       # TensorRT inference → CTBR [thrust, ωr, ωp, ωy]
    4. safe_action = failsafe.check(action, data)  # Safety filter
    5. flight_controller.send_command(safe_action)  # CTBR → PSDK JoystickCommand
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

## Research Topic #5: PSDK Angular Rate + Thrust Control Mode

**Discovery**: DJI PSDK header `dji_flight_controller.h` defines control mode enums enabling angular-rate + thrust setpoint control on PSDK-connected drones. The recommended mode combination for RL policy deployment:

```c
// Horizontal control modes (T_DjiFlightControllerHorizontalControlMode)
DJI_FLIGHT_CONTROLLER_HORIZONTAL_VELOCITY_CONTROL_MODE       = 0,
DJI_FLIGHT_CONTROLLER_HORIZONTAL_POSITION_CONTROL_MODE       = 1,
DJI_FLIGHT_CONTROLLER_HORIZONTAL_ANGULAR_RATE_CONTROL_MODE   = 3,  // ← body angular rates (±150 deg/s)

// Vertical control modes (T_DjiFlightControllerVerticalControlMode)
DJI_FLIGHT_CONTROLLER_VERTICAL_VELOCITY_CONTROL_MODE         = 0,
DJI_FLIGHT_CONTROLLER_VERTICAL_POSITION_CONTROL_MODE         = 1,
DJI_FLIGHT_CONTROLLER_VERTICAL_THRUST_CONTROL_MODE           = 2,  // ← thrust percentage (0-100%)

// Yaw control modes (T_DjiFlightControllerYawControlMode)
DJI_FLIGHT_CONTROLLER_YAW_ANGLE_CONTROL_MODE                 = 0,
DJI_FLIGHT_CONTROLLER_YAW_ANGLE_RATE_CONTROL_MODE            = 1,  // ← yaw rate (±150 deg/s)

// Coordinate system
DJI_FLIGHT_CONTROLLER_HORIZONTAL_GROUND_COORDINATE            = 0,  // NEU (ground)
DJI_FLIGHT_CONTROLLER_HORIZONTAL_BODY_COORDINATE               = 1,  // FRU (body frame) ← our choice

// Stable mode (T_DjiFlightControllerStableMode)
DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_ENABLE             = 0,  // DJI attitude stabilization ON
DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_DISABLE            = 1,  // DJI attitude stabilization OFF
```

**Corrected understanding (updated based on PSDK header research)**:

The earlier framing "CTBR bypasses DJI's internal attitude controller" was **inaccurate**. The correct 3-layer model:

| Layer | Who controls | What it does |
|-------|-------------|--------------|
| **Outer-loop** (our RL policy) | AI @ 20-50Hz+ | Angular-rate setpoints (roll/pitch/yaw rates) + thrust percentage. With stable mode disabled, DJI's attitude-level PID is out of the loop, but the policy still issues *setpoints* that DJI's lower layers execute. |
| **Safety arbiter** (DJI, always-on) | DJI firmware | Motor mixing (setpoint → individual motor RPM), ESC protection, joystick authority management, failsafe triggers (low battery, geofence, RC pause, PSDK disconnect). Cannot be disabled. |
| **True inner-loop** (DJI, black box) | DJI ESC | Motor mixing matrix, ESC-level control. Not accessible via any API. For commercial operations, this is correct — you should not touch it. |

**Correct framing**: "AI controls outer-loop, DJI keeps inner-loop stability + failsafe"
- RL policy outputs high-frequency angular-rate + thrust setpoints
- DJI executes these setpoints through its motor mixing and ESC layer
- DJI's safety arbiter can seize control at any time (RC takeover, low battery, geofence)
- This retained safety layer is a **commercial advantage**: regulatory compliance, customer risk assurance, operational insurance

**Architecture decision**: Use angular-rate + thrust as sole action space. No velocity fallback in initial implementation.
- Angular rate + thrust is the consensus best action space for sim-to-real transfer (Swift, SimpleFlight, RAPTOR)
- Eliminates the complexity of DJI's velocity-to-attitude mapping (which adds unpredictable intermediate dynamics)
- **Additional advantage**: Supports up to **400Hz** command rate vs velocity's 50Hz (per DJI OSDK docs, Issues [#509](https://github.com/dji-sdk/Onboard-SDK/issues/509), [#456](https://github.com/dji-sdk/Onboard-SDK/issues/456))

**DJI Joystick Authority Model**:
- Must first obtain joystick authority (`ObtainJoystickCtrlAuthority`)
- Authority source is switchable: RC / MSDK / Internal / OSDK
- Authority can be **forcibly returned to RC** under:
  - RC not in P mode
  - RC pause button pressed
  - Low battery triggering go-home or landing
  - PSDK disconnection
  - Approaching DJI flight boundaries
- This is a feature, not a limitation — it provides the operational safety net required for commercial facade operations

**Phase 2 validation** (3 critical test objectives):
1. **Command rate & latency**: Call `SetJoystickMode()` with angular-rate + thrust enums. Confirm acceptance. Measure sustainable command rate (target: 20-50Hz, minimum: 10-20Hz). Measure end-to-end latency from PSDK command to measurable motor response.
2. **Mode availability constraints**: Test under water mist / reflective surface conditions. Determine which GPS/health flags or visual system states affect mode availability. Map the "degraded mode" landscape.
3. **Authority takeover boundaries**: Systematically trigger all takeover conditions (RC pause, low battery, geofence approach). Measure transition timing. Design graceful degradation protocol: detect authority loss → safe hover command → resume on authority return.

If mode is rejected by firmware → implement velocity fallback (training is cheap: ~4hrs on RTX 4090).

**Source**: [DJI Payload-SDK dji_flight_controller.h](https://github.com/dji-sdk/Payload-SDK/blob/master/psdk_lib/include/dji_flight_controller.h)

---

## DJI M350 RTK — Platform Parameters

Physical and performance parameters for Isaac Lab simulation and System Identification. Values marked `[EST]` are analytical estimates requiring Phase 3 SysID validation.

### Physical Specifications

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| Aircraft (no batteries, no payload) | 3.77 | kg | DJI Official |
| Single TB65 battery | ~1.35 | kg | DJI Official |
| Aircraft + 2x TB65 (no payload) | 6.47 | kg | DJI Official |
| Max payload capacity | 2.73 | kg | Derived |
| Max takeoff weight (MTOW) | 9.2 | kg | DJI Official |
| Diagonal wheelbase (motor-to-motor) | 895 | mm | DJI Official |
| Arm length (center to motor) | ~447.5 | mm | Derived |
| Unfolded (no props) L×W×H | 810×670×430 | mm | DJI Official |
| Propeller diameter | 53 (21") | cm | DJI 2110s |
| Motor count | 4 (X-frame) | — | DJI Official |
| Motor mounting | Inverted (props below arms) | — | DJI Official |

### Inertia & Propulsion [EST — Requires Phase 3 SysID]

| Parameter | Symbol | Value | Unit | Confidence |
|-----------|--------|-------|------|------------|
| Roll inertia | Ixx | 0.12 | kg·m² | Low |
| Pitch inertia | Iyy | 0.12 | kg·m² | Low |
| Yaw inertia | Izz | 0.22 | kg·m² | Low |
| Thrust coefficient | kf | 8.5e-4 | N/(rad/s)² | Low |
| Torque coefficient | km | 1.1e-5 | N·m/(rad/s)² | Low |
| km/kf ratio | — | ~0.013 | m | Medium |
| Motor time constant | τ_m | 0.02 | s | Low |
| Max thrust per motor | — | ~45 | N | Medium |
| Total max thrust | — | ~180 | N | Medium |
| Hover thrust fraction | — | ~35% | — | Medium |

### Flight Envelope (DJI Official)

| Parameter | Value | Unit |
|-----------|-------|------|
| Max horizontal speed | 23 (S-mode) / 17 (P-mode) | m/s |
| Max ascent speed | 6 | m/s |
| Max descent speed | 5 | m/s |
| Max tilt angle | 30 | deg |
| Max angular velocity roll/pitch | 300 | deg/s |
| Max angular velocity yaw | 100 | deg/s |
| Max wind resistance | 12 | m/s (Beaufort 6) |
| Max altitude (standard props) | 5,000 | m |
| Max flight time (no payload) | 55 | min |
| Operating temperature | -20 to 50 | °C |
| IP rating | IP55 | — |

### PSDK/OSDK Control & Telemetry Rates

| Control Mode | Max Command Rate | Note |
|--------------|-----------------|------|
| **Angular rate + thrust (CTBR)** | **400 Hz** | **← Our mode** |
| Attitude (angle) control | 200 Hz | — |
| Velocity control | 50 Hz | — |
| Position control | 50 Hz | — |

| Telemetry Topic | Max Rate |
|----------------|----------|
| Angular velocity (body frame) | 400 Hz |
| Quaternion / attitude | 200 Hz |
| Body acceleration | 200 Hz |
| Fused velocity | 200 Hz |
| Barometer altitude | 200 Hz |
| Fused position (Cartesian) | 50 Hz |
| Raw GPS/RTK position | 5 Hz |

### Positioning Accuracy

| Mode | Horizontal | Vertical |
|------|-----------|----------|
| RTK FIX | 1 cm + 1 ppm | 1.5 cm + 1 ppm |
| RTK positioning | ±0.1 m | ±0.1 m |
| GNSS positioning | ±1.5 m | ±0.5 m |
| Vision positioning | ±0.3 m | ±0.1 m |

### Battery (TB65 ×2)

| Parameter | Value |
|-----------|-------|
| Chemistry | Li-ion |
| Nominal voltage | 44.76 V |
| Capacity per battery | 5,880 mAh / 263.2 Wh |
| Total energy (2×TB65) | 526.4 Wh |
| Hot-swap capable | Yes |

### Parameters Requiring SysID (Phase 3)

| Parameter | Method |
|-----------|--------|
| Ixx, Iyy, Izz | Trifilar pendulum or data-driven from flight IMU |
| kf (thrust coefficient) | Hover calibration: kf = mg / (4·ω²_hover) |
| km (torque coefficient) | Yaw step-response analysis |
| Drag coefficients | High-speed flight data regression |
| CG offset | Physical measurement + flight trim analysis |

**Sources**: [DJI M350 RTK Specs](https://enterprise.dji.com/matrice-350-rtk/specs), [DJI OSDK Flight Controller Docs](https://developer.dji.com/onboard-sdk/documentation/guides/component-guide-flight-control.html), [OSDK Issue #509](https://github.com/dji-sdk/Onboard-SDK/issues/509), [OSDK Issue #456](https://github.com/dji-sdk/Onboard-SDK/issues/456), [OSDK Issue #938 (inertia not disclosed)](https://github.com/dji-sdk/Onboard-SDK/issues/938)

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
