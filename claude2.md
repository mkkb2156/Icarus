# claude2.md — Icarus PX4 Open-Source Variant

Icarus-PX4 is the open-source deployment variant of the Icarus Embodied AI drone system for autonomous facade operations. The RL training pipeline (Isaac Lab, PPO, 4096 parallel drones) is **100% shared** with the DJI variant (see `CLAUDE.md`). The difference is deployment: PX4 Offboard mode replaces DJI PSDK, Pixhawk replaces M350's internal FC, and ROS 2 replaces the proprietary C bridge. Policy outputs `[thrust, omega_roll, omega_pitch, omega_yaw]` via PX4's `VehicleRatesSetpoint` — the direct equivalent of DJI's angular-rate + thrust joystick mode. Communication uses ROS 2 Humble + Micro XRCE-DDS over Ethernet.

## Architecture

```
icarus/
  training/                        # SHARED with DJI variant (100% reusable)
    facade_drone_env.py            # DirectRLEnv subclass (core environment)
    facade_drone_env_cfg.py        # Environment configuration (@configclass)
    reward.py                      # compute_facade_reward() — 6-objective reward
    curriculum.py                  # 4-stage curriculum (wind/spray/noise ramp)
    domain_randomization.py        # DR config (mass, wind, spray, sensors)
    __init__.py                    # gym.register("Icarus-FacadeDrone-Direct-v0")
    agents/
      rl_games_ppo_cfg.yaml        # PPO hyperparameters for rl_games
  px4/                             # PX4 deployment stack (replaces psdk/)
    offboard_node.py               # ROS 2 node: Offboard mode + heartbeat management
    telemetry_node.py              # ROS 2 subscriber for PX4 uORB topics via XRCE-DDS
    policy_node.py                 # TensorRT inference + obs assembly → rates setpoint
    safety_monitor.py              # Failsafe state machine, geofence, mode transitions
    data_alignment.py              # ROS 2 message_filters for multi-rate sensor sync
    params/
      px4_offboard.yaml            # PX4 parameters for offboard rate control
      safety_limits.yaml           # Geofence, max tilt, max velocity, wall bounds
    launch/
      icarus_px4.launch.py         # ROS 2 launch file for full stack
  inference/                       # SHARED — TensorRT engine loader
    engine.py                      # TensorRT FP16 engine loader & inference
    policy.py                      # RL policy wrapper (obs tensor → action)
scripts/
  train.py                         # SHARED — Training entry point
  play.py                          # SHARED — Inference/visualization
  export_onnx.py                   # SHARED — PyTorch → ONNX export
  px4_sitl_test.py                 # PX4 SITL integration test (Gazebo)
  setup_px4_env.sh                 # One-shot PX4 + ROS 2 + Gazebo setup
demo/                              # SHARED — Streamlit investor demo
models/                            # SHARED — Exported .onnx / .engine (gitignored)
```

Key docs: `CLAUDE.md` (DJI variant spec), `PLAN.md` (full technical spec, zh-TW), `ROADMAP.md` (5-phase product roadmap, zh-TW).

## Key Technical Decisions

### Shared with DJI variant (unchanged)
- **Action space**: `[thrust, omega_roll, omega_pitch, omega_yaw]` — angular rate + thrust (Swift, SimpleFlight, RAPTOR consensus)
- **Observation**: 23-dim with rotation matrix (9 values, not quaternion) per SimpleFlight IEEE RA-L 2024
- **Physics 200Hz / Policy 50Hz** (decimation=4)
- **DirectRLEnv** pattern (Isaac Lab quadcopter example)
- **MLP** `[256, 128, 64]` with ELU activation, ~41K params
- **Target wall distance**: d=1.5m
- **Reward**: 6-objective (distance, oscillation, smoothness, energy, alive, spray)
- **ONNX export** and TensorRT FP16 inference

### PX4-specific decisions
- **Offboard mode** with `VehicleRatesSetpoint` — direct body rate + thrust control, equivalent to DJI's `HORIZONTAL_ANGULAR_RATE` + `VERTICAL_THRUST` mode
- **ROS 2 Humble + Micro XRCE-DDS** over Ethernet — lowest latency path (~5-10ms), preferred over MAVSDK/UART
- **PX4 EKF2** for state estimation — open-source sensor fusion (IMU, GPS/RTK, barometer, magnetometer)
- **Configurable failsafe** — full source access to safety logic (vs DJI black-box safety arbiter)
- **PX4 SITL** (Gazebo Harmonic) for pre-hardware integration testing
- **Advantage over DJI**: Motor mixing is open-source and auditable. Failsafe is configurable. No vendor lock-in. Full flight controller source code available for debugging sim-to-real gaps.

## PX4 Offboard Control Pattern

This is the critical deployment pattern — equivalent of the "Isaac Lab DirectRLEnv Pattern" section in `CLAUDE.md`, but for real-world PX4 deployment.

### Control Mode: Body Rate + Thrust via Offboard

PX4 Offboard mode accepts body-rate + thrust setpoints, which is the direct equivalent of the RL policy's action space:

```
Policy output:  [thrust, omega_roll, omega_pitch, omega_yaw]  (normalized [-1,1])
                           ↓ denormalize
PX4 setpoint:   VehicleRatesSetpoint {
                   roll:  omega_roll  (rad/s, body frame)
                   pitch: omega_pitch (rad/s, body frame)
                   yaw:   omega_yaw  (rad/s, body frame)
                   thrust_body: [0, 0, -thrust]  (normalized [0,1], NED)
                 }
```

### MAVLink: SET_ATTITUDE_TARGET (alternative to ROS 2 direct)

For MAVLink-based implementations (MAVSDK):
- **Message**: `SET_ATTITUDE_TARGET` (#82)
- **type_mask**: `0b10000111` = ignore attitude quaternion, use body rates only
- **body_roll_rate**: rad/s (FRD body frame)
- **body_pitch_rate**: rad/s (FRD body frame)
- **body_yaw_rate**: rad/s (FRD body frame)
- **thrust**: [0, 1] normalized collective thrust

### ROS 2 Topics (preferred path)

Publish to PX4 via XRCE-DDS bridge:
- `/fmu/in/offboard_control_mode` (`OffboardControlMode`) — must publish at **>2Hz** (heartbeat)
  - Set `body_rate = true`, all others `false`
- `/fmu/in/vehicle_rates_setpoint` (`VehicleRatesSetpoint`) — publish at **50-200Hz** (policy rate)
  - `roll`, `pitch`, `yaw` in rad/s (FRD body frame)
  - `thrust_body[2]` = -thrust (NED, negative = up)

### Mode Transition Protocol

1. Start publishing `OffboardControlMode` + `VehicleRatesSetpoint` at >2Hz
2. Wait for stream to stabilize (~1 second)
3. Send `VehicleCommand` to switch to OFFBOARD mode (`MAV_CMD_DO_SET_MODE`)
4. PX4 accepts if setpoint stream is active; rejects otherwise
5. If setpoint stream stops for >`COM_OF_LOSS_T` seconds → failsafe triggers

### Authority Model (vs DJI)

| Aspect | DJI M350 (PSDK) | PX4 (Offboard) |
|--------|-----------------|-----------------|
| Override | RC reclaims on pause/low-battery/geofence/disconnect | RC switch exits Offboard mode instantly |
| Safety logic | Black-box safety arbiter (always on) | Open-source failsafe (fully configurable) |
| Motor mixing | Proprietary, not auditable | Open-source, auditable in `src/modules/mc_att_control` |
| Signal loss | DJI failsafe (RTH/land) | `COM_OF_LOSS_T` timeout → configurable action |
| Rate PID | DJI internal (tuned, not accessible) | PX4 rate controller (can bypass for direct actuator) |

## Hardware Reference

### Recommended Configuration

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Flight Controller** | Holybro Pixhawk 6X | STM32H753, triple-redundant IMU, FMUv6X standard |
| **Companion Computer** | NVIDIA Jetson Orin NX 16GB | 157 TOPS INT8, 100 TOPS sparse, TensorRT native |
| **Carrier Board** | Holybro Jetson Baseboard | Pixhawk + Jetson integration, Ethernet, UART, I2C |
| **FC ↔ Jetson Link** | Ethernet (1Gbps) | XRCE-DDS, ~5-10ms latency |
| **Frame** | 650-900mm X-frame | ~6-8kg AUW with spray payload |
| **Motors** | T-Motor U8II or MN5008 class | Matched to AUW with 2:1 thrust margin |
| **GPS/RTK** | Holybro H-RTK F9P | cm-level RTK positioning |
| **Wall Distance** | TFmini-S LiDAR or Garmin LIDAR-Lite v4 | 0.1-12m range, 100Hz update |
| **RC** | FrSky X9D+ or RadioMaster TX16S | Failsafe override always available |

### Alternative Hardware

- **FC**: CUAV V6X, mRo Pixhawk (FMUv6X compatible)
- **Integrated**: ARK Jetson PAB Carrier (Jetson + Pixhawk on one board)
- **Companion**: Jetson Orin Nano 8GB (budget option, 70 TOPS INT8, sufficient for ~41K param MLP)

## Communication Stack

### Primary: ROS 2 + Micro XRCE-DDS over Ethernet

```
[PX4 Autopilot]                          [Jetson Orin NX]
  uORB topics                              ROS 2 Humble
     ↓                                        ↑
  Micro XRCE-DDS Client  ←── Ethernet ──→  Micro XRCE-DDS Agent
     (on Pixhawk NuttX)     (1Gbps)          (MicroXRCEAgent)
                                               ↓
                                          ROS 2 topics
                                          /fmu/out/* (telemetry)
                                          /fmu/in/*  (commands)
```

### PX4 → Jetson (telemetry subscriptions)

| ROS 2 Topic | PX4 uORB Source | Rate | Content |
|-------------|----------------|------|---------|
| `/fmu/out/vehicle_angular_velocity` | `vehicle_angular_velocity` | 400Hz | Body angular velocity (rad/s) |
| `/fmu/out/vehicle_attitude` | `vehicle_attitude` | 250Hz | Quaternion attitude |
| `/fmu/out/vehicle_local_position` | `vehicle_local_position` | 50Hz | NED position + velocity |
| `/fmu/out/sensor_combined` | `sensor_combined` | 250Hz | IMU accelerometer + gyro |
| `/fmu/out/estimator_status` | `estimator_status` | 50Hz | EKF2 health + innovation |

### Jetson → PX4 (command publishing)

| ROS 2 Topic | Rate | Purpose |
|-------------|------|---------|
| `/fmu/in/offboard_control_mode` | >2Hz | Heartbeat: `body_rate=true` |
| `/fmu/in/vehicle_rates_setpoint` | 50-200Hz | Body rate + thrust from RL policy |
| `/fmu/in/vehicle_command` | On-demand | Mode transitions (arm, offboard, land) |

### Data Flow: Observation Assembly

```
/fmu/out/vehicle_angular_velocity (400Hz) ─┐
/fmu/out/vehicle_attitude (250Hz) ─────────┤
/fmu/out/vehicle_local_position (50Hz) ────┤──→ [data_alignment.py] ──→ ObsTensor(23) ──→ [TensorRT] ──→ [rates_setpoint]
/wall_distance (LiDAR, 100Hz) ─────────────┤                              <1ms                          50-200Hz
/wind_estimate (custom, 10Hz) ─────────────┘
```

Alignment strategy: `message_filters.ApproximateTimeSynchronizer` with 400Hz angular velocity as master. GPS/position interpolated at policy rate. LiDAR held if <100ms stale.

## Safety Architecture

### PX4 Failsafe Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `COM_OF_LOSS_T` | `0.5` | Offboard signal loss timeout (seconds). Triggers failsafe if no setpoint for 0.5s |
| `COM_OBL_RC_ACT` | `0` (RTL) | Action when RC lost during offboard: 0=RTL, 1=Land, 2=Hold |
| `COM_RC_OVERRIDE` | `3` | RC stick override: 3=always allow override in any auto mode |
| `COM_FAIL_ACT_T` | `5.0` | Delay before failsafe action executes |
| `GF_ACTION` | `3` (RTL) | Geofence violation action |
| `GF_MAX_HOR_DIST` | `200` | Max horizontal distance from home (m) |
| `GF_MAX_VER_DIST` | `100` | Max vertical distance from home (m) |

### Application-Level Safety (safety_monitor.py)

In addition to PX4's built-in failsafe, the `safety_monitor.py` ROS 2 node implements:
- **Wall proximity guard**: Emergency brake if distance_to_wall < 0.3m
- **Attitude limit**: Switch to HOLD if tilt > 45 degrees
- **Action rate limiter**: Clamp angular rate commands to safe bounds
- **Watchdog**: If policy_node stops publishing for >100ms, command hover setpoint
- **Battery monitor**: Initiate RTL at configurable voltage threshold
- **EKF2 health check**: Degrade to hover if estimator innovations exceed threshold

### Open-Source Advantage

Unlike DJI's black-box safety arbiter, PX4's safety logic is:
- Fully auditable in `src/modules/commander/failsafe/` and `src/modules/mc_rate_control/`
- Configurable per-parameter via QGroundControl or `param set` commands
- Modifiable for custom safety behaviors (e.g., wall-aware emergency maneuvers)
- Transparent during incident investigation (full flight log with `ulog2csv`)

## Build & Run

```bash
# === PX4 + ROS 2 Environment Setup (Ubuntu 22.04) ===

# 1. Install ROS 2 Humble
sudo apt install ros-humble-desktop ros-humble-ros-gzharmonic

# 2. Install PX4 Autopilot + SITL
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
cd PX4-Autopilot && make px4_sitl gz_x500

# 3. Build Micro XRCE-DDS Agent
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent && mkdir build && cd build
cmake .. && make && sudo make install

# 4. Create ROS 2 workspace
mkdir -p ~/icarus_ws/src && cd ~/icarus_ws/src
git clone https://github.com/PX4/px4_msgs.git
ln -s /path/to/Icarus/icarus/px4 icarus_px4
cd ~/icarus_ws && colcon build

# === SITL Testing (no hardware required) ===

# Terminal 1: PX4 SITL + Gazebo
cd PX4-Autopilot && make px4_sitl gz_x500

# Terminal 2: XRCE-DDS Agent
MicroXRCEAgent udp4 -p 8888

# Terminal 3: Icarus policy node (uses trained ONNX model)
source ~/icarus_ws/install/setup.bash
ros2 launch icarus_px4 icarus_px4.launch.py \
    model_path:=models/facade_policy_fp16.engine \
    use_sim:=true

# === RL Training (same as DJI variant) ===
python scripts/train.py --num_envs 4096 --headless --max_iterations 5000

# === ONNX Export (same as DJI variant) ===
python scripts/export_onnx.py --checkpoint logs/latest/nn/best.pth \
    --output models/facade_policy.onnx

# === TensorRT Engine Build (on Jetson Orin NX) ===
trtexec --onnx=models/facade_policy.onnx \
    --saveEngine=models/facade_policy_fp16.engine \
    --fp16 --workspace=256

# === Real Hardware Deployment ===
# 1. Flash PX4 firmware to Pixhawk 6X
# 2. Set PX4 parameters (see px4/params/px4_offboard.yaml)
# 3. Connect Jetson to Pixhawk via Ethernet
# 4. Run: MicroXRCEAgent udp4 -p 8888
# 5. Run: ros2 launch icarus_px4 icarus_px4.launch.py model_path:=...
# 6. Arm via RC, switch to OFFBOARD mode
```

## PX4 Parameters for Offboard Rate Control

```yaml
# Offboard mode configuration
COM_OF_LOSS_T: 0.5        # Offboard loss timeout (s)
COM_OBL_RC_ACT: 0         # RC loss action: 0=RTL
COM_RC_OVERRIDE: 3        # Allow RC override in auto modes

# Rate controller (may need tuning for custom frame)
MC_ROLLRATE_P: 0.15       # Roll rate P gain
MC_ROLLRATE_I: 0.2        # Roll rate I gain
MC_ROLLRATE_D: 0.003      # Roll rate D gain
MC_PITCHRATE_P: 0.15
MC_PITCHRATE_I: 0.2
MC_PITCHRATE_D: 0.003
MC_YAWRATE_P: 0.2
MC_YAWRATE_I: 0.1

# Note: If RL policy directly outputs actuator commands (future),
# set SYS_CTRL_ALLOC=1 and use VehicleActuatorSetpoint instead

# EKF2 configuration
EKF2_AID_MASK: 1          # GPS fusion
EKF2_HGT_REF: 1           # Barometric height reference
EKF2_MAG_TYPE: 1           # Automatic magnetometer fusion

# XRCE-DDS
XRCE_DDS_CFG: 1000        # Ethernet port for XRCE-DDS

# Safety
GF_ACTION: 3              # Geofence: RTL
GF_MAX_HOR_DIST: 200      # Max horizontal distance (m)
MPC_Z_VEL_MAX_DN: 1.5     # Max descend velocity (m/s)
MPC_Z_VEL_MAX_UP: 3.0     # Max ascend velocity (m/s)
MPC_TILTMAX_AIR: 35       # Max tilt in flight (deg)
```

## DJI vs PX4 Comparison

| Aspect | DJI M350 + PSDK (`CLAUDE.md`) | PX4 + Pixhawk (`claude2.md`) |
|--------|-------------------------------|------------------------------|
| **Training** | Identical (Isaac Lab) | Identical (Isaac Lab) |
| **Policy** | Identical (MLP 41K params) | Identical (MLP 41K params) |
| **Action space** | angular-rate + thrust via PSDK joystick | body rate + thrust via Offboard mode |
| **Compute** | Manifold 3 (Jetson Orin NX) | Jetson Orin NX (same chip) |
| **Inference** | TensorRT FP16 | TensorRT FP16 |
| **Communication** | PSDK C bridge (ctypes) | ROS 2 + XRCE-DDS (native Python) |
| **FC source code** | Proprietary (black box) | Open source (fully auditable) |
| **Safety** | DJI safety arbiter (black box) | PX4 failsafe (configurable, open) |
| **Motor mixing** | Proprietary | Open source |
| **Hardware cost** | ~$15,000 (M350 + Manifold) | ~$3,000-5,000 (Pixhawk + Jetson + frame) |
| **Regulatory** | DJI brand recognition | Requires own safety case |
| **Sim-to-real gap** | Hidden dynamics in DJI stack | Fewer hidden dynamics, better debugging |

## Domain Knowledge

| Term | Meaning |
|------|---------|
| PX4 Autopilot | Open-source flight controller firmware. Runs on Pixhawk hardware (NuttX RTOS). Full source at github.com/PX4/PX4-Autopilot |
| Offboard Mode | PX4 flight mode where an external computer provides setpoints. Equivalent to DJI's PSDK joystick authority. Requires >2Hz heartbeat. |
| uORB | PX4's internal publish-subscribe message bus. Topics are bridged to ROS 2 via Micro XRCE-DDS. |
| Micro XRCE-DDS | DDS-XRCE protocol bridging PX4 uORB topics to ROS 2 DDS. Runs over Ethernet (UDP) or serial. ~5-10ms latency over Ethernet. |
| EKF2 | PX4's Extended Kalman Filter for state estimation. Fuses IMU, GPS, barometer, magnetometer, vision. Open source and tunable. |
| VehicleRatesSetpoint | PX4 uORB message for body rate + thrust commands. Fields: `roll`, `pitch`, `yaw` (rad/s) + `thrust_body[3]` (NED normalized). |
| QGroundControl | Open-source ground control station for PX4. Used for parameter configuration, mission planning, flight log analysis. |
| SITL | Software-In-The-Loop. PX4 simulator using Gazebo Harmonic. Full PX4 firmware runs on desktop — test policies without hardware. |
| Pixhawk 6X | Holybro's FMUv6X standard flight controller. STM32H753, triple-redundant IMU, Ethernet port for XRCE-DDS. |
| MAVLink | Lightweight messaging protocol for drone communication. `SET_ATTITUDE_TARGET` (#82) for body rate + thrust control. |
| FRD | Forward-Right-Down body frame convention used by PX4 and MAVLink. Note: DJI uses FRU (Forward-Right-Up). |
| DR | Domain Randomization — mass ±15%, wind 0-12 m/s, spray 0-15 N. Shared with DJI variant. |
| SimpleFlight | IEEE RA-L 2024 — rotation matrix input + action smoothness penalty are key for sim-to-real |
| Swift | Nature 2023 — champion drone racing with RL, validated angular-rate + thrust action space |
