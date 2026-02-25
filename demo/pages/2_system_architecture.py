"""System Architecture page — technical deep-dive for investors."""

import streamlit as st

st.set_page_config(page_title="System Architecture — Icarus", layout="wide")

st.title("System Architecture")
st.markdown("Technical deep-dive into the Icarus Embodied AI drone control system.")

st.markdown("---")

# ─── Three-Layer Architecture ─────────────────────────────────────────

st.header("Three-Layer Execution Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1. Virtual Evolution
    **Platform**: NVIDIA Isaac Lab (GPU cluster)

    - **4,096 parallel drones** training simultaneously
    - 200Hz physics simulation, 50Hz policy
    - Domain randomization:
      - Wind: 0-12 m/s (Ornstein-Uhlenbeck)
      - Spray recoil: 0-15N
      - Mass: ±15% (water tank level)
    - Curriculum learning (4 stages)
    - PPO training → convergence in 2-6 hours
    """)

with col2:
    st.markdown("""
    ### 2. Data Alignment
    **Method**: System Identification

    - 3 simple flight maneuvers (~1 min)
    - Extract: mass, inertia, thrust curves
    - Calibrate simulator to reality
    - Narrow DR to ±10% around true values
    - Retrain policy v2 with calibrated sim
    - Shadow testing: AI computes, human flies
    """)

with col3:
    st.markdown("""
    ### 3. Edge Execution
    **Hardware**: DJI Manifold 3 (Jetson Orin)

    - TensorRT FP16 inference: **<1ms**
    - PSDK angular rate + thrust: up to **400Hz**
    - Multi-rate sensor fusion:
      - IMU: 400Hz
      - Attitude: 200Hz
      - GPS/RTK: 50Hz
    - Safety failsafe state machine
    - Hot-swappable battery operation
    """)

st.markdown("---")

# ─── Runtime Data Pipeline ────────────────────────────────────────────

st.header("Runtime Data Pipeline")

st.code("""
[PSDK Telemetry] → [Multi-rate Alignment] → [Observation (23-dim)] → [TensorRT Policy] → [Safety Filter] → [Rate+Thrust Setpoint] → [DJI Safety Arbiter]
     400Hz IMU         50-200Hz fusion                                       <1ms              every cycle       50-200Hz              (always active)
""", language="text")

st.markdown("---")

# ─── DJI M350 RTK Platform ───────────────────────────────────────────

st.header("Platform: DJI Matrice 350 RTK")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Physical Specs

    | Parameter | Value |
    |-----------|-------|
    | Mass (flight-ready) | 6.47 kg |
    | Max takeoff weight | 9.2 kg |
    | Max payload | 2.73 kg |
    | Diagonal wheelbase | 895 mm |
    | Propeller diameter | 53 cm (21") |
    | IP rating | IP55 |
    """)

with col2:
    st.markdown("""
    ### Flight Performance

    | Parameter | Value |
    |-----------|-------|
    | Max horizontal speed | 23 m/s |
    | Max wind resistance | 12 m/s (Beaufort 6) |
    | Max tilt angle | 30° |
    | Max angular rate (roll/pitch) | 300°/s |
    | Max flight time | 55 min (no payload) |
    | RTK positioning | ±0.1m H / ±0.1m V |
    """)

st.markdown("---")

# ─── Angular Rate + Thrust Control ───────────────────────────────────

st.header("Angular Rate + Thrust: High-Frequency Outer-Loop Control")

st.markdown("""
**Why angular rate + thrust over traditional velocity/position control?**

| Feature | Velocity Control | Angular Rate + Thrust (Our Approach) |
|---------|-----------------|--------------------------------------|
| Command rate | 50 Hz | **Up to 400 Hz** |
| DJI attitude PID | Active (adds hidden dynamics) | **Disabled** (policy sets rate targets directly) |
| DJI safety arbiter | Active | **Active** (motor mixing, failsafe, authority — retained as commercial safety feature) |
| Disturbance handling | Reactive (DJI PID response) | **Proactive** (learned compensation) |
| Sim-to-real gap | Large (hidden intermediate dynamics) | **Reduced** (fewer abstraction layers between policy and motors) |

The RL policy outputs `[thrust, ω_roll, ω_pitch, ω_yaw]` as setpoints — the same
representation used by championship-winning drone racing AI (Swift, Nature 2023).
DJI's safety layer (motor mixing, failsafe, RC authority) remains active at all times,
providing the operational safety net required for commercial facade operations.
""")

st.markdown("---")

# ─── Research Foundation ──────────────────────────────────────────────

st.header("Research Foundation")

st.markdown("""
Icarus is built on peer-reviewed, state-of-the-art research:

| Reference | Source | Key Contribution |
|-----------|--------|-----------------|
| **Swift** | UZH, *Nature* 2023 | Champion drone racing AI — angular rate + thrust action space, non-parametric noise model |
| **SimpleFlight** | Tsinghua, *IEEE RA-L* 2024 | 5 key factors for zero-shot sim-to-real: rotation matrix input, action smoothness penalty |
| **RAPTOR** | rl-tools, 2025 | 2,084-param GRU controlling 10 different drones — meta-imitation learning |
| **Aerial Gym** | NTNU, *IEEE RA-L* 2025 | GPU-parallel MAV simulation — Isaac Gym architecture template |
| **Isaac Lab** | NVIDIA | Official quadcopter RL environment — DirectRLEnv pattern |
""")

st.markdown("---")

# ─── Roadmap ──────────────────────────────────────────────────────────

st.header("Product Roadmap")

phases = [
    ("Phase 0", "Environment Setup", "2-3 weeks", "Completed", "Isaac Lab validated, baseline trained"),
    ("Phase 1", "Virtual Training", "6-8 weeks", "Completed", "4096-env training, ONNX export, demo dashboard"),
    ("Phase 2", "PSDK Integration", "4-6 weeks", "Next", "Manifold 3 inference, angular rate + thrust validation, safety system"),
    ("Phase 3", "Sim-to-Real", "4-6 weeks", "Planned", "System ID, shadow testing, policy v2"),
    ("Phase 4", "Autonomous Flight", "4-6 weeks", "Planned", "AI maintains d=1.5m for 5+ min, spray mode"),
    ("Phase 5", "Product Features", "8-12 weeks", "Planned", "Path planning, multi-drone, mission reporting"),
]

for phase, name, duration, status, description in phases:
    col1, col2, col3, col4 = st.columns([1, 2, 1, 4])
    with col1:
        st.markdown(f"**{phase}**")
    with col2:
        st.markdown(f"**{name}**")
    with col3:
        if status == "Completed":
            st.success(status)
        elif status == "Next":
            st.info(status)
        else:
            st.caption(status)
    with col4:
        st.caption(f"{duration} — {description}")
