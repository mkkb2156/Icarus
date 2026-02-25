"""Icarus Demo — Investor Presentation App.

Launch:
    cd demo && streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Icarus — Embodied AI Drone System",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Icarus")
st.subheader("Embodied AI Drone Facade Operations System")

st.markdown("---")

# Hero section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### The Problem

    High-rise facade cleaning and inspection is a **$XX billion global market** plagued by:
    - **Human risk**: Workers suspended at 100m+ heights
    - **Low efficiency**: Manual operation, weather-dependent downtime
    - **Inconsistent quality**: Human fatigue → missed defects

    ### Our Solution

    **Icarus** uses **Embodied AI** to give drones "muscle memory" —
    trained in simulation across **millions of virtual flights**, then deployed
    on real DJI M350 RTK drones with **<1ms reaction time**.

    The AI learns to maintain precise wall distance (1.5m) even under
    12 m/s wind and spray recoil forces — something traditional PID controllers cannot achieve.
    """)

with col2:
    st.markdown("""
    ### Key Numbers

    | Metric | Value |
    |--------|-------|
    | Parallel training envs | **4,096** |
    | Model parameters | **41K** |
    | Inference latency | **<1ms** |
    | Wind resistance | **12 m/s** |
    | Training time | **2-6 hrs** |
    | Target platform | **DJI M350 RTK** |
    """)

st.markdown("---")

# Architecture overview
st.markdown("""
### Three-Layer Execution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Virtual Evolution Layer  (Cloud/GPU)                               │
│  Isaac Lab: 4096 parallel drones, domain randomization, PPO         │
│  Output: trained policy (.onnx, 41K params)                         │
├─────────────────────────────────────────────────────────────────────┤
│  Data Alignment Layer  (Sim-to-Real)                                │
│  50hrs real flight data calibrates simulator constants               │
│  System ID: mass, inertia, thrust curves, drag coefficients         │
├─────────────────────────────────────────────────────────────────────┤
│  Edge Execution Layer  (Manifold 3 / Jetson Orin)                   │
│  TensorRT FP16 inference <1ms, PSDK CTBR output at 50-200Hz        │
│  Safety failsafe: NORMAL → CAUTION → OVERRIDE → EMERGENCY          │
└─────────────────────────────────────────────────────────────────────┘
```
""")

st.markdown("---")

st.markdown("""
### Navigate the Demo

Use the **sidebar** to explore:

1. **Training Results** — Simulation videos, training curves, acceptance metrics
2. **System Architecture** — Technical deep-dive, platform specs, research foundation
3. **Mission Planner** — Natural language → 3D cleaning path visualization
""")
